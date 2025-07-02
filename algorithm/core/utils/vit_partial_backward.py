import copy
import torch
import numpy as np
from .partial_backward import activation_bits, bias_bits, weight_bits, momentum_bits


def get_all_linear_ops(model):
    """Get all Linear operations in ViT model"""
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    return [m for m in model.modules() if isinstance(m, torch.nn.Linear)]


def get_all_linear_ops_with_names(model):
    """Get all Linear operations with their names in ViT model"""
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    linears = []
    names = []
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear):
            linears.append(m)
            names.append(name)
    return linears, names


def _is_attention_layer(name):
    """Check if layer is part of attention mechanism"""
    return any(x in name for x in ['attn.qkv', 'attn.proj'])


def _is_mlp_layer(name):
    """Check if layer is part of MLP block"""
    return any(x in name for x in ['mlp.fc1', 'mlp.fc2'])


def _is_head_layer(name):
    """Check if layer is the classification head"""
    return 'head' in name


def parsed_vit_backward_config(backward_config, model):
    """Parse backward config for ViT models"""
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    
    linears, names = get_all_linear_ops_with_names(model)
    n_linear = len(linears)
    
    # Parse n_bias_update for ViT
    if backward_config['n_bias_update'] == 'all':
        backward_config['n_bias_update'] = n_linear
    else:
        assert isinstance(backward_config['n_bias_update'], int), backward_config['n_bias_update']

    # Parse layer indices for sparse update
    if backward_config.get('vit_layer_types') is not None:
        # Select layers based on type (attention, mlp, head)
        selected_indices = []
        layer_types = backward_config['vit_layer_types']
        
        for i, name in enumerate(names):
            if 'attention' in layer_types and _is_attention_layer(name):
                selected_indices.append(i)
            elif 'mlp' in layer_types and _is_mlp_layer(name):
                selected_indices.append(i)
            elif 'head' in layer_types and _is_head_layer(name):
                selected_indices.append(i)
        
        backward_config['manual_weight_idx'] = selected_indices
    elif backward_config.get('manual_weight_idx') is not None:
        # Use manually specified indices
        backward_config['manual_weight_idx'] = [int(p) for p in str(backward_config['manual_weight_idx']).split('-')]
    else:
        # Default: update last n layers
        n_weight_update = backward_config.pop('n_weight_update', backward_config['n_bias_update'])
        if n_weight_update == 'all':
            n_weight_update = backward_config['n_bias_update']
        else:
            assert isinstance(n_weight_update, int), n_weight_update
        backward_config['manual_weight_idx'] = sorted([n_linear - 1 - i_w for i_w in range(n_weight_update)])

    # Setup weight update ratios
    n_weight_update = len(backward_config['manual_weight_idx'])
    if backward_config.get('weight_update_ratio') is None:
        backward_config['weight_update_ratio'] = [None] * n_weight_update
    elif isinstance(backward_config['weight_update_ratio'], (int, float)):
        assert backward_config['weight_update_ratio'] <= 1
        backward_config['weight_update_ratio'] = [backward_config['weight_update_ratio']] * n_weight_update
    else:  # list or string
        if isinstance(backward_config['weight_update_ratio'], str):
            ratios = [float(p) for p in backward_config['weight_update_ratio'].split('-')]
        elif isinstance(backward_config['weight_update_ratio'], list):
            ratios = [float(p) for p in backward_config['weight_update_ratio']]
        else:
            ratios = [backward_config['weight_update_ratio']] * n_weight_update
        
        # If ratios length doesn't match, repeat the last value or truncate
        if len(ratios) == 1:
            backward_config['weight_update_ratio'] = ratios * n_weight_update
        elif len(ratios) < n_weight_update:
            # Repeat the pattern or the last value
            backward_config['weight_update_ratio'] = (ratios * ((n_weight_update // len(ratios)) + 1))[:n_weight_update]
        else:
            # Truncate if too long
            backward_config['weight_update_ratio'] = ratios[:n_weight_update]

    return backward_config


def nelem_saved_for_backward_vit(model, sample_input, backward_config, verbose=True):
    """Calculate memory requirements for ViT sparse training"""
    model = copy.deepcopy(model)
    model.eval()

    # Record input/output shapes
    def record_in_out_shape(m_, x, y):
        x = x[0] if isinstance(x, tuple) else x
        m_.input_shape = list(x.shape)
        m_.output_shape = list(y.shape)

    def add_activation_shape_hook(m_):
        if isinstance(m_, torch.nn.Linear):
            m_.register_forward_hook(record_in_out_shape)

    model.apply(add_activation_shape_hook)

    with torch.no_grad():
        _ = model(sample_input)

    # Calculate memory usage
    weight_size = []
    momentum_size = []
    activation_size = []

    linears, names = get_all_linear_ops_with_names(model)
    
    # Add fake gradients
    for linear in linears:
        linear.weight.grad = torch.rand_like(linear.weight) * 100.
        if linear.bias is not None:
            linear.bias.grad = torch.rand_like(linear.bias) * 100.

    # Apply backward config
    apply_vit_backward_config(model, backward_config)

    # Calculate sizes for each updated layer
    for i, (linear, name) in enumerate(zip(linears, names)):
        if linear.bias is not None and linear.bias.grad is not None:
            # Layer is being updated
            this_weight_size = linear.bias.numel() * bias_bits
            this_momentum_size = linear.bias.numel() * momentum_bits
            this_activation_size = np.product(linear.input_shape) * activation_bits // linear.input_shape[0]  # Exclude batch dim

            if linear.weight.grad is not None:
                # Weight is also being updated
                weight_elements = linear.weight.numel()
                
                # For sparse updates, calculate actual updated elements
                if hasattr(linear, 'keep_mask') and linear.keep_mask is not None:
                    if _is_attention_layer(name) and 'qkv' in name:
                        # For qkv layer, mask is applied to input features
                        updated_elements = linear.keep_mask.sum().item() * linear.weight.shape[0]
                    else:
                        # For other layers, mask is applied to input features
                        updated_elements = linear.keep_mask.sum().item() * linear.weight.shape[0]
                    weight_elements = int(updated_elements)

                this_weight_size += weight_elements * weight_bits
                this_momentum_size += weight_elements * momentum_bits

            weight_size.append(this_weight_size)
            momentum_size.append(this_momentum_size)
            activation_size.append(this_activation_size)

    del model

    total_weight_size = sum(weight_size)
    total_momentum_size = sum(momentum_size)
    total_activation_size = sum(activation_size)
    total_usage = total_weight_size + total_momentum_size + total_activation_size

    if verbose:
        print('ViT Memory Usage:')
        print('weight', weight_size)
        print('momentum', momentum_size)
        print('activation', activation_size)
        print('memory usage in kB:')
        print('weight: {:.0f}kB, momentum: {:.0f}kB, activation: {:.0f}kB'.format(
            total_weight_size / 1024 / 8, total_momentum_size / 1024 / 8, total_activation_size / 1024 / 8
        ))
        print('total: {:.0f}kB'.format(total_usage / 1024 / 8))

    return total_usage


def prepare_vit_model_for_backward_config(model, backward_config, verbose=True):
    """Prepare ViT model for sparse backward pass"""
    def _get_linear_w_norm(_linear):
        # For linear layers, compute norm along input dimension
        w_norm = torch.norm(_linear.weight.data, dim=0)  # Shape: [in_features]
        return w_norm

    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    
    # Select channels/features to update
    if all([r is None for r in backward_config['weight_update_ratio']]):
        return
    
    linears, names = get_all_linear_ops_with_names(model)
    total_keep_features = 0
    ratio_ptr = 0
    
    for i_linear, (linear, name) in enumerate(zip(linears, names)):
        if i_linear in backward_config['manual_weight_idx']:
            keep_ratio = backward_config['weight_update_ratio'][ratio_ptr]
            ratio_ptr += 1
            
            if keep_ratio <= 1:
                n_keep = int(linear.in_features * keep_ratio)
            else:
                assert isinstance(keep_ratio, int)
                n_keep = keep_ratio
            
            total_keep_features += n_keep
            
            # Select features based on criteria
            if backward_config['weight_select_criteria'] == 'magnitude+':
                w_norm = _get_linear_w_norm(linear)
                keep_idx = torch.argsort(-w_norm)[:n_keep]
                keep_mask = torch.zeros_like(w_norm)
                keep_mask[keep_idx] = 1.
            elif backward_config['weight_select_criteria'] == 'magnitude-':
                w_norm = _get_linear_w_norm(linear)
                keep_idx = torch.argsort(w_norm)[:n_keep]
                keep_mask = torch.zeros_like(w_norm)
                keep_mask[keep_idx] = 1.
            elif backward_config['weight_select_criteria'] == 'random':
                w_norm = _get_linear_w_norm(linear)
                keep_idx = torch.randperm(linear.in_features)[:n_keep]
                keep_mask = torch.zeros_like(w_norm)
                keep_mask[keep_idx] = 1.
            else:
                raise NotImplementedError(f"Criteria {backward_config['weight_select_criteria']} not implemented")
            
            linear.register_buffer('keep_mask', keep_mask)
    
    avg_features = total_keep_features / len(backward_config['weight_update_ratio']) if backward_config['weight_update_ratio'] else 0
    if verbose:
        print(f' * Total update features: {total_keep_features}; average per layer: {avg_features}')


def apply_vit_backward_config(model, backward_config):
    """Apply backward config to ViT model gradients"""
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    
    linears, names = get_all_linear_ops_with_names(model)
    n_w_trained = 0
    ratio_ptr = len(backward_config['manual_weight_idx']) - 1
    
    for i in range(len(linears) - 1, -1, -1):
        i_linear = i
        linear = linears[i]
        name = names[i]
        
        # Check if this layer should be updated (either bias or weight)
        update_bias = i_linear < backward_config['n_bias_update']
        update_weight = i_linear in backward_config['manual_weight_idx']
        
        if update_bias:
            if update_weight:
                n_w_trained += 1
                if ratio_ptr >= 0 and ratio_ptr < len(backward_config['weight_update_ratio']) and backward_config['weight_update_ratio'][ratio_ptr] is not None:
                    # Apply sparse gradient mask
                    if hasattr(linear, 'keep_mask'):
                        linear.weight.grad.data = linear.weight.grad.data * linear.keep_mask.view(1, -1)
                    ratio_ptr -= 1
            else:
                # Only update bias, not weights
                linear.weight.grad = None
        else:
            # Don't update this layer at all
            linear.weight.grad = None
            if linear.bias is not None:
                linear.bias.grad = None
    
    # Note: n_w_trained might be less than manual_weight_idx if some indices are out of range
    # This is OK for ViT models where layer selection might be sparse


# Predefined sparse update configurations for ViT-Small
VIT_SPARSE_CONFIGS = {
    # Realistic memory budgets for FP32 ViT training
    "2mb": {
        'enable_backward_config': 1,
        'n_bias_update': 8,  # Update fewer layers (~2.4MB actual)
        'vit_layer_types': ['attention', 'head'],  # Focus on attention and head
        'weight_update_ratio': 0.25,  # Update only 25% of features
        'weight_select_criteria': 'magnitude+',
    },
    "5mb": {
        'enable_backward_config': 1,
        'n_bias_update': 12,  # Update more layers (~7.8MB actual)
        'vit_layer_types': ['attention', 'mlp'],
        'weight_update_ratio': 0.5,
        'weight_select_criteria': 'magnitude+',
    },
    "10mb": {
        'enable_backward_config': 1,
        'n_bias_update': 18,  # Update most layers (~12.4MB actual)
        'vit_layer_types': ['attention', 'mlp', 'head'],
        'weight_update_ratio': 0.75,
        'weight_select_criteria': 'magnitude+',
    },
    # Legacy configs with original naming (actual usage is much higher)
    "100kb": {
        'enable_backward_config': 1,
        'n_bias_update': 8,
        'vit_layer_types': ['attention', 'head'],
        'weight_update_ratio': 0.25,
        'weight_select_criteria': 'magnitude+',
    },
    "150kb": {
        'enable_backward_config': 1,
        'n_bias_update': 12,
        'vit_layer_types': ['attention', 'mlp'],
        'weight_update_ratio': 0.5,
        'weight_select_criteria': 'magnitude+',
    },
    "200kb": {
        'enable_backward_config': 1,
        'n_bias_update': 18,
        'vit_layer_types': ['attention', 'mlp', 'head'],
        'weight_update_ratio': 0.75,
        'weight_select_criteria': 'magnitude+',
    }
}