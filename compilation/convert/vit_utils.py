import torch
import torch.nn as nn
import tvm
from tvm import relay
import numpy as np


def convert_linear_layer(idx, linear_layer, input_tensor, prefix=""):
    """Convert PyTorch Linear layer to TVM relay"""
    weight = linear_layer.weight.data.numpy()
    bias = linear_layer.bias.data.numpy() if linear_layer.bias is not None else None
    
    # Create weight parameter
    weight_var = relay.var(f"{prefix}weight_{idx}", shape=weight.shape, dtype="float32")
    weight_param = {f"{prefix}weight_{idx}": tvm.nd.array(weight)}
    
    args = [weight_var]
    params = weight_param
    
    # Linear transformation: input @ weight.T + bias
    out = relay.nn.dense(input_tensor, weight_var, units=weight.shape[0])
    
    if bias is not None:
        bias_var = relay.var(f"{prefix}bias_{idx}", shape=bias.shape, dtype="float32")
        bias_param = {f"{prefix}bias_{idx}": tvm.nd.array(bias)}
        args.append(bias_var)
        params.update(bias_param)
        out = relay.nn.bias_add(out, bias_var)
    
    return out, args, params


def convert_layer_norm(idx, norm_layer, input_tensor, prefix=""):
    """Convert PyTorch LayerNorm to TVM relay"""
    weight = norm_layer.weight.data.numpy()
    bias = norm_layer.bias.data.numpy()
    eps = norm_layer.eps
    
    weight_var = relay.var(f"{prefix}ln_weight_{idx}", shape=weight.shape, dtype="float32")
    bias_var = relay.var(f"{prefix}ln_bias_{idx}", shape=bias.shape, dtype="float32")
    
    weight_param = {f"{prefix}ln_weight_{idx}": tvm.nd.array(weight)}
    bias_param = {f"{prefix}ln_bias_{idx}": tvm.nd.array(bias)}
    
    args = [weight_var, bias_var]
    params = {**weight_param, **bias_param}
    
    # LayerNorm implementation
    out = relay.nn.layer_norm(input_tensor, weight_var, bias_var, axis=-1, epsilon=eps)
    
    return out, args, params


def convert_gelu_activation(input_tensor):
    """Convert GELU activation to TVM relay"""
    # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    x = input_tensor
    
    # Constants
    sqrt_2_over_pi = relay.const(np.sqrt(2.0 / np.pi), dtype="float32")
    coeff = relay.const(0.044715, dtype="float32")
    half = relay.const(0.5, dtype="float32")
    one = relay.const(1.0, dtype="float32")
    
    # x^3
    x_cubed = relay.power(x, relay.const(3.0, dtype="float32"))
    
    # 0.044715 * x^3
    term = relay.multiply(coeff, x_cubed)
    
    # x + 0.044715 * x^3
    inner = relay.add(x, term)
    
    # sqrt(2/π) * (x + 0.044715 * x^3)
    scaled = relay.multiply(sqrt_2_over_pi, inner)
    
    # tanh(...)
    tanh_term = relay.tanh(scaled)
    
    # 1 + tanh(...)
    one_plus_tanh = relay.add(one, tanh_term)
    
    # x * (1 + tanh(...))
    x_times_term = relay.multiply(x, one_plus_tanh)
    
    # 0.5 * x * (1 + tanh(...))
    out = relay.multiply(half, x_times_term)
    
    return out


def convert_attention_block(idx, attn_module, input_tensor, prefix=""):
    """Convert ViT attention block to TVM relay (simplified version)"""
    # For now, we'll convert attention as a series of linear layers
    # This is a simplified version - full attention would require more complex operations
    
    # QKV projection
    qkv_out, qkv_args, qkv_params = convert_linear_layer(
        f"{idx}_qkv", attn_module.qkv, input_tensor, prefix
    )
    
    # For simplicity, we'll skip the actual attention computation
    # and just apply the output projection
    proj_out, proj_args, proj_params = convert_linear_layer(
        f"{idx}_proj", attn_module.proj, qkv_out, prefix
    )
    
    all_args = qkv_args + proj_args
    all_params = {**qkv_params, **proj_params}
    
    return proj_out, all_args, all_params


def convert_mlp_block(idx, mlp_module, input_tensor, prefix=""):
    """Convert ViT MLP block to TVM relay"""
    # FC1
    fc1_out, fc1_args, fc1_params = convert_linear_layer(
        f"{idx}_fc1", mlp_module.fc1, input_tensor, prefix
    )
    
    # GELU activation
    gelu_out = convert_gelu_activation(fc1_out)
    
    # FC2
    fc2_out, fc2_args, fc2_params = convert_linear_layer(
        f"{idx}_fc2", mlp_module.fc2, gelu_out, prefix
    )
    
    all_args = fc1_args + fc2_args
    all_params = {**fc1_params, **fc2_params}
    
    return fc2_out, all_args, all_params


def convert_transformer_block(idx, block_module, input_tensor, prefix=""):
    """Convert ViT transformer block to TVM relay"""
    # Pre-norm 1
    norm1_out, norm1_args, norm1_params = convert_layer_norm(
        f"{idx}_norm1", block_module.norm1, input_tensor, prefix
    )
    
    # Attention
    attn_out, attn_args, attn_params = convert_attention_block(
        f"{idx}_attn", block_module.attn, norm1_out, prefix
    )
    
    # Residual connection
    residual1 = relay.add(input_tensor, attn_out)
    
    # Pre-norm 2
    norm2_out, norm2_args, norm2_params = convert_layer_norm(
        f"{idx}_norm2", block_module.norm2, residual1, prefix
    )
    
    # MLP
    mlp_out, mlp_args, mlp_params = convert_mlp_block(
        f"{idx}_mlp", block_module.mlp, norm2_out, prefix
    )
    
    # Residual connection
    out = relay.add(residual1, mlp_out)
    
    all_args = norm1_args + attn_args + norm2_args + mlp_args
    all_params = {**norm1_params, **attn_params, **norm2_params, **mlp_params}
    
    return out, all_args, all_params


def vit_model_to_ir(model, input_shape=[1, 3, 224, 224]):
    """Convert ViT model to TVM IR (simplified version for demonstration)"""
    data = relay.var("input", shape=input_shape, dtype="float32")
    
    tot_args = [data]
    tot_params = {}
    out = data
    
    # This is a simplified conversion - a full implementation would need:
    # 1. Patch embedding conversion
    # 2. Position embedding handling
    # 3. Full attention mechanism
    # 4. Proper tensor reshaping for attention
    
    # For now, we'll just handle the final classification head
    if hasattr(model, 'head') and isinstance(model.head, nn.Linear):
        # Assume input is already flattened
        head_out, head_args, head_params = convert_linear_layer(
            "head", model.head, out, "final_"
        )
        
        tot_args += head_args
        tot_params.update(head_params)
        out = head_out
    
    expr = relay.Function(tot_args, out)
    mod = tvm.IRModule.from_expr(expr)
    mod = relay.transform.InferType()(mod)
    
    return mod, tot_params, 0


def build_vit_training_ir(model, sparse_config, input_shape=[1, 3, 224, 224]):
    """Build training IR for ViT with sparse update configuration"""
    # This would integrate with the autodiff system
    # For now, return the forward IR
    return vit_model_to_ir(model, input_shape)