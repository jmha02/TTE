import torch
from core.utils.config import configs 
from quantize.custom_quantized_format import build_quantized_network_from_cfg
from quantize.quantize_helper import create_scaled_head, create_quantized_head
from ..ofa_nn.networks import vit_small_patch16_224

__all__ = ['build_mcu_model', 'build_fp_model']


def build_mcu_model():
    cfg_path = f"assets/mcu_models/{configs.net_config.net_name}.pkl"
    cfg = torch.load(cfg_path)
    
    model = build_quantized_network_from_cfg(cfg, n_bit=8)

    if configs.net_config.mcu_head_type == 'quantized':
        model = create_quantized_head(model)
    elif configs.net_config.mcu_head_type == 'fp':
        model = create_scaled_head(model, norm_feat=False)
    else:
        raise NotImplementedError

    return model


def build_fp_model():
    """Build floating-point models (non-quantized) like ViT"""
    if configs.net_config.net_name == 'vit_small_patch16_224':
        model = vit_small_patch16_224(
            num_classes=configs.data_provider.num_classes,
            img_size=configs.data_provider.image_size
        )
    else:
        raise NotImplementedError(f"FP model {configs.net_config.net_name} not supported")
    
    return model
