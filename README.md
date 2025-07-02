# TTE

### Key Innovations

1. **System-Algorithm Co-Design**: Joint optimization of training algorithms and system implementation
2. **Quantization-Aware Scaling (QAS)**: Automatic gradient scaling for stable INT8 training
3. **Sparse Update**: Selective parameter updates based on importance analysis
4. **Compile-Time Autodiff**: Moving gradient computation from runtime to compile-time
5. **Multi-Architecture Support**: Both ConvNets (quantized) and Vision Transformers (FP32)

## Architecture

The Tiny Training Engine consists of three main components working together:

### 1. Algorithm Layer (`algorithm/`)
- **Quantization-Aware Scaling**: Stabilizes INT8 gradient computation
- **Sparse Update Strategies**: Memory-efficient parameter selection
- **Multi-Model Support**: ConvNets and Vision Transformers
- **Memory Budget Planning**: Configurable memory constraints

### 2. Compilation System (`compilation/`)
- **IR Translation**: PyTorch → TVM IR → Training Graph
- **Compile-Time Autodiff**: Pre-computed gradient operations
- **Graph Optimization**: Pruning and reordering for efficiency
- **Sparse Integration**: Hardware-aware sparse pattern optimization

### 3. Runtime System
- **TinyEngine Backend**: MCU-optimized execution engine
- **Memory Management**: Dynamic allocation with 256KB budget
- **Kernel Optimization**: Hand-tuned operations for edge devices

## Supported Models

### Quantized ConvNets (INT8 Training)
| Model | Parameters | ImageNet Acc | Memory Budget | Use Case |
|-------|------------|--------------|---------------|----------|
| MobileNetV2-0.35 | 1.7M | 45.7% | 49-138KB | Ultra-low power |
| MCUNet-5fps | 1.1M | 54.1% | 49-148KB | Real-time inference |
| ProxylessNet-0.3 | 2.1M | 48.3% | 49-148KB | Balanced accuracy |

### Vision Transformers (FP32 Training)
| Model | Parameters | Architecture | Memory Budget | Use Case |
|-------|------------|--------------|---------------|----------|
| ViT-Small | 21.7M | 12L, 6H, 384D | 100-200KB | Attention mechanisms |
| ViT-Base | *Future* | 12L, 12H, 768D | *Planned* | Larger capacity |

## System Components

### Algorithm Components (`algorithm/`)

#### Core Architecture
```
algorithm/
├── core/
│   ├── model/           # Model builders (ConvNet + ViT)
│   ├── dataset/         # Data loading and augmentation
│   ├── optimizer/       # SGD with quantization-aware scaling
│   ├── trainer/         # Training loops and validation
│   └── utils/           # Sparse update and memory planning
├── configs/             # YAML configuration files
├── quantize/            # INT8 quantization and operators
└── train_cls.py         # Main training script
```

#### Key Files
- **`train_cls.py`**: Main training entry point with dual model support
- **`core/model/model_entry.py`**: Model factory for ConvNets and ViTs
- **`core/utils/partial_backward.py`**: ConvNet sparse update logic
- **`core/utils/vit_partial_backward.py`**: ViT sparse update logic
- **`quantize/quantized_ops_diff.py`**: Differentiable quantized operators

### Compilation System (`compilation/`)

#### Pipeline Overview
```
compilation/
├── convert/             # Model format converters
│   ├── pth_utils.py     # PyTorch → TVM IR
│   ├── vit_utils.py     # ViT-specific conversions
│   └── mcunetv3_wrapper.py  # Quantized operator wrappers
├── autodiff/            # Compile-time differentiation
│   ├── auto_diff.py     # Main autodiff engine
│   ├── diff_ops.py      # Gradient operator definitions
│   └── mcuop.py         # MCU-optimized operations
├── ir_utils/            # IR manipulation utilities
├── mcu_ir_gen.py        # IR generation script
└── ir2json.py           # JSON export for MCU deployment
```

#### Compilation Flow
1. **Model Loading**: Load PyTorch model with sparse config
2. **IR Translation**: Convert to TVM intermediate representation
3. **Autodiff Generation**: Create backward computation graph
4. **Sparse Optimization**: Apply memory-aware pruning
5. **Code Generation**: Export optimized operators
6. **MCU Packaging**: Bundle for microcontroller deployment

### Model Implementations

#### ConvNet Models (`algorithm/core/ofa_nn/networks/`)
- **MobileNetV2**: Depthwise separable convolutions
- **ProxylessNets**: Neural architecture search optimized
- **MCUNet**: Memory-constraint aware design

#### ViT Implementation (`algorithm/core/ofa_nn/networks/vit_small.py`)
```python
class VisionTransformerSmall(MyNetwork):
    def __init__(self, embed_dim=384, depth=12, num_heads=6):
        self.patch_embed = PatchEmbed(patch_size=16, embed_dim=embed_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(dim=embed_dim, num_heads=num_heads)
            for _ in range(depth)
        ])
        self.head = nn.Linear(embed_dim, num_classes)
    
    def get_sparse_update_layers(self):
        # Returns layer names for sparse targeting
        return ['blocks.*.attn.qkv', 'blocks.*.attn.proj', 'head']
```

## Development Guide

### Code Organization

**Algorithm Development:**
- Models: `algorithm/core/ofa_nn/networks/`
- Training: `algorithm/core/trainer/`
- Sparse Logic: `algorithm/core/utils/*_partial_backward.py`
- Configs: `algorithm/configs/`

**System Development:**
- IR Generation: `compilation/convert/`
- Autodiff: `compilation/autodiff/`
- Optimization: `compilation/ir_utils/`

**Testing:**
- Integration Tests: `test_vit_integration.py`
- Training Examples: `run_vit_training.py`
- Unit Tests: Throughout codebase