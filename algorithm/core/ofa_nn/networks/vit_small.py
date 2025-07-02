import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import MyNetwork


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)  # B, N, C
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformerSmall(MyNetwork):
    """Vision Transformer Small - Non-quantized version for sparse training"""
    
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, 
                 embed_dim=384, depth=12, num_heads=6, mlp_ratio=4., qkv_bias=True, 
                 drop_rate=0., attn_drop_rate=0.):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate)
            for _ in range(depth)])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # Initialize weights
        torch.nn.init.trunc_normal_(self.pos_embed, std=.02)
        torch.nn.init.trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        x = x[:, 0]  # Use CLS token
        x = self.head(x)
        return x

    @property
    def module_str(self):
        return f'VisionTransformerSmall(embed_dim={self.embed_dim}, depth={len(self.blocks)}, num_heads={self.blocks[0].attn.num_heads})'

    @property
    def config(self):
        return {
            'name': 'VisionTransformerSmall',
            'embed_dim': self.embed_dim,
            'depth': len(self.blocks),
            'num_heads': self.blocks[0].attn.num_heads,
            'num_classes': self.num_classes,
        }

    @staticmethod
    def build_from_config(config):
        return VisionTransformerSmall(
            num_classes=config.get('num_classes', 1000),
            embed_dim=config.get('embed_dim', 384),
            depth=config.get('depth', 12),
            num_heads=config.get('num_heads', 6),
        )

    def zero_last_gamma(self):
        # No batch norm gamma to zero in ViT
        pass

    @property
    def grouped_block_index(self):
        # Group transformer blocks for sparse update
        return [[i] for i in range(len(self.blocks))]

    def get_sparse_update_layers(self):
        """Get layer names suitable for sparse update"""
        layers = []
        
        # Patch embedding
        layers.append('patch_embed.proj')
        
        # Transformer blocks - focus on attention layers
        for i, block in enumerate(self.blocks):
            layers.extend([
                f'blocks.{i}.attn.qkv',
                f'blocks.{i}.attn.proj', 
                f'blocks.{i}.mlp.fc1',
                f'blocks.{i}.mlp.fc2'
            ])
        
        # Classification head
        layers.append('head')
        
        return layers


def vit_small_patch16_224(num_classes=1000, **kwargs):
    """ViT-Small model (ViT-S/16) from original paper"""
    model = VisionTransformerSmall(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, num_classes=num_classes, **kwargs)
    return model