import os
import argparse
import random
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image


# Residual Adapter for BiomedCLIP
class BiomedClipAdapter(nn.Module):
    """
    Adapter module for BiomedCLIP.
    
    Modified from standard CLIP adapter to handle:
    - BiomedCLIP's different feature dimensions (768 for ViT-B/16)
    - Domain-specific biomedical features
    - Potentially different bottleneck dimensions
    
    Reference: Based on adapter architecture for vision-language models
    """
    def __init__(self, c_in, bottleneck=768):
        super(BiomedClipAdapter, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(c_in, bottleneck, bias=False),
            nn.LeakyReLU(inplace=False)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(bottleneck, c_in, bias=False),
            nn.LeakyReLU(inplace=False)
        )
        
        # Initialize adapter parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """
        Initialize adapter parameters.
        
        Uses small initialization to start close to identity mapping,
        allowing the pretrained BiomedCLIP features to dominate initially.
        """
        for module in self.fc1:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.01)
        
        for module in self.fc2:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.01)
    
    def forward(self, x):
        x = self.fc1(x)
        y = self.fc2(x)
        return x, y

class BiomedCLIP_Inplanted(nn.Module):
    """
    BiomedCLIP with implanted adapters for segmentation and detection tasks.
    """
    def __init__(self, biomedclip_model, features, feature_dim=768, embed_dim=512):
        """
        Initialize BiomedCLIP with adapters.
        
        Args:
            biomedclip_model: The BiomedCLIP model
            features: List of layer indices to extract features from (e.g., [3, 6, 9, 12])
            feature_dim: Feature dimension from vision encoder (768 for ViT-B/16)
            embed_dim: Shared embedding dimension (512 for BiomedCLIP)
        """
        super().__init__()
        self.clipmodel = biomedclip_model
        self.image_encoder = biomedclip_model.visual
        self.features = features
        self.feature_dim = feature_dim
        self.embed_dim = embed_dim
        
        # Check if using timm model or custom VisionTransformer
        self.is_timm_model = hasattr(self.image_encoder, 'trunk')
        
        # Create adapters for each feature layer
        # BiomedCLIP ViT-B/16 uses 768-dim features
        self.seg_adapters = nn.ModuleList(
            [BiomedClipAdapter(feature_dim, bottleneck=768) for i in range(len(features))]
        )
        self.det_adapters = nn.ModuleList(
            [BiomedClipAdapter(feature_dim, bottleneck=768) for i in range(len(features))]
        )
        
        # ✅ ADD PROJECTION LAYERS: Project from bottleneck (768) to embed_dim (512)
        # This aligns vision features with text features in shared embedding space
        self.seg_projections = nn.ModuleList([
            nn.Linear(768, embed_dim, bias=False) for _ in range(len(features))
        ])
        self.det_projections = nn.ModuleList([
            nn.Linear(768, embed_dim, bias=False) for _ in range(len(features))
        ])
        
        # Initialize projections
        for proj in self.seg_projections:
            nn.init.normal_(proj.weight, std=0.01)
        for proj in self.det_projections:
            nn.init.normal_(proj.weight, std=0.01)
    
    def forward_timm(self, x):
        """Forward pass for timm-based BiomedCLIP model."""
        trunk = self.image_encoder.trunk
        
        # Patch embedding
        if hasattr(trunk, 'patch_embed'):
            x = trunk.patch_embed(x)
        else:
            x = trunk.conv_proj(x) if hasattr(trunk, 'conv_proj') else trunk.stem(x)
            x = x.flatten(2).transpose(1, 2)
        
        # Add position embeddings and CLS token
        if hasattr(trunk, 'cls_token'):
            cls_token = trunk.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
        
        if hasattr(trunk, 'pos_embed'):
            x = x + trunk.pos_embed
        
        if hasattr(trunk, 'pos_drop'):
            x = trunk.pos_drop(x)
        
        # Process through transformer blocks with adapters
        seg_patch_tokens = []
        det_patch_tokens = []
        
        blocks = trunk.blocks if hasattr(trunk, 'blocks') else trunk.layers
        
        for i, block in enumerate(blocks):
            x = block(x)
            
            # Extract features at specified layers
            if (i + 1) in self.features:
                idx = self.features.index(i + 1)
                seg_adapt_med, seg_adapt_out = self.seg_adapters[idx](x)
                det_adapt_med, det_adapt_out = self.det_adapters[idx](x)
                
                # Residual connection with adapters
                x = 0.8 * x + 0.1 * seg_adapt_out + 0.1 * det_adapt_out
                
                # ✅ PROJECT TO EMBED_DIM (512)
                seg_adapt_med_proj = self.seg_projections[idx](seg_adapt_med)
                det_adapt_med_proj = self.det_projections[idx](det_adapt_med)
                
                seg_patch_tokens.append(seg_adapt_med_proj)
                det_patch_tokens.append(det_adapt_med_proj)
        
        # Apply final norm
        if hasattr(trunk, 'norm'):
            x = trunk.norm(x)
        
        # Extract CLS token and project
        pooled = x[:, 0]
        pooled = self.image_encoder.head_drop(pooled)
        pooled = self.image_encoder.proj(pooled)
        
        return pooled, seg_patch_tokens, det_patch_tokens
    
    def forward_custom_vit(self, x):
        """Forward pass for custom VisionTransformer implementation."""
        # Patch embedding
        if self.image_encoder.input_patchnorm:
            x = x.reshape(x.shape[0], x.shape[1], self.image_encoder.grid_size[0], 
                         self.image_encoder.patch_size[0], self.image_encoder.grid_size[1], 
                         self.image_encoder.patch_size[1])
            x = x.permute(0, 2, 4, 1, 3, 5)
            x = x.reshape(x.shape[0], self.image_encoder.grid_size[0] * self.image_encoder.grid_size[1], -1)
            x = self.image_encoder.patchnorm_pre_ln(x)
            x = self.image_encoder.conv1(x)
        else:
            x = self.image_encoder.conv1(x)
            x = x.reshape(x.shape[0], x.shape[1], -1)
            x = x.permute(0, 2, 1)
        
        # Add class embedding and positional embedding
        x = torch.cat(
            [self.image_encoder.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], 
             dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.image_encoder.positional_embedding.to(x.dtype)
        
        # Apply patch dropout
        x = self.image_encoder.patch_dropout(x)
        x = self.image_encoder.ln_pre(x)
        
        # Permute for transformer
        x = x.permute(1, 0, 2)  # NLD -> LND
        
        # Process through transformer blocks with adapters
        attn_out = []
        seg_patch_tokens = []
        det_patch_tokens = []
        
        num_blocks = len(self.image_encoder.transformer.resblocks)
        
        for i in range(num_blocks):
            if i + 1 == num_blocks:
                x, attn = self.image_encoder.transformer.resblocks[i](x, attn_mask=None)
                attn_out.append(attn)
            else:
                x, attn_map = self.image_encoder.transformer.resblocks[i](x, attn_mask=None)
            
            # Apply adapters at specified layers
            if (i + 1) in self.features:
                idx = self.features.index(i + 1)
                seg_adapt_med, seg_adapt_out = self.seg_adapters[idx](x)
                det_adapt_med, det_adapt_out = self.det_adapters[idx](x)
                
                # Residual connection with adapters
                x = 0.8 * x + 0.1 * seg_adapt_out + 0.1 * det_adapt_out
                
                # ✅ PROJECT TO EMBED_DIM (512)
                seg_adapt_med_proj = self.seg_projections[idx](seg_adapt_med)
                det_adapt_med_proj = self.det_projections[idx](det_adapt_med)
                
                seg_patch_tokens.append(seg_adapt_med_proj)
                det_patch_tokens.append(det_adapt_med_proj)
        
        # Permute back
        x = x.permute(1, 0, 2)  # LND -> NLD
        seg_patch_tokens = [seg_patch_tokens[t].permute(1, 0, 2) for t in range(len(seg_patch_tokens))]
        det_patch_tokens = [det_patch_tokens[t].permute(1, 0, 2) for t in range(len(det_patch_tokens))]
        
        # Global pooling and projection
        pooled, tokens = self.image_encoder._global_pool(x)
        pooled = self.image_encoder.ln_post(pooled)
        
        if self.image_encoder.proj is not None:
            pooled = pooled @ self.image_encoder.proj
        
        return pooled, seg_patch_tokens, det_patch_tokens
    
    def forward(self, x):
        """Forward pass through BiomedCLIP with adapters."""
        if self.is_timm_model:
            return self.forward_timm(x)
        else:
            return self.forward_custom_vit(x)


            
def create_biomedclip_with_adapters(biomedclip_model, features=[3, 6, 9, 12], feature_dim=768, embed_dim=512):
    """
    Create BiomedCLIP model with adapters for downstream tasks.
    
    Args:
        biomedclip_model: Pretrained BiomedCLIP model
        features: List of layer indices to extract and adapt
        feature_dim: Feature dimension (768 for ViT-B/16)
        embed_dim: Shared embedding dimension (512 for BiomedCLIP)
    """
    return BiomedCLIP_Inplanted(biomedclip_model, features, feature_dim, embed_dim)