from dataclasses import dataclass
from typing import Union, Tuple, Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer import TimmModel, HFTextEncoder

import numpy as np

# Import from local modules (commented out as requested)
# from .model import VisionTransformer, LayerNormFp32, LayerNorm, QuickGELU

@dataclass
class CLIPVisionCfg:
    layers: Union[Tuple[int, int, int, int], int] = 12
    width: int = 768
    head_width: int = 64
    mlp_ratio: float = 4.0
    patch_size: int = 16
    image_size: Union[Tuple[int, int], int] = 224
    ls_init_value: Optional[float] = None  # layer scale initial value
    patch_dropout: float = 0.2  # what fraction of patches to dropout during training (BiomedCLIP uses 0.4)
    input_patchnorm: bool = False  # whether to use dual patchnorm
    global_average_pool: bool = False  # whether to global average pool the last embedding layer
    attentional_pool: bool = False  # whether to use attentional pooler in the last embedding layer
    n_queries: int = 256  # n_queries for attentional pooler
    attn_pooler_heads: int = 8  # n heads for attentional_pooling
    timm_model_name: str = "vit_base_patch16_224"  # a valid model name overrides layers, width, patch_size
    timm_model_pretrained: bool = False  # use (imagenet) pretrained weights for named model (BiomedCLIP uses True)
    timm_pool: str = ""  # feature pooling for timm model
    timm_proj: str = "linear"  # linear projection for timm model output
    timm_proj_bias: bool = False  # enable bias final projection
    timm_drop: float = 0.  # head dropout
    timm_drop_path: Optional[float] = None  # backbone stochastic depth
    output_tokens: bool = True



@dataclass
class CLIPTextCfg:
    context_length: int = 256
    vocab_size: int = 30522  # Adjusted for BiomedBERT
    width: int = 768  # Adjusted for BiomedBERT
    heads: int = 12  # Adjusted for BiomedBERT
    layers: int = 12  # Adjusted for BiomedBERT
    ls_init_value: Optional[float] = None  # layer scale initial value
    hf_model_name: Optional[str] = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
    hf_tokenizer_name: Optional[str] = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
    hf_model_pretrained: bool = True
    proj: str = 'mlp'
    pooler_type: str = 'cls_last_hidden_state_pooler'
    # Add these missing fields for compatibility
    hf_proj_type: str = 'mlp'  # Same as proj, for HF compatibility
    hf_pooler_type: str = 'cls_last_hidden_state_pooler'  # Same as pooler_type
    embed_cls: bool = False
    pad_id: int = 0
    output_tokens: bool = False




def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == 'bf16':
        cast_dtype = torch.bfloat16
    elif precision == 'fp16':
        cast_dtype = torch.float16
    return cast_dtype





def _build_vision_tower(
        embed_dim: int,
        vision_cfg: CLIPVisionCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None
):
    """
    Build vision tower for BiomedCLIP.
    
    Reference: Zhang et al. "BiomedCLIP: a multimodal biomedical foundation model 
    pretrained from fifteen million scientific image-text pairs" (2023)
    Paper: https://arxiv.org/abs/2303.00915
    
    BiomedCLIP uses:
    - ViT-B/16 initialized with ImageNet-pretrained weights (not random)
    - Image resolution: 224x224 (found optimal for biomedical images)
    - Patch dropout: 0.4 for regularization during pretraining
    - Uses timm library for loading pretrained Vision Transformer
    """
    if isinstance(vision_cfg, dict):
        vision_cfg = CLIPVisionCfg(**vision_cfg)

    # Check if using timm model (BiomedCLIP approach)
    if vision_cfg.timm_model_name:
        # Use timm for Vision Transformer with optional ImageNet pretraining
        visual = TimmModel(
            model_name=vision_cfg.timm_model_name,
            embed_dim=embed_dim,
            image_size=vision_cfg.image_size,
            pool=vision_cfg.timm_pool,
            proj=vision_cfg.timm_proj,
            proj_bias=vision_cfg.timm_proj_bias,
            drop=vision_cfg.timm_drop,
            drop_path=vision_cfg.timm_drop_path,
            pretrained=vision_cfg.timm_model_pretrained,
            output_tokens=vision_cfg.output_tokens,
        )
        return visual

    # Fallback to custom Vision Transformer implementation
    act_layer = QuickGELU if quick_gelu else nn.GELU
    norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm

    # Vision Transformer only (ResNet excluded)
    vision_heads = vision_cfg.width // vision_cfg.head_width
    visual = VisionTransformer(
        image_size=vision_cfg.image_size,
        patch_size=vision_cfg.patch_size,
        width=vision_cfg.width,
        layers=vision_cfg.layers,
        heads=vision_heads,
        mlp_ratio=vision_cfg.mlp_ratio,
        ls_init_value=vision_cfg.ls_init_value,
        patch_dropout=vision_cfg.patch_dropout,
        input_patchnorm=vision_cfg.input_patchnorm,
        global_average_pool=vision_cfg.global_average_pool,
        attentional_pool=vision_cfg.attentional_pool,
        n_queries=vision_cfg.n_queries,
        attn_pooler_heads=vision_cfg.attn_pooler_heads,
        output_tokens=vision_cfg.output_tokens,
        output_dim=embed_dim,
        act_layer=act_layer,
        norm_layer=norm_layer,
    )

    return visual


def _build_text_tower(
        embed_dim: int,
        text_cfg: CLIPTextCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None,
):
    """
    Build text tower for BiomedCLIP using BiomedBERT.
    
    Reference: Zhang et al. "BiomedCLIP: a multimodal biomedical foundation model 
    pretrained from fifteen million scientific image-text pairs" (2023)
    Paper: https://arxiv.org/abs/2303.00915
    
    BiomedCLIP uses PubMedBERT (domain-specific pretrained) instead of GPT-2.
    The text encoder is initialized with pretrained weights, not from scratch.
    """
    if isinstance(text_cfg, dict):
        text_cfg = CLIPTextCfg(**text_cfg)

    # Check if we should use HuggingFace pretrained model (BiomedBERT)
    if text_cfg.hf_model_name and text_cfg.hf_model_pretrained:
        # Load pretrained BiomedBERT - this is the BiomedCLIP approach
        # Use the HF-specific parameters
        proj_type = text_cfg.hf_proj_type if hasattr(text_cfg, 'hf_proj_type') else text_cfg.proj
        pooler_type = text_cfg.hf_pooler_type if hasattr(text_cfg, 'hf_pooler_type') else text_cfg.pooler_type
        
        text = HFTextEncoder(
            model_name=text_cfg.hf_model_name,
            output_dim=embed_dim,
            proj_type=proj_type,
            pooler_type=pooler_type,
            pretrained=text_cfg.hf_model_pretrained,
        )
        return text
    
    # Fallback to standard CLIP TextTransformer if not using HF model
    act_layer = QuickGELU if quick_gelu else nn.GELU
    norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm

    text = TextTransformer(
        context_length=text_cfg.context_length,
        vocab_size=text_cfg.vocab_size,
        width=text_cfg.width,
        heads=text_cfg.heads,
        layers=text_cfg.layers,
        ls_init_value=text_cfg.ls_init_value,
        output_dim=embed_dim,
        embed_cls=text_cfg.embed_cls,
        output_tokens=text_cfg.output_tokens,
        pad_id=text_cfg.pad_id,
        act_layer=act_layer,
        norm_layer=norm_layer,
    )
    return text


class CustomTextCLIP(nn.Module):
    output_dict: torch.jit.Final[bool]

    def __init__(
            self,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            text_cfg: CLIPTextCfg,
            quick_gelu: bool = False,
            cast_dtype: Optional[torch.dtype] = None,
            output_dict: bool = False,
    ):
        super().__init__()
        self.output_dict = output_dict
        self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)
        self.text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991 
        self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)

    def lock_text_tower(self, unlocked_layers: int = 0, freeze_layer_norm: bool = True):
        self.text.lock(unlocked_layers, freeze_layer_norm)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.text.set_grad_checkpointing(enable)

    def encode_image(self, image, normalize: bool = False):
        features = self.visual(image)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize: bool = False):
        features = self.text(text)
        return F.normalize(features, dim=-1) if normalize else features

    def forward(self, image, text):
        image_features = self.encode_image(image, normalize=True)
        text_features = self.encode_text(text, normalize=True)
        if self.output_dict:
            return {
                "image_features": image_features,
                "text_features": text_features,
                "logit_scale": self.logit_scale.exp()
            }
        return image_features, text_features, self.logit_scale.exp()





def resize_pos_embed(state_dict, model, interpolation: str = 'bicubic', antialias: bool = True):
    # Rescale the grid of position embeddings when loading from state_dict
    old_pos_embed = state_dict.get('visual.positional_embedding', None)
    if old_pos_embed is None or not hasattr(model.visual, 'grid_size'):
        return

    grid_size = to_2tuple(model.visual.grid_size)
    extra_tokens = 1  # assumes CLS token exists
    new_seq_len = grid_size[0] * grid_size[1] + extra_tokens

    # If already correct length, nothing to do
    if new_seq_len == old_pos_embed.shape[0]:
        return

    if extra_tokens:
        pos_emb_tok, pos_emb_img = old_pos_embed[:extra_tokens], old_pos_embed[extra_tokens:]
    else:
        pos_emb_tok, pos_emb_img = None, old_pos_embed

    old_grid_size = to_2tuple(int(math.sqrt(len(pos_emb_img))))

    logging.info(f"Resizing position embedding grid-size from {old_grid_size} → {grid_size}")

    pos_emb_img = pos_emb_img.reshape(1, old_grid_size[0], old_grid_size[1], -1).permute(0, 3, 1, 2)

    pos_emb_img = F.interpolate(
        pos_emb_img,
        size=grid_size,
        mode=interpolation,
        antialias=antialias,
    )

    pos_emb_img = pos_emb_img.permute(0, 2, 3, 1).reshape(1, grid_size[0] * grid_size[1], -1)[0]

    if pos_emb_tok is not None:
        new_pos_embed = torch.cat([pos_emb_tok, pos_emb_img], dim=0)
    else:
        new_pos_embed = pos_emb_img

    state_dict['visual.positional_embedding'] = new_pos_embed
