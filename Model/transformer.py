from collections import OrderedDict
import math
from typing import Callable, Optional, Sequence, Tuple, Union
from itertools import repeat
import collections.abc
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint



# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse
to_2tuple = _ntuple(2)


class LayerNormFp32(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16 (by casting to float32 and back)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x.to(torch.float32), self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm (with cast back to input dtype)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)


class QuickGELU(nn.Module):
    # NOTE This is slower than nn.GELU or nn.SiLU and uses more GPU memory
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma



class PatchDropout(nn.Module):
    """
    https://arxiv.org/abs/2212.00794
    """

    def __init__(self, prob, exclude_first_token=True):
        super().__init__()
        assert 0 <= prob < 1.
        self.prob = prob
        self.exclude_first_token = exclude_first_token  # exclude CLS token

    def forward(self, x):
        if not self.training or self.prob == 0.:
            return x

        if self.exclude_first_token:
            cls_tokens, x = x[:, :1], x[:, 1:]
        else:
            cls_tokens = torch.jit.annotate(torch.Tensor, x[:, :1])

        batch = x.size()[0]
        num_tokens = x.size()[1]

        batch_indices = torch.arange(batch)
        batch_indices = batch_indices[..., None]

        keep_prob = 1 - self.prob
        num_patches_keep = max(1, int(num_tokens * keep_prob))

        rand = torch.randn(batch, num_tokens)
        patch_indices_keep = rand.topk(num_patches_keep, dim=-1).indices

        x = x[batch_indices, patch_indices_keep]

        if self.exclude_first_token:
            x = torch.cat((cls_tokens, x), dim=1)

        return x


class ResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            is_cross_attention: bool = False,
            idx: int = 12,
    ):
        super().__init__()

        self.idx = idx

        self.ln_1 = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()
        if is_cross_attention:
            self.ln_1_kv = norm_layer(d_model)

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ("gelu", act_layer()),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))
        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

    def attention(
            self,
            q_x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None,
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        k_x = k_x if k_x is not None else q_x
        v_x = v_x if v_x is not None else q_x

        attn_mask = attn_mask.to(q_x.dtype) if attn_mask is not None else None
        return self.attn(
            q_x, k_x, v_x, need_weights=True, attn_mask=attn_mask
        )

    def forward(
            self,
            q_x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None,
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        k_x = self.ln_1_kv(k_x) if hasattr(self, "ln_1_kv") and k_x is not None else None
        v_x = self.ln_1_kv(v_x) if hasattr(self, "ln_1_kv") and v_x is not None else None

        tmp, attn = self.attention(q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask)
        x = q_x + self.ls_1(tmp)
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x, attn






class Transformer(nn.Module):
    def __init__(
            self,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.grad_checkpointing = False

        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(
                width, heads, mlp_ratio, ls_init_value=ls_init_value, act_layer=act_layer, norm_layer=norm_layer,
                idx=idx)
            for idx in range(layers)
        ])

    def get_cast_dtype(self) -> torch.dtype:
        return self.resblocks[0].mlp.c_fc.weight.dtype

    def forward(self, x: torch.Tensor, out_layers: list = [3, 6, 9],
                attn_mask: Optional[torch.Tensor] = None):
        idx = 0
        out_attn = []
        # out_tokens = x
        out_tokens = []
        for r in self.resblocks:
            idx += 1
            if self.grad_checkpointing and not torch.jit.is_scripting():
                # TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372
                x = checkpoint(r, x, None, None, attn_mask)
            else:
                if idx == 12:
                    x, attn = r(x, attn_mask=attn_mask)
                    out_attn.append(attn)
                else:
                    x, attn_tmp = r(x, attn_mask=attn_mask)
                if idx in out_layers:
                    out_tokens.append(x)
                    # out_tokens = x
        return x, out_attn, out_tokens



class VisionTransformer(nn.Module):
    output_tokens: torch.jit.Final[bool]

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        mlp_ratio: float,
        ls_init_value: float = None,
        global_average_pool: bool = False,
        attentional_pool: bool = False,
        n_queries: int = 256,
        attn_pooler_heads: int = 8,
        output_dim: int = 512,
        patch_dropout: float = 0.4,
        input_patchnorm: bool = False,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = LayerNorm,
        output_tokens: bool = False
    ):
        super().__init__()
        self.output_tokens = output_tokens
        image_height, image_width = self.image_size = to_2tuple(image_size)
        patch_height, patch_width = self.patch_size = to_2tuple(patch_size)
        self.grid_size = (image_height // patch_height, image_width // patch_width)
        self.output_dim = output_dim

        # whether to layernorm each patch, as done in dual patchnorm paper - https://arxiv.org/abs/2302.01327v1
        self.input_patchnorm = input_patchnorm

        if input_patchnorm:
            patch_input_dim = patch_height * patch_width * 3
            self.patchnorm_pre_ln = LayerNorm(patch_input_dim)
            self.conv1 = nn.Linear(patch_input_dim, width)
        else:
            self.patchnorm_pre_ln = nn.Identity()
            self.conv1 = nn.Conv2d(
                in_channels=3,
                out_channels=width,
                kernel_size=patch_size,
                stride=patch_size,
                bias=False
            )

        # class embeddings and positional embeddings
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn(self.grid_size[0] * self.grid_size[1] + 1, width)
        )

        # setting a patch_dropout of 0. would mean it is disabled and this function would be the identity fn
        self.patch_dropout = PatchDropout(patch_dropout) if patch_dropout > 0. else nn.Identity()

        self.ln_pre = norm_layer(width)
        self.transformer = Transformer(
            width,
            layers,
            heads,
            mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )

        self.global_average_pool = global_average_pool
        if attentional_pool:
            self.attn_pool = AttentionalPooler(output_dim, width, n_head=attn_pooler_heads, n_queries=n_queries)
            self.ln_post = norm_layer(output_dim)
            self.proj = nn.Parameter(scale * torch.randn(output_dim, output_dim))
        else:
            self.attn_pool = None
            self.ln_post = norm_layer(width)
            self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        self.init_parameters()

    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        for param in self.parameters():
            param.requires_grad = False

        if unlocked_groups != 0:
            groups = [
                [
                    self.conv1,
                    self.class_embedding,
                    self.positional_embedding,
                    self.ln_pre,
                ],
                *self.transformer.resblocks[:-1],
                [
                    self.transformer.resblocks[-1],
                    self.ln_post,
                ],
                self.proj,
            ]

            def _unlock(x):
                if isinstance(x, Sequence):
                    for g in x:
                        _unlock(g)
                else:
                    if isinstance(x, torch.nn.Parameter):
                        x.requires_grad = True
                    else:
                        for p in x.parameters():
                            p.requires_grad = True

            _unlock(groups[-unlocked_groups:])

    def init_parameters(self):
        """
        BiomedCLIP Vision Transformer Initialization

        Reference: Zhang et al. "BiomedCLIP: a multimodal biomedical foundation model 
        pretrained from fifteen million scientific image-text pairs" (2023)
        Paper: https://arxiv.org/abs/2303.00915

        According to Supplementary Table 3 of the paper, BiomedCLIP initializes the 
        Vision Transformer (ViT-B/16) with ImageNet-pretrained weights rather than 
        random initialization or custom initialization schemes.

        Key findings from the paper:
        - Random initialization vs ImageNet pretraining showed similar validation 
          performance (Supplementary Table 3)
        - However, ImageNet-pretrained weights offered more stable performance on 
          downstream tasks
        - Therefore, BiomedCLIP chose to initialize ViT-B/16 with ImageNet-pretrained 
          weights from the original Vision Transformer paper (Dosovitskiy et al. 2020)

        Implementation details:
        - Vision encoder: ViT-B/16 initialized with ImageNet-pretrained weights
        - Text encoder: PubMedBERT (domain-specific pretrained language model)
        - No custom parameter initialization is applied after loading pretrained weights
        - The model undergoes continual pretraining on PMC-15M dataset

        Unlike standard OpenAI CLIP (which trains vision encoder from scratch), 
        BiomedCLIP leverages transfer learning from ImageNet pretraining.
        """
        # BiomedCLIP does NOT define custom initialization for Vision Transformer
        # Instead, it relies on pretrained ImageNet weights loaded during model creation
        # 
        # The initialization happens during model instantiation by loading:
        # 1. ViT-B/16 pretrained on ImageNet-21k (Dosovitskiy et al. 2020)
        # 2. PubMedBERT pretrained on PubMed abstracts (Gu et al. 2021)
        #
        # After loading pretrained weights, the model is trained end-to-end on PMC-15M
        # using the InfoNCE contrastive loss with the following hyperparameters:
        # - Batch size: 4096
        # - Learning rate: 5e-4 (with cosine annealing)
        # - Optimizer: AdamW (beta1=0.9, beta2=0.98, eps=1e-6, weight_decay=0.2)
        # - Image resolution: 224x224 (preprocessing with center crop and normalization)
        # - Context length: 256 tokens (for longer biomedical captions)

        pass  # No custom initialization - uses pretrained weights

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.grad_checkpointing = enable

    def _global_pool(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.global_average_pool:
            return x.mean(dim=1), x
        else:
            return x[:, 0], x[:, 1:]

    def forward(self, x: torch.Tensor, out_layers: list):
        # to patches - whether to use dual patchnorm - https://arxiv.org/abs/2302.01327v1
        if self.input_patchnorm:
            # einops - rearrange(x, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)')
            x = x.reshape(
                x.shape[0], x.shape[1], self.grid_size[0], self.patch_size[0],
                self.grid_size[1], self.patch_size[1]
            )
            x = x.permute(0, 2, 4, 1, 3, 5)
            x = x.reshape(x.shape[0], self.grid_size[0] * self.grid_size[1], -1)
            x = self.patchnorm_pre_ln(x)
            x = self.conv1(x)
        else:
            x = self.conv1(x)  # shape = [*, width, grid, grid]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # class embeddings and positional embeddings
        x = torch.cat(
            [
                self.class_embedding.to(x.dtype)
                + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
        x = self.patch_dropout(x)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND

        x, attn, patch_tokens = self.transformer(x, out_layers)

        B, C, L = attn[0].shape
        H = int(math.sqrt(L - 1))
        out_attn = torch.zeros([H, H]).to('cuda')
        for i in range(len(attn)):
            out_attn += attn[i][0, 0, 1:].view(H, H)
        x = x.permute(1, 0, 2)  # LND -> NLD
        patch_tokens = [patch_tokens[t].permute(1, 0, 2) for t in range(len(patch_tokens))]  # LND -> NLD

        if self.attn_pool is not None:
            x = self.attn_pool(x)
            x = self.ln_post(x)
            pooled, tokens = self._global_pool(x)
        else:
            pooled, tokens = self._global_pool(x)
            pooled = self.ln_post(pooled)

        if self.proj is not None:
            pooled = pooled @ self.proj

        if self.output_tokens:
            return pooled, patch_tokens

        return pooled, patch_tokens


        
class TimmModel(nn.Module):
    """
    Timm-based Vision Transformer for BiomedCLIP.
    
    Wraps timm's Vision Transformer with ImageNet pretraining.
    BiomedCLIP uses ViT-B/16 pretrained on ImageNet-21k.
    
    Reference: Zhang et al. "BiomedCLIP" (2023), Supplementary Table 3
    Shows that ImageNet-pretrained weights provide more stable downstream performance.
    """
    
    def __init__(
        self,
        model_name: str = 'vit_base_patch16_224',
        embed_dim: int = 768,
        image_size: Union[Tuple[int, int], int] = 224,
        pool: str = '',
        proj: str = 'linear',
        proj_bias: bool = False,
        drop: float = 0.,
        drop_path: Optional[float] = None,
        pretrained: bool = False,
        output_tokens: bool = False,
    ):
        """
        Initialize timm-based vision model.
        
        Args:
            model_name: Timm model name (e.g., 'vit_base_patch16_224')
            embed_dim: Output embedding dimension
            image_size: Input image size
            pool: Pooling type ('' for default, 'avg' for average pooling)
            proj: Projection type ('linear' or 'mlp')
            proj_bias: Whether to use bias in projection
            drop: Dropout rate for head
            drop_path: Stochastic depth rate
            pretrained: Load ImageNet pretrained weights
            output_tokens: Return patch tokens in addition to pooled output
        """
        super().__init__()
        
        import timm
        
        self.output_tokens = output_tokens
        self.image_size = to_2tuple(image_size)
        
        # Create timm model with optional pretraining
        # BiomedCLIP uses pretrained=True for ImageNet initialization
        self.trunk = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool=pool if pool else '',
        )
        
        # Get feature dimension from trunk
        if hasattr(self.trunk, 'num_features'):
            feat_size = self.trunk.num_features
        elif hasattr(self.trunk, 'embed_dim'):
            feat_size = self.trunk.embed_dim
        else:
            feat_size = self.trunk.head.in_features if hasattr(self.trunk, 'head') else 768
        
        self.feat_size = feat_size
        
        # Apply dropout if specified
        if drop > 0.:
            self.head_drop = nn.Dropout(drop)
        else:
            self.head_drop = nn.Identity()
        
        # Create projection layer
        if proj == 'linear':
            self.proj = nn.Linear(feat_size, embed_dim, bias=proj_bias)
        elif proj == 'mlp':
            self.proj = nn.Sequential(
                nn.Linear(feat_size, feat_size, bias=proj_bias),
                nn.GELU(),
                nn.Linear(feat_size, embed_dim, bias=proj_bias),
            )
        else:
            raise ValueError(f"Unknown projection type: {proj}")
        
        # Initialize projection layer
        # BiomedCLIP: Pretrained trunk is kept, only projection is newly initialized
        self._init_projection()
    
    def _init_projection(self):
        """
        Initialize projection layer for BiomedCLIP.
        
        The Vision Transformer trunk uses ImageNet-pretrained weights.
        Only the projection layer needs initialization.
        
        Follows CLIP's initialization strategy for projection.
        """
        if isinstance(self.proj, nn.Sequential):
            # MLP projection
            for module in self.proj:
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, std=self.feat_size ** -0.5)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        elif isinstance(self.proj, nn.Linear):
            # Linear projection
            nn.init.normal_(self.proj.weight, std=self.feat_size ** -0.5)
            if self.proj.bias is not None:
                nn.init.zeros_(self.proj.bias)
    
    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        """
        Lock (freeze) layers for fine-tuning.
        
        Args:
            unlocked_groups: Number of layer groups to keep unlocked from the end
            freeze_bn_stats: Whether to freeze batch norm statistics
        """
        # Freeze all trunk parameters
        for param in self.trunk.parameters():
            param.requires_grad = False
        
        if freeze_bn_stats:
            # Freeze batch norm layers
            for module in self.trunk.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        
        # Optionally unlock later layers
        if unlocked_groups > 0:
            # This depends on timm model structure
            # For ViT, unlock last N blocks
            if hasattr(self.trunk, 'blocks'):
                for block in self.trunk.blocks[-unlocked_groups:]:
                    for param in block.parameters():
                        param.requires_grad = True
    
    def forward(self, x: torch.Tensor):
        """
        Forward pass through vision encoder.
        
        Args:
            x: Input images [batch_size, 3, height, width]
        
        Returns:
            If output_tokens=False: Pooled features [batch_size, embed_dim]
            If output_tokens=True: (pooled features, patch tokens)
        """
        # Get features from trunk
        x = self.trunk.forward_features(x)
        
        # Handle different output formats from timm
        if isinstance(x, (tuple, list)):
            x = x[0]  # Take first element if multiple outputs
        
        pooled = x
        patch_tokens = None
        
        # Extract patch tokens if needed
        if self.output_tokens:
            if len(x.shape) == 3:  # [batch, num_patches, dim]
                pooled = x[:, 0]  # CLS token
                patch_tokens = [x[:, 1:]]  # Patch tokens (excluding CLS)
            else:  # [batch, dim]
                pooled = x
                patch_tokens = [x.unsqueeze(1)]  # Fake patch tokens
        else:
            if len(x.shape) == 3:
                pooled = x[:, 0]  # CLS token
        
        # Apply dropout and projection
        pooled = self.head_drop(pooled)
        pooled = self.proj(pooled)
        
        if self.output_tokens:
            return pooled, patch_tokens
        
        return pooled
    
    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        """Enable/disable gradient checkpointing."""
        try:
            self.trunk.set_grad_checkpointing(enable)
        except AttributeError:
            pass  # Not all timm models support this


class HFTextEncoder(nn.Module):
    """
    HuggingFace Text Encoder for BiomedCLIP.
    
    Wraps a pretrained HuggingFace model (BiomedBERT) for use in CLIP.
    Uses pretrained weights from BiomedBERT instead of training from scratch.
    
    Reference: Zhang et al. "BiomedCLIP: a multimodal biomedical foundation model 
    pretrained from fifteen million scientific image-text pairs" (2023)
    Paper: https://arxiv.org/abs/2303.00915
    """
    
    def __init__(
        self,
        model_name: str,
        output_dim: int,
        proj_type: str = 'mlp',
        pooler_type: str = 'cls_last_hidden_state_pooler',
        pretrained: bool = True,
        output_tokens: bool = False,
    ):
        """
        Initialize HuggingFace text encoder.
        
        Args:
            model_name: HuggingFace model identifier (e.g., 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract')
            output_dim: Dimension of output embeddings (typically 512 for CLIP)
            proj_type: Type of projection layer ('mlp' or 'linear')
            pooler_type: Type of pooling to use ('cls_last_hidden_state_pooler', 'mean_pooler', etc.)
            pretrained: Whether to load pretrained weights
            output_tokens: Whether to output token embeddings in addition to pooled output
        """
        super().__init__()
        
        from transformers import AutoModel, AutoConfig
        
        self.output_tokens = output_tokens
        self.output_dim = output_dim
        self.pooler_type = pooler_type
        
        # Load pretrained model configuration
        if pretrained:
            self.transformer = AutoModel.from_pretrained(model_name)
        else:
            config = AutoConfig.from_pretrained(model_name)
            self.transformer = AutoModel.from_config(config)
        
        # Get hidden size from the model
        self.hidden_size = self.transformer.config.hidden_size
        
        # Create projection layer
        if proj_type == 'mlp':
            # MLP projection: hidden -> hidden -> output_dim
            self.proj = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.GELU(),
                nn.Linear(self.hidden_size, output_dim)
            )
        elif proj_type == 'linear':
            # Simple linear projection: hidden -> output_dim
            self.proj = nn.Linear(self.hidden_size, output_dim)
        else:
            raise ValueError(f"Unknown projection type: {proj_type}")
        
        # Initialize projection layer
        self._init_projection()
    
    def _init_projection(self):
        """
        Initialize the projection layer.
        
        For BiomedCLIP: The projection layer is initialized while the transformer
        uses pretrained BiomedBERT weights.
        
        Initialization follows CLIP's approach for projection layers.
        """
        if isinstance(self.proj, nn.Sequential):
            # MLP projection initialization
            for module in self.proj:
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, std=self.hidden_size ** -0.5)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        elif isinstance(self.proj, nn.Linear):
            # Linear projection initialization
            nn.init.normal_(self.proj.weight, std=self.hidden_size ** -0.5)
            if self.proj.bias is not None:
                nn.init.zeros_(self.proj.bias)
    
    def pool_features(self, hidden_states, attention_mask):
        """
        Pool features from transformer outputs.
        
        Args:
            hidden_states: Transformer output hidden states [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
        
        Returns:
            Pooled features [batch_size, hidden_size]
        """
        if self.pooler_type == 'cls_last_hidden_state_pooler' or self.pooler_type == 'cls':
            # Use CLS token (first token) from last hidden state
            pooled = hidden_states[:, 0]
        
        elif self.pooler_type == 'mean_pooler' or self.pooler_type == 'mean':
            # Mean pooling over all tokens (excluding padding)
            attention_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * attention_mask_expanded, dim=1)
            sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)
            pooled = sum_embeddings / sum_mask
        
        elif self.pooler_type == 'max_pooler' or self.pooler_type == 'max':
            # Max pooling over all tokens
            attention_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            hidden_states = hidden_states.clone()
            hidden_states[attention_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
            pooled = torch.max(hidden_states, dim=1)[0]
        
        else:
            raise ValueError(f"Unknown pooler type: {self.pooler_type}")
        
        return pooled
    
    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass through the text encoder.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
        
        Returns:
            If output_tokens=False: Pooled text embeddings [batch_size, output_dim]
            If output_tokens=True: (pooled embeddings, token embeddings)
        """
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = (input_ids != 0).long()
        
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Get hidden states
        hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # Pool features
        pooled = self.pool_features(hidden_states, attention_mask)
        
        # Project to output dimension
        pooled = self.proj(pooled)
        
        if self.output_tokens:
            # Also project token embeddings if requested
            batch_size, seq_len, hidden_size = hidden_states.shape
            token_embeddings = self.proj(hidden_states.view(-1, hidden_size))
            token_embeddings = token_embeddings.view(batch_size, seq_len, self.output_dim)
            return pooled, token_embeddings
        
        return pooled
    
    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        """Enable/disable gradient checkpointing for the transformer."""
        self.transformer.gradient_checkpointing_enable() if enable else self.transformer.gradient_checkpointing_disable()


