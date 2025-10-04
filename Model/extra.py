import json
import logging
import re
from copy import deepcopy
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

# from your local BiomedCLIP model code
#from .model import CLIP, convert_weights_to_lp, resize_pos_embed, get_cast_dtype


# Paths for configs and checkpoints
_MODEL_CONFIG_PATHS = [
    Path("/kaggle/input/biomedclip/pytorch/default/1/BiomedCLIP-PubMedBERT-ViT-B-16.json")
]
_MODEL_CONFIGS = {}
_MODEL_CKPT_PATHS = {
    "BiomedCLIP-PubMedBERT-ViT-B-16": Path(
        "/kaggle/input/biomedclip/pytorch/default/1/open_clip_pytorch_model.bin"
    )
}


def _natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", string_.lower())]


def _rescan_model_configs():
    global _MODEL_CONFIGS
    config_ext = (".json",)
    config_files = []

    for config_path in _MODEL_CONFIG_PATHS:
        if config_path.is_file() and config_path.suffix in config_ext:
            config_files.append(config_path)
        elif config_path.is_dir():
            for ext in config_ext:
                config_files.extend(config_path.glob(f"*{ext}"))

    for cf in config_files:
        with open(cf, "r") as f:
            model_cfg = json.load(f)
            if (
                "model_cfg" in model_cfg
                and all(a in model_cfg["model_cfg"] for a in ("embed_dim", "vision_cfg", "text_cfg"))
            ):
                _MODEL_CONFIGS[cf.stem] = model_cfg

    _MODEL_CONFIGS = {
        k: v for k, v in sorted(_MODEL_CONFIGS.items(), key=lambda x: _natural_key(x[0]))
    }


_rescan_model_configs()


def list_models():
    return list(_MODEL_CONFIGS.keys())


def get_model_config(model_name):
    if model_name in _MODEL_CONFIGS:
        return deepcopy(_MODEL_CONFIGS[model_name]["model_cfg"])
    else:
        return None


def load_state_dict(checkpoint_path: str, map_location="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    if next(iter(state_dict.items()))[0].startswith("module"):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    return state_dict



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



def load_checkpoint(model, checkpoint_path, strict=True):
    state_dict = load_state_dict(checkpoint_path)

    # 🔑 Fix: rename keys to match CLIP implementation
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("text.transformer."):
            new_k = k.replace("text.transformer.", "text_model.")
        elif k.startswith("text."):
            new_k = k.replace("text.", "text_model.")
        else:
            new_k = k
        new_state_dict[new_k] = v

    resize_pos_embed(new_state_dict, model)
    incompatible_keys = model.load_state_dict(new_state_dict, strict=False)  # keep non-strict
    return incompatible_keys

def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == 'bf16':
        cast_dtype = torch.bfloat16
    elif precision == 'fp16':
        cast_dtype = torch.float16
    return cast_dtype



def create_model(
    model_name: str,
    img_size: int,
    pretrained: Optional[str] = None,
    precision: str = "fp32",
    device: Union[str, torch.device] = "cpu",
    jit: bool = False,
    force_patch_dropout: Optional[float] = None,
    force_image_size: Optional[Union[int, Tuple[int, int]]] = None,
    output_dict: Optional[bool] = None,
    require_pretrained: bool = False,
):
    model_name = model_name.replace("/", "-")
    model_cfg = get_model_config(model_name)

    if model_cfg is None:
        raise RuntimeError(f"Model config for {model_name} not found.")

    # Update vision image size if requested
    if model_cfg["vision_cfg"]["image_size"] != img_size:
        model_cfg["vision_cfg"]["image_size"] = img_size

    if force_patch_dropout is not None:
        model_cfg["vision_cfg"]["patch_dropout"] = force_patch_dropout
    if force_image_size is not None:
        model_cfg["vision_cfg"]["image_size"] = force_image_size

    cast_dtype = get_cast_dtype(precision)

    # ✅ FIXED indentation starts here
    model_pre = load_biomedclip_model(
        name=_MODEL_CKPT_PATHS[model_name],
        precision=precision,
        device=device,
        cache_dir=None,
        jit=jit
    )
    state_dict = model_pre.state_dict()

    # to always output dict even if it is clip
    if output_dict and hasattr(model_pre, "output_dict"):
        model_pre.output_dict = True

    model = CLIP(**model_cfg, cast_dtype=cast_dtype)

    # for resnet case
    if not hasattr(model.visual, "grid_size"):
        model.visual.grid_size = int(np.sqrt(model.visual.attnpool.positional_embedding.shape[0] - 1))

    resize_pos_embed(state_dict, model)
    incompatible_keys = model.load_state_dict(state_dict, strict=True)

    model.to(device=device)

    if precision in ("fp16", "bf16"):
        convert_weights_to_lp(model, dtype=torch.bfloat16 if precision == "bf16" else torch.float16)

    # set normalization
    model.visual.image_mean = (0.48145466, 0.4578275, 0.40821073)
    model.visual.image_std = (0.26862954, 0.26130258, 0.27577711)

    if output_dict and hasattr(model, "output_dict"):
        model.output_dict = True

    if jit:
        model = torch.jit.script(model)

    return model
