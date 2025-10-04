import json
import logging
import re
from copy import deepcopy
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

# from your local BiomedCLIP model code
from .model import resize_pos_embed, get_cast_dtype, CustomTextCLIP

#CLIP, convert_weights_to_lp, 


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





def create_model(
    model_name: str,
    img_size: int,
    pretrained: Optional[str] = None,
    precision: str = "fp32",
    device: Union[str, torch.device] = "cpu",
    jit: bool = False,
    force_quick_gelu: bool = False,
    force_custom_text: bool = True,
    force_patch_dropout: Optional[float] = None,
    force_image_size: Optional[Union[int, Tuple[int, int]]] = None,
    output_dict: Optional[bool] = None,
    require_pretrained: bool = False,
):

    model_name = model_name.replace('/', '-')  # for callers using old naming with / in ViT names
    checkpoint_path = None
    model_cfg = None

    if isinstance(device, str):
        device = torch.device(device)

    if pretrained and pretrained.lower() == 'openai':
        logging.info(f'Loading pretrained {model_name} from OpenAI.')
        model_cfg = model_cfg or get_model_config(model_name)
        if model_cfg['vision_cfg']['image_size'] != img_size:
            model_cfg['vision_cfg']['image_size'] = img_size
            print("Model Configuration 1",model_cfg)
        cast_dtype = get_cast_dtype(precision)

        model_pre = load_openai_model(
            name=_MODEL_CKPT_PATHS[model_name],
            precision=precision,
            device=device,
            jit=jit,
        )
        state_dict = model_pre.state_dict()

        model = CustomTextCLIP(**model_cfg, cast_dtype=cast_dtype)
        resize_pos_embed(state_dict, model)
        incompatible_keys = model.load_state_dict(state_dict, strict=True)
        model.to(device=device)
        if precision in ("fp16", "bf16"):
            convert_weights_to_lp(model, dtype=torch.bfloat16 if precision == 'bf16' else torch.float16)

        if output_dict and hasattr(model, "output_dict"):
            model.output_dict = True

        if jit:
            model = torch.jit.script(model)
    else:
        model_cfg = model_cfg or get_model_config(model_name)
        if model_cfg is not None:
            print(f'Loaded {model_name} model config.')
            print("Model Configuration 2",model_cfg)
        else:
            raise RuntimeError(f'Model config for {model_name} not found.')

        if force_quick_gelu:
            model_cfg["quick_gelu"] = True

        if force_patch_dropout is not None:
            model_cfg["vision_cfg"]["patch_dropout"] = force_patch_dropout

        if force_image_size is not None:
            model_cfg["vision_cfg"]["image_size"] = force_image_size

        cast_dtype = get_cast_dtype(precision)
        custom_text = model_cfg.pop('custom_text', False) or force_custom_text

        if custom_text:
            model = CustomTextCLIP(**model_cfg, cast_dtype=cast_dtype)
        else:
            model = CLIP(**model_cfg, cast_dtype=cast_dtype)

        pretrained_loaded = False
        if pretrained:
            checkpoint_path = _MODEL_CKPT_PATHS[model_name]
            if checkpoint_path:
                print(f'Loading pretrained {model_name} weights ({pretrained}).')
                load_checkpoint(model, checkpoint_path)
            else:
                raise RuntimeError(f'Pretrained weights ({pretrained}) not found for model {model_name}.')
            pretrained_loaded = True

        if require_pretrained and not pretrained_loaded:
            raise RuntimeError(
                f'Pretrained weights were required for (model: {model_name}, pretrained: {pretrained}) but not loaded.')

        model.to(device=device)
        if precision in ("fp16", "bf16"):
            convert_weights_to_lp(model, dtype=torch.bfloat16 if precision == 'bf16' else torch.float16)

        if output_dict and hasattr(model, "output_dict"):
            model.output_dict = True

        if jit:
            model = torch.jit.script(model)

    return model