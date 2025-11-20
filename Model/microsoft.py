""" BiomedCLIP pretrained model loader """

import torch
from typing import Optional, Union
from transformers import AutoModel, AutoProcessor
import os
import warnings

__all__ = ["load_biomedclip_model"]

def load_biomedclip_model(
        name: str,
        precision: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        cache_dir: Optional[str] = None,
        jit: bool =True,
):
    """
    Load the BiomedCLIP model from Hugging Face Hub or local checkpoint.

    Parameters
    ----------
    name : str
        Model name on Hugging Face (default BiomedCLIP) or path to local .bin checkpoint.
    precision : str
        Precision: 'fp32' or 'fp16'. Defaults to 'fp32' on CPU, 'fp16' on GPU.
    device : str or torch.device
        Device to place the model on.
    cache_dir : str, optional
        Directory for caching Hugging Face models.

    Returns
    -------
    model : torch.nn.Module
        BiomedCLIP model
    processor : transformers.AutoProcessor
        Processor for preprocessing inputs
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if precision is None:
        precision = "fp32" if device == "cpu" else "fp16"

    dtype = torch.float16 if precision == "fp16" else torch.float32

    if os.path.isfile(name):
        model_path = name
        print("path of model",model_path)
    else:
        raise RuntimeError(f"Model {name} not found; available models")

    try:
        # Try to load as JIT archive
        model = torch.jit.load(model_path, map_location=device if jit else "cpu").eval()
        state_dict = None
    
    except RuntimeError:
        # Fallback: load as state dict
        if jit:
            warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
            jit = False
        state_dict = torch.load(model_path, map_location="cpu")

        #print("\n \n here is the state dict",len(state_dict))
        # Handle possible "state_dict" wrapper
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        # Remove 'module.' prefix if present
        if next(iter(state_dict.keys())).startswith("module."):
            state_dict = {k[7:]: v for k, v in state_dict.items()}

    if not jit:
        cast_dtype = get_cast_dtype(precision)
        try:
            # Build model from state dict'
            #print("inside try block",model.state_dict)
            #print("inside try block",state_dict)
            
            model = build_model_from_biomedclip_state_dict(state_dict or model.state_dict(), cast_dtype=cast_dtype)
        except KeyError:
            # If another wrapper (e.g., state_dict inside dict)
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                sd = {k[7:]: v for k, v in state_dict["state_dict"].items()}
                model = build_model_from_biomedclip_state_dict(sd, cast_dtype=cast_dtype)

        # model from OpenAI state dict is in manually cast fp16 mode, must be converted for AMP/fp32/bf16 use
        model = model.to(device)
        if precision.startswith('amp') or precision == 'fp32':
            model.float()
        elif precision == 'bf16':
            convert_weights_to_lp(model, dtype=torch.bfloat16)

        return model
