# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Modified: Uses transformers library for reliable GPT-2 weight loading.
# Source for "Build a Large Language Model From Scratch"

import json
import os
from typing import Any

import numpy as np
import torch
from tools.log import log as _gpt_log
import requests
from tqdm import tqdm


def download_and_load_gpt2(model_size, models_dir):
    """Download GPT-2 model files and return settings + parameter dict."""
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size not in {allowed_sizes}")

    model_dir = os.path.join(models_dir, model_size)
    _download_gpt2_files(model_dir, model_size)

    with open(os.path.join(model_dir, "hparams.json"), "r") as f:
        settings = json.load(f)

    params = _load_weights_from_ckpt(model_dir, settings, model_size)
    return settings, params

def load_weights_from_ckpt(model_size, models_dir):
    model_dir = os.path.join(models_dir, model_size)
    with open(os.path.join(model_dir, "hparams.json"), "r") as f:
        settings = json.load(f)
    params = _load_weights_from_ckpt(model_dir, settings, model_size)
    return settings, params

# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _download_gpt2_files(model_dir, model_size):
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    backup_url = "https://f001.backblazeb2.com/file/LLMs-from-scratch/gpt2"
    files = [
        "checkpoint", "encoder.json", "hparams.json",
        "model.ckpt.data-00000-of-00001", "model.ckpt.index",
        "model.ckpt.meta", "vocab.bpe",
    ]
    os.makedirs(model_dir, exist_ok=True)
    for fn in files:
        download_file(f"{base_url}/{model_size}/{fn}",
                      os.path.join(model_dir, fn),
                      backup=f"{backup_url}/{model_size}/{fn}")


def download_file(url, dest, backup=None):
    """Download *url* to *dest* with progress bar."""
    def _try(u):
        r = requests.get(u, stream=True, timeout=60)
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        if os.path.exists(dest) and total == os.path.getsize(dest):
            return True
        desc = os.path.basename(dest)
        with tqdm(total=total, unit="iB", unit_scale=True, desc=desc) as pb:
            with open(dest, "wb") as out:
                for chunk in r.iter_content(1024):
                    if chunk:
                        out.write(chunk)
                        pb.update(len(chunk))
        return True

    try:
        if _try(url): return
    except Exception: pass
    if backup:
        try: 
            if _try(backup): return
        except Exception: pass
    raise RuntimeError(f"Failed to download {os.path.basename(url)}")


# ---------------------------------------------------------------------------
# Weight loading via transformers (pure PyTorch – no tensorflow needed)
# ---------------------------------------------------------------------------

_MODEL_SIZE_TO_HF = {
    "124M": "openai-community/gpt2",
    "355M": "openai-community/gpt2-medium",
    "774M": "openai-community/gpt2-large",
    "1558M": "openai-community/gpt2-xl",
}

# Per-layer HF weight names → nested-dict path tuples (relative to block dict)
_LAYER_HF_TO_PATH = {
    "transformer.h.{n}.attn.c_attn.weight":   ("attn", "c_attn", "w"),
    "transformer.h.{n}.attn.c_attn.bias":     ("attn", "c_attn", "b"),
    "transformer.h.{n}.attn.c_proj.weight":   ("attn", "c_proj", "w"),
    "transformer.h.{n}.attn.c_proj.bias":     ("attn", "c_proj", "b"),
    "transformer.h.{n}.ln_1.weight":          ("ln_1", "g"),
    "transformer.h.{n}.ln_1.bias":            ("ln_1", "b"),
    "transformer.h.{n}.ln_2.weight":          ("ln_2", "g"),
    "transformer.h.{n}.ln_2.bias":            ("ln_2", "b"),
    "transformer.h.{n}.mlp.c_fc.weight":      ("mlp", "c_fc", "w"),
    "transformer.h.{n}.mlp.c_fc.bias":        ("mlp", "c_fc", "b"),
    "transformer.h.{n}.mlp.c_proj.weight":    ("mlp", "c_proj", "w"),
    "transformer.h.{n}.mlp.c_proj.bias":      ("mlp", "c_proj", "b"),
}

# Top-level HF weight names → nested-dict path tuples
_TOP_LEVEL_HF_TO_PATH = {
    "transformer.wte.weight":  ("wte",),
    "transformer.wpe.weight":  ("wpe",),
    "transformer.ln_f.weight": ("ln_f", "g"),
    "transformer.ln_f.bias":   ("ln_f", "b"),
}


def _load_weights_from_ckpt(model_dir, settings, model_size):
    """
    Load pretrained GPT-2 weights from HuggingFace into our nested-dict format.
    """
    n_layer = settings["n_layer"]
    params: dict[str, Any] = {"blocks": [{} for _ in range(n_layer)]}

    try:
        from transformers import GPT2LMHeadModel
        import warnings; warnings.filterwarnings("ignore")

        hf_model_name = _MODEL_SIZE_TO_HF[model_size]
        _gpt_log.info("Loading weights from HuggingFace: %s", hf_model_name)
        hf_model = GPT2LMHeadModel.from_pretrained(hf_model_name)
        sd = hf_model.state_dict()

        # 1) Per-layer weights — store into params["blocks"][layer_idx]
        for hf_pattern, path_parts in _LAYER_HF_TO_PATH.items():
            for layer_idx in range(n_layer):
                hf_key = hf_pattern.format(n=layer_idx)
                if hf_key not in sd:
                    continue
                target = params["blocks"][layer_idx]
                for p in path_parts[:-1]:
                    target = target.setdefault(p, {})
                target[path_parts[-1]] = sd[hf_key].numpy()

        # 2) Top-level weights — wte, wpe, ln_f
        for hf_key, path_parts in _TOP_LEVEL_HF_TO_PATH.items():
            if hf_key in sd:
                target = params
                for p in path_parts[:-1]:
                    target = target.setdefault(p, {})
                target[path_parts[-1]] = sd[hf_key].numpy()

        _gpt_log.info("Loaded %d weight tensors from HuggingFace", len(sd))
        return params

    except ImportError:
        _gpt_log.error(
            "transformers not installed. "
            "Install with: pip install transformers"
        )
        raise

    except Exception as e:
        raise RuntimeError(f"Failed to load GPT-2 weights: {e}") from e
