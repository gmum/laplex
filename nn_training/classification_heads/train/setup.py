import inspect
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import swin_t, Swin_T_Weights, maxvit_t, MaxVit_T_Weights

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "laplex_lib"))

_BACKBONES: dict = {
    "swin_t":   (swin_t,   Swin_T_Weights.IMAGENET1K_V1),
    "maxvit_t": (maxvit_t, MaxVit_T_Weights.IMAGENET1K_V1),
}


def build_model(config, head_cls: type) -> nn.Module:
    """Load pretrained backbone, replace classification head, optionally freeze.

    Returns the bare nn.Module (DDP wrapping is done by the caller).
    """
    if config.MODEL not in _BACKBONES:
        raise ValueError(
            f"Unknown backbone {config.MODEL!r}. Choose from {list(_BACKBONES)}."
        )
    model_fn, weights = _BACKBONES[config.MODEL]
    model = model_fn(weights=weights)

    # Build head kwargs from config, filtered to what the head class accepts
    sig = inspect.signature(head_cls.__init__)
    valid = set(sig.parameters) - {"self", "in_features", "out_features"}
    candidates = {
        "rank":       getattr(config, "RANK",       16),
        "ab_scale":   getattr(config, "AB_SCALE",   "auto"),
        "c_scale":    getattr(config, "C_SCALE",    "auto"),
        "tau_init":   getattr(config, "TAU_INIT",   1.0),
        "sigma_init": getattr(config, "SIGMA_INIT", 0.5),
    }
    head_kwargs = {k: v for k, v in candidates.items() if k in valid}

    model = replace_head(model, config.MODEL, head_cls, **head_kwargs)

    if getattr(config, "FROZEN", False):
        model = freeze_backbone(model, config.MODEL)

    return model


def replace_head(
    model: nn.Module,
    model_name: str,
    head_cls: type,
    **head_kwargs,
) -> nn.Module:
    """Replace the classification head with a new layer of type head_cls."""
    if model_name == "swin_t":
        in_features = model.head.in_features
        model.head = head_cls(in_features, 1000, **head_kwargs)

    elif model_name == "maxvit_t":
        in_f3 = model.classifier[3].in_features
        out_f3 = model.classifier[3].out_features
        model.classifier[3] = head_cls(in_f3, out_f3, **head_kwargs)

        in_f5 = model.classifier[5].in_features
        model.classifier[5] = head_cls(in_f5, 1000, **head_kwargs)

    return model


def freeze_backbone(model: nn.Module, model_name: str) -> nn.Module:
    """Freeze all parameters, then unfreeze the classification head."""
    for p in model.parameters():
        p.requires_grad = False

    if model_name == "swin_t":
        for p in model.head.parameters():
            p.requires_grad = True

    elif model_name == "maxvit_t":
        for p in model.classifier[3].parameters():
            p.requires_grad = True
        for p in model.classifier[5].parameters():
            p.requires_grad = True

    return model
