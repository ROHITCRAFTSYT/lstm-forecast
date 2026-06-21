"""Small shared utilities: seeding and device selection."""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and (if available) torch RNGs for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover - torch is a core dep, but keep utils import-light
        pass


def resolve_device(preference: str = "auto") -> str:
    """Resolve a torch device string from a preference.

    ``"auto"`` picks cuda, then mps (Apple Silicon), then cpu.
    """
    import torch

    pref = preference.lower()
    if pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
