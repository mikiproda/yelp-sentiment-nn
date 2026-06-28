"""Reproducibility: seed every RNG we touch and force deterministic cuDNN.

Called at the top of every entry point (data prep, each training run, evaluation).
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and (if available) PyTorch + CUDA + HF transformers."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        # torch is absent in the lightweight data-prep environment; that's fine.
        pass

    try:
        import transformers

        transformers.set_seed(seed)
    except ImportError:
        pass
