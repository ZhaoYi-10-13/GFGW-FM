"""Utility functions for GFGW-FM."""

import torch
import numpy as np
from typing import Optional

# Import scheduling utilities
from .scheduling import (
    TimeSampler,
    AnnealingSchedule,
    TwoStageSchedule,
    LRScheduler,
    OTEpsilonSchedule,
    FGWLambdaSchedule,
    TrainingScheduler,
)

# Import pretrained model loading utilities (ECM/TCM compatible)
from .pretrained import (
    PretrainedModelLoader,
    load_pretrained_edm_model,
    create_ema_from_pretrained,
    copy_params_and_buffers,
)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_module_summary(
    module: torch.nn.Module,
    inputs: list,
    max_nesting: int = 3,
):
    """Print summary of module architecture."""
    print(f"Model: {module.__class__.__name__}")
    print(f"Parameters: {count_parameters(module):,}")


class EasyDict(dict):
    """Dictionary with attribute access."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        del self[name]


__all__ = [
    'set_seed',
    'count_parameters',
    'print_module_summary',
    'EasyDict',
    'TimeSampler',
    'AnnealingSchedule',
    'TwoStageSchedule',
    'LRScheduler',
    'OTEpsilonSchedule',
    'FGWLambdaSchedule',
    'TrainingScheduler',
    # Pretrained model loading (ECM/TCM compatible)
    'PretrainedModelLoader',
    'load_pretrained_edm_model',
    'create_ema_from_pretrained',
    'copy_params_and_buffers',
]
