"""Loss functions for GFGW-FM."""

from .ot_solver import (
    SinkhornSolver,
    FusedGromovWassersteinSolver,
    SemiDiscreteOTSolver,
    OTMatchingModule,
)
from .flow_matching import (
    FlowMatchingLoss,
    FeatureMatchingLoss,
    GFGWFlowMatchingLoss,
    TextureConsistencyLoss,
)

__all__ = [
    'SinkhornSolver',
    'FusedGromovWassersteinSolver',
    'SemiDiscreteOTSolver',
    'OTMatchingModule',
    'FlowMatchingLoss',
    'FeatureMatchingLoss',
    'GFGWFlowMatchingLoss',
    'TextureConsistencyLoss',
]
