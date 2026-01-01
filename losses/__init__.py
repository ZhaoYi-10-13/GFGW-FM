"""Loss functions for GFGW-FM."""

from .ot_solver import (
    SinkhornSolver,
    FusedGromovWassersteinSolver,
    SemiDiscreteOTSolver,
    OTMatchingModule,
    LogStabilizedSinkhorn,
    EnhancedFGWSolver,
    HungarianMatcher,
    SemiDiscreteOTSolverV2,
    OTMatchingModuleV2,
)
from .flow_matching import (
    FlowMatchingLoss,
    FeatureMatchingLoss,
    GFGWFlowMatchingLoss,
    TextureConsistencyLoss,
    ComprehensiveFlowLoss,
    create_loss_fn,
    MultiScaleFlowLoss,
)

# Try to import advanced losses
try:
    from .advanced_losses import (
        PseudoHuberLoss,
        LPIPSLoss,
        BoundaryConditionLoss,
        ConsistencyLoss,
        AdaptiveWeighting,
        StructurePreservationLoss,
        GFGWFlowMatchingLossV2,
    )
    HAS_ADVANCED_LOSSES = True
except ImportError:
    HAS_ADVANCED_LOSSES = False

__all__ = [
    'SinkhornSolver',
    'FusedGromovWassersteinSolver',
    'SemiDiscreteOTSolver',
    'OTMatchingModule',
    'LogStabilizedSinkhorn',
    'EnhancedFGWSolver',
    'HungarianMatcher',
    'SemiDiscreteOTSolverV2',
    'OTMatchingModuleV2',
    'FlowMatchingLoss',
    'FeatureMatchingLoss',
    'GFGWFlowMatchingLoss',
    'TextureConsistencyLoss',
    'ComprehensiveFlowLoss',
    'create_loss_fn',
    'MultiScaleFlowLoss',
]
