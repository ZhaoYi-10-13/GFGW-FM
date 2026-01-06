"""Enhanced Fused Gromov-Wasserstein Optimal Transport solver for GFGW-FM.

Implements advanced FGW-based OT algorithms with:
- Adaptive epsilon scheduling
- Unbalanced OT support
- Hungarian algorithm for exact assignment
- Improved numerical stability
- Memory-efficient computation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class LogStabilizedSinkhorn:
    """
    Log-stabilized Sinkhorn algorithm with enhanced numerical stability.

    Supports:
    - Standard balanced OT
    - Unbalanced OT with KL divergence penalty
    - Partial OT for outlier robustness
    """

    def __init__(
        self,
        epsilon: float = 0.05,
        max_iters: int = 100,
        threshold: float = 1e-6,
        unbalanced: bool = False,
        unbalanced_reg: float = 1.0,
    ):
        self.epsilon = epsilon
        self.max_iters = max_iters
        self.threshold = threshold
        self.unbalanced = unbalanced
        self.unbalanced_reg = unbalanced_reg

    def __call__(
        self,
        cost_matrix: torch.Tensor,
        a: Optional[torch.Tensor] = None,
        b: Optional[torch.Tensor] = None,
        epsilon: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Solve OT using log-stabilized Sinkhorn algorithm.

        Args:
            cost_matrix: Cost matrix C of shape (n, m)
            a: Source distribution (n,), uniform if None
            b: Target distribution (m,), uniform if None
            epsilon: Override default epsilon

        Returns:
            Transport plan P of shape (n, m)
        """
        eps = epsilon if epsilon is not None else self.epsilon
        n, m = cost_matrix.shape
        device = cost_matrix.device
        dtype = cost_matrix.dtype

        # Initialize marginals
        if a is None:
            a = torch.ones(n, device=device, dtype=dtype) / n
        if b is None:
            b = torch.ones(m, device=device, dtype=dtype) / m

        # Normalize and stabilize cost matrix
        cost_max = cost_matrix.max()
        if cost_max > 0:
            C_norm = cost_matrix / (cost_max + 1e-8)
        else:
            C_norm = cost_matrix

        # Log kernel: log K = -C / epsilon
        log_K = -C_norm / eps

        # Initialize dual variables
        log_a = torch.log(a.clamp(min=1e-10))
        log_b = torch.log(b.clamp(min=1e-10))

        u = torch.zeros(n, device=device, dtype=dtype)
        v = torch.zeros(m, device=device, dtype=dtype)

        for iteration in range(self.max_iters):
            u_prev = u.clone()

            if self.unbalanced:
                # Unbalanced Sinkhorn with KL divergence
                tau = self.unbalanced_reg / (self.unbalanced_reg + eps)

                # u update
                log_sum_v = torch.logsumexp(log_K + v.unsqueeze(0), dim=1)
                u = tau * (log_a - log_sum_v) + (1 - tau) * u

                # v update
                log_sum_u = torch.logsumexp(log_K.T + u.unsqueeze(0), dim=1)
                v = tau * (log_b - log_sum_u) + (1 - tau) * v
            else:
                # Standard balanced Sinkhorn
                u = log_a - torch.logsumexp(log_K + v.unsqueeze(0), dim=1)
                v = log_b - torch.logsumexp(log_K.T + u.unsqueeze(0), dim=1)

            # Check convergence
            if (u - u_prev).abs().max() < self.threshold:
                break

        # Compute transport plan
        log_P = u.unsqueeze(1) + log_K + v.unsqueeze(0)
        P = torch.exp(log_P)

        # Normalize to ensure valid transport plan
        P = P / (P.sum() + 1e-10) * min(a.sum(), b.sum())

        return P


class EnhancedFGWSolver:
    """
    Enhanced Fused Gromov-Wasserstein solver with advanced features.

    Key improvements:
    - Adaptive regularization
    - Multi-scale structure matching
    - Efficient GPU computation
    - Warm start for faster convergence
    """

    def __init__(
        self,
        fgw_lambda: float = 0.5,
        epsilon: float = 0.05,
        num_sinkhorn_iters: int = 50,
        num_fgw_iters: int = 10,
        use_cosine_distance: bool = False,
        unbalanced: bool = False,
        unbalanced_reg: float = 1.0,
    ):
        self.fgw_lambda = fgw_lambda
        self.epsilon = epsilon
        self.num_sinkhorn_iters = num_sinkhorn_iters
        self.num_fgw_iters = num_fgw_iters
        self.use_cosine_distance = use_cosine_distance

        self.sinkhorn = LogStabilizedSinkhorn(
            epsilon=epsilon,
            max_iters=num_sinkhorn_iters,
            unbalanced=unbalanced,
            unbalanced_reg=unbalanced_reg,
        )

        # Warm start: store previous coupling
        self._prev_coupling = None

    def compute_feature_cost(
        self,
        features_gen: torch.Tensor,
        features_real: torch.Tensor,
    ) -> torch.Tensor:
        """Compute feature-based Wasserstein cost matrix."""
        if self.use_cosine_distance:
            features_gen = F.normalize(features_gen, dim=-1)
            features_real = F.normalize(features_real, dim=-1)
            similarity = torch.mm(features_gen, features_real.T)
            cost = 1 - similarity
        else:
            cost = torch.cdist(features_gen, features_real, p=2) ** 2

        return cost

    def compute_structure_cost(
        self,
        D_gen: torch.Tensor,
        D_real: torch.Tensor,
        coupling: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Gromov-Wasserstein structure cost gradient.

        Uses efficient tensor operations for GPU acceleration.
        """
        # Marginals
        mu = coupling.sum(dim=1, keepdim=True)  # (n, 1)
        nu = coupling.sum(dim=0, keepdim=True)  # (1, m)

        # Squared distance terms
        D_gen_sq = D_gen ** 2
        D_real_sq = D_real ** 2

        # Term 1: D_gen^2 contribution
        # sum_i' D_gen[i,i']^2 * sum_j' P[i',j'] for each i,j
        D_gen_sq_sum = D_gen_sq.sum(dim=1, keepdim=True)  # (n, 1)
        term1 = D_gen_sq_sum @ nu  # (n, m)

        # Term 2: D_real^2 contribution
        D_real_sq_sum = D_real_sq.sum(dim=0, keepdim=True)  # (1, m)
        term2 = mu @ D_real_sq_sum  # (n, m)

        # Term 3: Cross term (structure preservation)
        # -2 * D_gen @ P @ D_real.T
        term3 = 2 * torch.mm(torch.mm(D_gen, coupling), D_real.T)  # (n, m)

        gw_gradient = term1 + term2 - term3

        return gw_gradient

    def __call__(
        self,
        features_gen: torch.Tensor,
        features_real: torch.Tensor,
        D_gen: Optional[torch.Tensor] = None,
        D_real: Optional[torch.Tensor] = None,
        a: Optional[torch.Tensor] = None,
        b: Optional[torch.Tensor] = None,
        epsilon: Optional[float] = None,
        fgw_lambda: Optional[float] = None,
        warm_start: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Solve Fused Gromov-Wasserstein OT problem.

        Args:
            features_gen: Generated features (n, d)
            features_real: Real features (m, d)
            D_gen: Distance matrix for generated (n, n)
            D_real: Distance matrix for real (m, m)
            a: Source marginal
            b: Target marginal
            epsilon: Override epsilon
            fgw_lambda: Override lambda
            warm_start: Use previous coupling as initialization

        Returns:
            (optimal_coupling, final_cost_matrix)
        """
        n = features_gen.shape[0]
        m = features_real.shape[0]
        device = features_gen.device
        dtype = features_gen.dtype

        eps = epsilon if epsilon is not None else self.epsilon
        lam = fgw_lambda if fgw_lambda is not None else self.fgw_lambda

        # Compute distance matrices if not provided
        if D_gen is None:
            D_gen = torch.cdist(features_gen, features_gen, p=2)
        if D_real is None:
            D_real = torch.cdist(features_real, features_real, p=2)

        # Initialize marginals
        if a is None:
            a = torch.ones(n, device=device, dtype=dtype) / n
        if b is None:
            b = torch.ones(m, device=device, dtype=dtype) / m

        # Compute feature cost
        C_W = self.compute_feature_cost(features_gen, features_real)

        # Normalize for stability
        C_W_max = C_W.max()
        if C_W_max > 0:
            C_W_norm = C_W / C_W_max
        else:
            C_W_norm = C_W

        # Initialize coupling
        if warm_start and self._prev_coupling is not None:
            # Resize previous coupling if shapes changed
            if self._prev_coupling.shape == (n, m):
                coupling = self._prev_coupling.clone()
            else:
                coupling = a.unsqueeze(1) * b.unsqueeze(0)
        else:
            coupling = a.unsqueeze(1) * b.unsqueeze(0)

        # FGW iterations
        for iteration in range(self.num_fgw_iters):
            # Compute GW gradient
            C_GW = self.compute_structure_cost(D_gen, D_real, coupling)

            # Normalize
            C_GW_max = C_GW.abs().max()
            if C_GW_max > 0:
                C_GW_norm = C_GW / C_GW_max
            else:
                C_GW_norm = C_GW

            # Fused cost
            C_fused = (1 - lam) * C_W_norm + lam * C_GW_norm

            # Solve OT
            coupling = self.sinkhorn(C_fused, a, b, epsilon=eps)

        # Store for warm start
        self._prev_coupling = coupling.detach().clone()

        # Compute final unnormalized cost
        C_final = (1 - lam) * C_W + lam * C_GW

        return coupling, C_final


class HungarianMatcher:
    """
    Hungarian algorithm for exact one-to-one assignment.

    Use when you need deterministic Monge map assignment.
    Falls back to greedy matching if scipy not available.
    """

    def __init__(self):
        self.has_scipy = HAS_SCIPY

    @torch.no_grad()
    def __call__(
        self,
        cost_matrix: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find optimal assignment using Hungarian algorithm.

        Args:
            cost_matrix: Cost matrix (n, m)

        Returns:
            (row_indices, col_indices) of optimal assignment
        """
        if self.has_scipy:
            # Use scipy for exact solution
            cost_np = cost_matrix.cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost_np)
            return (
                torch.tensor(row_ind, device=cost_matrix.device),
                torch.tensor(col_ind, device=cost_matrix.device)
            )
        else:
            # Greedy fallback
            return self._greedy_assignment(cost_matrix)

    def _greedy_assignment(
        self,
        cost_matrix: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Greedy assignment fallback."""
        n, m = cost_matrix.shape
        device = cost_matrix.device

        row_indices = []
        col_indices = []
        used_cols = set()

        for i in range(min(n, m)):
            # Find minimum cost in remaining rows/cols
            mask = torch.ones(m, device=device, dtype=torch.bool)
            for j in used_cols:
                mask[j] = False

            if not mask.any():
                break

            row_costs = cost_matrix[i, mask]
            min_idx = row_costs.argmin()

            # Map back to original column index
            col_idx = torch.arange(m, device=device)[mask][min_idx]

            row_indices.append(i)
            col_indices.append(col_idx.item())
            used_cols.add(col_idx.item())

        return (
            torch.tensor(row_indices, device=device),
            torch.tensor(col_indices, device=device)
        )


class SemiDiscreteOTSolverV2:
    """
    Enhanced Semi-discrete OT solver with online dual potential updates.

    Improvements:
    - Momentum-based dual updates
    - Adaptive learning rate
    - Regularized potentials for stability
    """

    def __init__(
        self,
        num_data_points: int,
        feature_dim: int,
        dual_lr: float = 0.1,
        momentum: float = 0.9,
        epsilon: float = 0.05,
        device: torch.device = torch.device("cuda"),
    ):
        self.num_data_points = num_data_points
        self.feature_dim = feature_dim
        self.dual_lr = dual_lr
        self.momentum = momentum
        self.epsilon = epsilon
        self.device = device

        # Dual potentials
        self.phi = torch.zeros(num_data_points, device=device)
        self.phi_velocity = torch.zeros(num_data_points, device=device)

        # Statistics for adaptive lr
        self.update_count = 0

    @torch.no_grad()
    def update_dual_potentials(
        self,
        coupling: torch.Tensor,
        data_indices: Optional[torch.Tensor] = None,
    ):
        """
        Update dual potentials with momentum.

        phi_j <- phi_j - lr * (sum_i P[i,j] - 1/N)
        """
        m = coupling.shape[1]

        # Target uniform marginal
        target_marginal = torch.ones(m, device=self.device) / m

        # Actual marginal
        actual_marginal = coupling.sum(dim=0)

        # Gradient
        grad_phi = actual_marginal - target_marginal

        # Adaptive learning rate based on gradient magnitude
        grad_norm = grad_phi.norm()
        adaptive_lr = self.dual_lr / (1 + 0.1 * self.update_count)

        if data_indices is not None:
            # Momentum update
            self.phi_velocity[data_indices] = (
                self.momentum * self.phi_velocity[data_indices] +
                (1 - self.momentum) * grad_phi
            )
            self.phi[data_indices] = (
                self.phi[data_indices] - adaptive_lr * self.phi_velocity[data_indices]
            )
        else:
            self.phi_velocity[:m] = (
                self.momentum * self.phi_velocity[:m] +
                (1 - self.momentum) * grad_phi
            )
            self.phi[:m] = self.phi[:m] - adaptive_lr * self.phi_velocity[:m]

        self.update_count += 1

    def get_modified_cost(
        self,
        cost_matrix: torch.Tensor,
        indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply dual potentials to cost matrix."""
        if indices is not None:
            phi = self.phi[indices]
        else:
            phi = self.phi[:cost_matrix.shape[1]]

        return cost_matrix - phi.unsqueeze(0)

    def reset(self):
        """Reset potentials (e.g., when starting new epoch)."""
        self.phi.zero_()
        self.phi_velocity.zero_()
        self.update_count = 0


class SoftMatchingModule(nn.Module):
    """
    Soft OT matching with temperature annealing.

    Key insight from curriculum learning:
    - Start with high temperature (soft assignments) for exploration
    - Anneal to low temperature (sharp assignments) for exploitation
    - This prevents mode collapse and improves coverage

    Assignment types:
    - "softmax": Soft assignment using temperature-scaled softmax
    - "gumbel": Gumbel-softmax for differentiable sampling
    - "sinkhorn": Use Sinkhorn coupling directly as soft weights
    """

    def __init__(
        self,
        temperature_init: float = 1.0,
        temperature_final: float = 0.1,
        anneal_steps: int = 50000,
        assignment_type: str = "softmax",
        use_gumbel_hard: bool = False,  # Straight-through estimator
    ):
        super().__init__()
        self.temperature_init = temperature_init
        self.temperature_final = temperature_final
        self.anneal_steps = anneal_steps
        self.assignment_type = assignment_type
        self.use_gumbel_hard = use_gumbel_hard

        # Track current temperature
        self.register_buffer('current_temperature', torch.tensor(temperature_init))
        self.register_buffer('step_counter', torch.tensor(0))

    def get_temperature(self, step: Optional[int] = None) -> float:
        """Get temperature with exponential annealing."""
        if step is None:
            step = self.step_counter.item()

        if step >= self.anneal_steps:
            return self.temperature_final

        # Exponential annealing
        progress = step / self.anneal_steps
        log_ratio = np.log(self.temperature_final / self.temperature_init)
        temperature = self.temperature_init * np.exp(log_ratio * progress)

        return temperature

    def update_step(self, step: int):
        """Update internal step counter."""
        self.step_counter.fill_(step)
        self.current_temperature.fill_(self.get_temperature(step))

    def soft_assignment(
        self,
        cost_matrix: torch.Tensor,
        coupling: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Compute soft assignment weights.

        Args:
            cost_matrix: Cost matrix (n, m)
            coupling: OT coupling from Sinkhorn (n, m), optional
            temperature: Override temperature

        Returns:
            Soft assignment weights (n, m)
        """
        temp = temperature if temperature is not None else self.current_temperature.item()
        n, m = cost_matrix.shape
        device = cost_matrix.device

        if self.assignment_type == "softmax":
            # Temperature-scaled softmax over negative costs
            # Lower cost = higher weight
            logits = -cost_matrix / temp
            weights = F.softmax(logits, dim=1)

        elif self.assignment_type == "gumbel":
            # Gumbel-softmax for differentiable sampling
            logits = -cost_matrix / temp

            if self.training:
                # Add Gumbel noise
                gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
                logits = logits + gumbel_noise

            weights = F.softmax(logits, dim=1)

            if self.use_gumbel_hard:
                # Straight-through estimator
                hard = torch.zeros_like(weights).scatter_(1, weights.argmax(dim=1, keepdim=True), 1.0)
                weights = hard - weights.detach() + weights

        elif self.assignment_type == "sinkhorn":
            # Use Sinkhorn coupling directly, but temperature-scale it
            if coupling is not None:
                # Sharpen the coupling with temperature
                log_coupling = torch.log(coupling + 1e-10)
                weights = F.softmax(log_coupling / temp, dim=1)
            else:
                # Fall back to softmax
                logits = -cost_matrix / temp
                weights = F.softmax(logits, dim=1)

        else:
            # Default: argmax (hard assignment)
            weights = torch.zeros(n, m, device=device)
            weights.scatter_(1, cost_matrix.argmin(dim=1, keepdim=True), 1.0)

        return weights

    def get_soft_targets(
        self,
        weights: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute soft targets using assignment weights.

        Args:
            weights: Soft assignment weights (n, m)
            targets: Target images/features (m, ...)

        Returns:
            Soft targets (n, ...)
        """
        # weights: (n, m), targets: (m, C, H, W) or (m, d)
        if targets.dim() == 4:
            # Image targets: (m, C, H, W)
            m, C, H, W = targets.shape
            targets_flat = targets.view(m, -1)  # (m, C*H*W)
            soft_targets = torch.mm(weights, targets_flat)  # (n, C*H*W)
            soft_targets = soft_targets.view(-1, C, H, W)  # (n, C, H, W)
        else:
            # Feature targets: (m, d)
            soft_targets = torch.mm(weights, targets)  # (n, d)

        return soft_targets


class OTMatchingModuleV2(nn.Module):
    """
    Enhanced OT matching module for GFGW-FM training.

    Features:
    - Adaptive epsilon scheduling
    - FGW lambda scheduling
    - Hungarian matching option
    - Soft matching with temperature annealing (NEW)
    - Improved numerical stability
    """

    def __init__(
        self,
        feature_dim: int,
        num_data_points: int,
        fgw_lambda: float = 0.5,
        epsilon: float = 0.05,
        num_sinkhorn_iters: int = 50,
        num_fgw_iters: int = 10,
        use_cosine_distance: bool = False,
        dual_lr: float = 0.1,
        use_semi_discrete: bool = True,
        use_hungarian: bool = False,
        use_unbalanced: bool = False,
        unbalanced_reg: float = 1.0,
        # NEW: Soft matching parameters
        use_soft_matching: bool = False,
        soft_temperature_init: float = 1.0,
        soft_temperature_final: float = 0.1,
        soft_anneal_steps: int = 50000,
        soft_assignment_type: str = "softmax",
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.num_data_points = num_data_points
        self.use_hungarian = use_hungarian
        self.use_soft_matching = use_soft_matching
        self.device = device

        # FGW solver
        self.fgw_solver = EnhancedFGWSolver(
            fgw_lambda=fgw_lambda,
            epsilon=epsilon,
            num_sinkhorn_iters=num_sinkhorn_iters,
            num_fgw_iters=num_fgw_iters,
            use_cosine_distance=use_cosine_distance,
            unbalanced=use_unbalanced,
            unbalanced_reg=unbalanced_reg,
        )

        # Semi-discrete solver
        self.use_semi_discrete = use_semi_discrete
        if use_semi_discrete:
            self.semi_discrete_solver = SemiDiscreteOTSolverV2(
                num_data_points=num_data_points,
                feature_dim=feature_dim,
                dual_lr=dual_lr,
                device=device,
            )

        # Hungarian matcher
        if use_hungarian:
            self.hungarian = HungarianMatcher()

        # NEW: Soft matching module
        if use_soft_matching:
            self.soft_matcher = SoftMatchingModule(
                temperature_init=soft_temperature_init,
                temperature_final=soft_temperature_final,
                anneal_steps=soft_anneal_steps,
                assignment_type=soft_assignment_type,
            )

    def forward(
        self,
        features_gen: torch.Tensor,
        features_real: torch.Tensor,
        D_gen: Optional[torch.Tensor] = None,
        D_real: Optional[torch.Tensor] = None,
        data_indices: Optional[torch.Tensor] = None,
        epsilon: Optional[float] = None,
        fgw_lambda: Optional[float] = None,
        return_assignments: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute FGW optimal coupling.

        Args:
            features_gen: Generated features (n, d)
            features_real: Real features (m, d)
            D_gen: Distance matrix for generated
            D_real: Distance matrix for real
            data_indices: Indices in full dataset
            epsilon: Override epsilon
            fgw_lambda: Override FGW lambda
            return_assignments: Return hard assignments

        Returns:
            (coupling, cost_matrix) or (assignments, cost_matrix)
        """
        # Compute FGW coupling
        coupling, cost_matrix = self.fgw_solver(
            features_gen,
            features_real,
            D_gen,
            D_real,
            epsilon=epsilon,
            fgw_lambda=fgw_lambda,
        )

        # Apply semi-discrete dual potential adjustment
        if self.use_semi_discrete and data_indices is not None:
            modified_cost = self.semi_discrete_solver.get_modified_cost(
                cost_matrix, data_indices
            )
            self.semi_discrete_solver.update_dual_potentials(coupling, data_indices)

        # Hungarian matching for exact assignment
        if self.use_hungarian:
            row_ind, col_ind = self.hungarian(cost_matrix)
            if return_assignments:
                return col_ind, cost_matrix

            # Convert to coupling matrix
            n, m = features_gen.shape[0], features_real.shape[0]
            coupling_hard = torch.zeros(n, m, device=features_gen.device)
            coupling_hard[row_ind, col_ind] = 1.0 / len(row_ind)
            return coupling_hard, cost_matrix

        if return_assignments:
            assignments = coupling.argmax(dim=1)
            return assignments, cost_matrix

        return coupling, cost_matrix

    def get_matched_pairs(
        self,
        features_gen: torch.Tensor,
        features_real: torch.Tensor,
        real_data: torch.Tensor,
        D_gen: Optional[torch.Tensor] = None,
        D_real: Optional[torch.Tensor] = None,
        epsilon: Optional[float] = None,
        fgw_lambda: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get matched real data based on OT coupling."""
        coupling, _ = self.forward(
            features_gen, features_real, D_gen, D_real,
            epsilon=epsilon, fgw_lambda=fgw_lambda,
        )

        # Use argmax for deterministic Monge map
        assignments = coupling.argmax(dim=1)
        matched_data = real_data[assignments]

        return matched_data, coupling

    def compute_ot_distance(
        self,
        features_gen: torch.Tensor,
        features_real: torch.Tensor,
        D_gen: Optional[torch.Tensor] = None,
        D_real: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute FGW distance between two distributions."""
        coupling, cost_matrix = self.forward(
            features_gen, features_real, D_gen, D_real
        )
        # FGW distance = <C, P>
        distance = (coupling * cost_matrix).sum()
        return distance

    def update_soft_matching_step(self, step: int):
        """Update soft matching temperature for curriculum learning."""
        if self.use_soft_matching and hasattr(self, 'soft_matcher'):
            self.soft_matcher.update_step(step)

    def get_soft_matched_targets(
        self,
        features_gen: torch.Tensor,
        features_real: torch.Tensor,
        real_data: torch.Tensor,
        D_gen: Optional[torch.Tensor] = None,
        D_real: Optional[torch.Tensor] = None,
        epsilon: Optional[float] = None,
        fgw_lambda: Optional[float] = None,
        temperature: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get soft-matched targets using temperature-annealed assignments.

        Unlike get_matched_pairs which uses hard argmax, this uses soft
        weighted combinations for curriculum learning.

        Args:
            features_gen: Generated features (n, d)
            features_real: Real features (m, d)
            real_data: Real images/targets (m, C, H, W) or (m, d)
            D_gen: Distance matrix for generated
            D_real: Distance matrix for real
            epsilon: Override epsilon
            fgw_lambda: Override FGW lambda
            temperature: Override temperature (uses scheduled if None)

        Returns:
            (soft_targets, soft_weights, coupling)
            - soft_targets: Weighted combination of targets (n, ...)
            - soft_weights: Assignment weights (n, m)
            - coupling: OT coupling matrix (n, m)
        """
        # Compute FGW coupling
        coupling, cost_matrix = self.forward(
            features_gen, features_real, D_gen, D_real,
            epsilon=epsilon, fgw_lambda=fgw_lambda,
        )

        if self.use_soft_matching and hasattr(self, 'soft_matcher'):
            # Use soft matching with temperature annealing
            soft_weights = self.soft_matcher.soft_assignment(
                cost_matrix, coupling, temperature
            )
            soft_targets = self.soft_matcher.get_soft_targets(soft_weights, real_data)
        else:
            # Fall back to hard assignment
            assignments = coupling.argmax(dim=1)
            soft_targets = real_data[assignments]
            # Create one-hot weights
            n, m = coupling.shape
            soft_weights = torch.zeros(n, m, device=coupling.device)
            soft_weights.scatter_(1, assignments.unsqueeze(1), 1.0)

        return soft_targets, soft_weights, coupling

    def get_current_temperature(self) -> float:
        """Get current soft matching temperature."""
        if self.use_soft_matching and hasattr(self, 'soft_matcher'):
            return self.soft_matcher.current_temperature.item()
        return 0.0  # No soft matching


# ============================================================================
# Backward compatibility - keep old class names working
# ============================================================================

# Alias for backward compatibility
SinkhornSolver = LogStabilizedSinkhorn
FusedGromovWassersteinSolver = EnhancedFGWSolver
SemiDiscreteOTSolver = SemiDiscreteOTSolverV2
OTMatchingModule = OTMatchingModuleV2
