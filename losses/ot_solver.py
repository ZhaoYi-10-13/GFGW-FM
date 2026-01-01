"""Fused Gromov-Wasserstein Optimal Transport solver for GFGW-FM.

Implements the FGW-based OT algorithm from the paper:
"Rectifying the Manifold: High-Fidelity One-Step Generation via
Global Fused Gromov-Wasserstein Flow Matching"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np


class SinkhornSolver:
    """
    Log-stabilized Sinkhorn algorithm for entropy-regularized OT.

    Based on: "Sinkhorn Distances: Lightspeed Computation of Optimal Transport"
    """

    def __init__(
        self,
        epsilon: float = 0.05,
        max_iters: int = 100,
        threshold: float = 1e-6,
    ):
        self.epsilon = epsilon
        self.max_iters = max_iters
        self.threshold = threshold

    def __call__(
        self,
        cost_matrix: torch.Tensor,
        a: Optional[torch.Tensor] = None,
        b: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Solve OT using log-stabilized Sinkhorn algorithm.

        Args:
            cost_matrix: Cost matrix C of shape (n, m)
            a: Source distribution (n,), uniform if None
            b: Target distribution (m,), uniform if None

        Returns:
            Transport plan P of shape (n, m)
        """
        n, m = cost_matrix.shape
        device = cost_matrix.device
        dtype = cost_matrix.dtype

        if a is None:
            a = torch.ones(n, device=device, dtype=dtype) / n
        if b is None:
            b = torch.ones(m, device=device, dtype=dtype) / m

        # Normalize cost matrix for stability
        cost_max = cost_matrix.max()
        if cost_max > 0:
            C_normalized = cost_matrix / cost_max
        else:
            C_normalized = cost_matrix

        # Initialize dual potentials in log space
        log_a = torch.log(a + 1e-10)
        log_b = torch.log(b + 1e-10)

        # Log kernel: log K = -C / epsilon
        log_K = -C_normalized / self.epsilon

        # Initialize
        u = torch.zeros(n, device=device, dtype=dtype)
        v = torch.zeros(m, device=device, dtype=dtype)

        for iteration in range(self.max_iters):
            u_prev = u.clone()

            # Row normalization: u = log(a) - logsumexp(log_K + v)
            u = log_a - torch.logsumexp(log_K + v.unsqueeze(0), dim=1)

            # Column normalization: v = log(b) - logsumexp(log_K.T + u)
            v = log_b - torch.logsumexp(log_K.T + u.unsqueeze(0), dim=1)

            # Check convergence
            if (u - u_prev).abs().max() < self.threshold:
                break

        # Compute transport plan: P = diag(exp(u)) @ K @ diag(exp(v))
        log_P = u.unsqueeze(1) + log_K + v.unsqueeze(0)
        P = torch.exp(log_P)

        # Ensure valid marginals (numerical stability)
        P = P / (P.sum() + 1e-10) * min(a.sum(), b.sum())

        return P


class FusedGromovWassersteinSolver:
    """
    Fused Gromov-Wasserstein optimal transport solver.

    Implements the FGW distance from:
    "Optimal Transport for structured data with application on graphs"

    FGW combines:
    1. Wasserstein cost: C_F(x_i, y_j) = ||f(x_i) - f(y_j)||^2
    2. Gromov-Wasserstein cost: sum |D_X[i,i'] - D_Y[j,j']|^2 * P[i,j] * P[i',j']
    """

    def __init__(
        self,
        fgw_lambda: float = 0.5,
        epsilon: float = 0.05,
        num_sinkhorn_iters: int = 50,
        num_fgw_iters: int = 10,
        use_cosine_distance: bool = False,
    ):
        """
        Initialize FGW solver.

        Args:
            fgw_lambda: Trade-off: 0=pure Wasserstein, 1=pure GW
            epsilon: Entropic regularization strength
            num_sinkhorn_iters: Sinkhorn iterations per FGW iteration
            num_fgw_iters: Outer FGW iterations (block coordinate descent)
            use_cosine_distance: Use cosine instead of Euclidean distance
        """
        self.fgw_lambda = fgw_lambda
        self.epsilon = epsilon
        self.num_sinkhorn_iters = num_sinkhorn_iters
        self.num_fgw_iters = num_fgw_iters
        self.use_cosine_distance = use_cosine_distance

        self.sinkhorn = SinkhornSolver(
            epsilon=epsilon,
            max_iters=num_sinkhorn_iters
        )

    def compute_feature_cost(
        self,
        features_gen: torch.Tensor,
        features_real: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Wasserstein (feature-based) cost matrix.

        C_F[i,j] = ||f(x_i) - f(y_j)||^2 or 1 - cos(f(x_i), f(y_j))
        """
        if self.use_cosine_distance:
            features_gen = F.normalize(features_gen, dim=-1)
            features_real = F.normalize(features_real, dim=-1)
            similarity = torch.mm(features_gen, features_real.T)
            cost = 1 - similarity
        else:
            # Squared Euclidean distance (standard for W2)
            cost = torch.cdist(features_gen, features_real, p=2) ** 2

        return cost

    def compute_gw_gradient(
        self,
        D_gen: torch.Tensor,
        D_real: torch.Tensor,
        coupling: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the gradient of GW cost w.r.t. coupling.

        This is the linearized cost matrix for the GW term.

        The GW cost is: sum_{i,j,i',j'} |D_gen[i,i'] - D_real[j,j']|^2 * P[i,j] * P[i',j']

        The gradient w.r.t. P[i,j] is:
        2 * sum_{i',j'} |D_gen[i,i'] - D_real[j,j']|^2 * P[i',j']
        = 2 * (D_gen^2 @ 1 @ nu^T + mu @ 1^T @ D_real^2 - 2 * D_gen @ P @ D_real^T)

        where mu = P @ 1, nu = P^T @ 1
        """
        # Marginals
        mu = coupling.sum(dim=1, keepdim=True)  # (n, 1)
        nu = coupling.sum(dim=0, keepdim=True)  # (1, m)

        # Terms of the quadratic expansion
        # |D_gen - D_real|^2 = D_gen^2 + D_real^2 - 2*D_gen*D_real

        # First term: D_gen^2 contribution
        D_gen_sq_sum = (D_gen ** 2).sum(dim=1, keepdim=True)  # (n, 1)
        term1 = D_gen_sq_sum @ nu  # (n, m)

        # Second term: D_real^2 contribution
        D_real_sq_sum = (D_real ** 2).sum(dim=0, keepdim=True)  # (1, m)
        term2 = mu @ D_real_sq_sum  # (n, m)

        # Third term: Cross term (most important for structure preservation)
        term3 = 2 * torch.mm(torch.mm(D_gen, coupling), D_real.T)  # (n, m)

        # Gradient of GW cost
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Solve Fused Gromov-Wasserstein OT problem.

        Uses block coordinate descent:
        1. Fix P, compute linearized cost
        2. Solve entropic OT with current cost
        3. Repeat

        Args:
            features_gen: Generated features (n, d)
            features_real: Real features (m, d)
            D_gen: Distance matrix for generated (n, n)
            D_real: Distance matrix for real (m, m)
            a: Source marginal (n,)
            b: Target marginal (m,)

        Returns:
            (optimal_coupling, final_cost_matrix)
        """
        n = features_gen.shape[0]
        m = features_real.shape[0]
        device = features_gen.device
        dtype = features_gen.dtype

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

        # Compute feature cost (Wasserstein term)
        C_W = self.compute_feature_cost(features_gen, features_real)

        # Normalize costs for numerical stability
        C_W_max = C_W.max()
        if C_W_max > 0:
            C_W_norm = C_W / C_W_max
        else:
            C_W_norm = C_W

        # Initialize coupling with independent product
        coupling = a.unsqueeze(1) * b.unsqueeze(0)

        # FGW iterations (block coordinate descent)
        for iteration in range(self.num_fgw_iters):
            # Compute GW gradient (linearized structure cost)
            C_GW = self.compute_gw_gradient(D_gen, D_real, coupling)

            # Normalize GW cost
            C_GW_max = C_GW.abs().max()
            if C_GW_max > 0:
                C_GW_norm = C_GW / C_GW_max
            else:
                C_GW_norm = C_GW

            # Fused cost
            C_fused = (1 - self.fgw_lambda) * C_W_norm + self.fgw_lambda * C_GW_norm

            # Solve entropic OT
            coupling = self.sinkhorn(C_fused, a, b)

        # Return unnormalized cost for monitoring
        C_final = (1 - self.fgw_lambda) * C_W + self.fgw_lambda * C_GW

        return coupling, C_final


class SemiDiscreteOTSolver:
    """
    Semi-discrete OT solver with online dual potential updates.

    Implements the dual update algorithm from the paper:
    phi_j <- phi_j + eta * (m_j - 1/N)

    where m_j is the mass assigned to data point j.
    """

    def __init__(
        self,
        num_data_points: int,
        feature_dim: int,
        dual_lr: float = 0.1,
        epsilon: float = 0.05,
        device: torch.device = torch.device("cuda"),
    ):
        self.num_data_points = num_data_points
        self.feature_dim = feature_dim
        self.dual_lr = dual_lr
        self.epsilon = epsilon
        self.device = device

        # Dual potentials phi (one per data point)
        self.phi = torch.zeros(num_data_points, device=device)

        # Running average for momentum-based updates
        self.phi_momentum = 0.9

    @torch.no_grad()
    def update_dual_potentials(
        self,
        features_gen: torch.Tensor,
        features_real: torch.Tensor,
        coupling: torch.Tensor,
        data_indices: Optional[torch.Tensor] = None,
    ):
        """
        Update dual potentials based on marginal constraint violation.

        phi_j <- phi_j - eta * (sum_i P[i,j] - 1/N)
        """
        m = features_real.shape[0]

        # Target marginal (uniform over data)
        target_marginal = torch.ones(m, device=self.device) / m

        # Actual marginal from coupling
        actual_marginal = coupling.sum(dim=0)

        # Gradient: difference from target
        grad_phi = actual_marginal - target_marginal

        # Update with indices if provided
        if data_indices is not None:
            self.phi[data_indices] = self.phi[data_indices] - self.dual_lr * grad_phi
        else:
            self.phi[:m] = self.phi[:m] - self.dual_lr * grad_phi

    def get_modified_cost(
        self,
        cost_matrix: torch.Tensor,
        indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Modify cost matrix with dual potentials.

        C_modified[i,j] = C[i,j] - phi[j]
        """
        if indices is not None:
            phi = self.phi[indices]
        else:
            phi = self.phi[:cost_matrix.shape[1]]

        return cost_matrix - phi.unsqueeze(0)


class OTMatchingModule(nn.Module):
    """
    Complete OT matching module for GFGW-FM training.

    Combines FGW solver with semi-discrete dual updates.
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
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__()

        self.fgw_solver = FusedGromovWassersteinSolver(
            fgw_lambda=fgw_lambda,
            epsilon=epsilon,
            num_sinkhorn_iters=num_sinkhorn_iters,
            num_fgw_iters=num_fgw_iters,
            use_cosine_distance=use_cosine_distance,
        )

        self.use_semi_discrete = use_semi_discrete
        if use_semi_discrete:
            self.semi_discrete_solver = SemiDiscreteOTSolver(
                num_data_points=num_data_points,
                feature_dim=feature_dim,
                dual_lr=dual_lr,
                epsilon=epsilon,
                device=device,
            )

        self.device = device

    def forward(
        self,
        features_gen: torch.Tensor,
        features_real: torch.Tensor,
        D_gen: Optional[torch.Tensor] = None,
        D_real: Optional[torch.Tensor] = None,
        data_indices: Optional[torch.Tensor] = None,
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
            return_assignments: Return hard assignments

        Returns:
            (coupling, cost_matrix)
        """
        # Compute FGW coupling
        coupling, cost_matrix = self.fgw_solver(
            features_gen, features_real, D_gen, D_real
        )

        # Apply dual potential modification if using semi-discrete
        if self.use_semi_discrete and data_indices is not None:
            # Modify coupling based on dual potentials
            modified_cost = self.semi_discrete_solver.get_modified_cost(
                cost_matrix, data_indices
            )

            # Update dual potentials for next iteration
            self.semi_discrete_solver.update_dual_potentials(
                features_gen, features_real, coupling, data_indices
            )

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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get matched real data based on OT coupling."""
        coupling, _ = self.forward(
            features_gen, features_real, D_gen, D_real
        )

        # Sample from coupling (or use argmax for Monge map)
        # Using argmax for deterministic mapping as per paper
        assignments = coupling.argmax(dim=1)

        matched_data = real_data[assignments]

        return matched_data, coupling
