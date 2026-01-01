"""Training schedules and utilities for GFGW-FM.

Implements various scheduling strategies from:
- TCM: Time sampling distributions, two-stage training
- SlimFlow: Annealing schedules
- ECM: Continuous time schedules
"""

import torch
import numpy as np
from typing import Optional, Tuple, Callable
import math


# ============================================================================
# Time Sampling Distributions
# ============================================================================

class TimeSampler:
    """
    Time sampler with various distributions.

    Supports:
    - uniform: Standard uniform distribution
    - logit_normal: Logit-normal distribution (from EDM)
    - logit_student_t: Heavy-tailed Student-t (from TCM)
    - beta: Beta distribution for more control
    - stratified: Stratified sampling for better coverage
    """

    def __init__(
        self,
        sampling_type: str = "logit_student_t",
        t_min: float = 0.001,
        t_max: float = 1.0,
        # Logit-normal parameters
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
        # Student-t parameters
        student_t_df: float = 1.0,  # Degrees of freedom
        # Beta parameters
        beta_a: float = 1.0,
        beta_b: float = 1.0,
    ):
        self.sampling_type = sampling_type
        self.t_min = t_min
        self.t_max = t_max
        self.logit_mean = logit_mean
        self.logit_std = logit_std
        self.student_t_df = student_t_df
        self.beta_a = beta_a
        self.beta_b = beta_b

    def sample(
        self,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Sample time values.

        Args:
            batch_size: Number of samples
            device: Target device

        Returns:
            Time values of shape (batch_size,) in [t_min, t_max]
        """
        if self.sampling_type == "uniform":
            t = self._sample_uniform(batch_size, device)
        elif self.sampling_type == "logit_normal":
            t = self._sample_logit_normal(batch_size, device)
        elif self.sampling_type == "logit_student_t":
            t = self._sample_logit_student_t(batch_size, device)
        elif self.sampling_type == "beta":
            t = self._sample_beta(batch_size, device)
        elif self.sampling_type == "stratified":
            t = self._sample_stratified(batch_size, device)
        elif self.sampling_type == "cosine":
            t = self._sample_cosine(batch_size, device)
        else:
            raise ValueError(f"Unknown sampling type: {self.sampling_type}")

        # Clamp to valid range
        t = t.clamp(self.t_min, self.t_max)

        return t

    def _sample_uniform(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Uniform sampling."""
        return torch.rand(batch_size, device=device) * (self.t_max - self.t_min) + self.t_min

    def _sample_logit_normal(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Logit-normal sampling from EDM.

        Sample z ~ N(mean, std), then t = sigmoid(z)
        """
        z = torch.randn(batch_size, device=device) * self.logit_std + self.logit_mean
        t = torch.sigmoid(z)
        return t * (self.t_max - self.t_min) + self.t_min

    def _sample_logit_student_t(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Logit-Student-t sampling from TCM.

        Heavy-tailed distribution - samples more boundary regions.
        """
        # Sample from Student-t distribution
        student_t = torch.distributions.StudentT(df=self.student_t_df)
        z = student_t.sample((batch_size,)).to(device)

        # Apply sigmoid to map to [0, 1]
        t = torch.sigmoid(z)

        # Scale to [t_min, t_max]
        return t * (self.t_max - self.t_min) + self.t_min

    def _sample_beta(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Beta distribution sampling."""
        beta = torch.distributions.Beta(self.beta_a, self.beta_b)
        t = beta.sample((batch_size,)).to(device)
        return t * (self.t_max - self.t_min) + self.t_min

    def _sample_stratified(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Stratified sampling for better coverage."""
        # Divide [t_min, t_max] into batch_size strata
        boundaries = torch.linspace(self.t_min, self.t_max, batch_size + 1, device=device)

        # Sample uniformly within each stratum
        t = torch.zeros(batch_size, device=device)
        for i in range(batch_size):
            t[i] = torch.rand(1, device=device) * (boundaries[i+1] - boundaries[i]) + boundaries[i]

        # Shuffle to avoid correlation with batch index
        perm = torch.randperm(batch_size, device=device)
        return t[perm]

    def _sample_cosine(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Cosine schedule sampling (more samples near boundaries)."""
        u = torch.rand(batch_size, device=device)
        # Cosine warping - more density near 0 and 1
        t = (1 - torch.cos(u * math.pi)) / 2
        return t * (self.t_max - self.t_min) + self.t_min


# ============================================================================
# Annealing Schedules
# ============================================================================

class AnnealingSchedule:
    """
    Annealing schedule for various training parameters.

    From SlimFlow: Gradual transition from soft to hard objectives.
    """

    def __init__(
        self,
        schedule_type: str = "cosine",
        start_value: float = 0.0,
        end_value: float = 1.0,
        warmup_steps: int = 50000,
        total_steps: Optional[int] = None,
    ):
        self.schedule_type = schedule_type
        self.start_value = start_value
        self.end_value = end_value
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps or warmup_steps

    def get_value(self, step: int) -> float:
        """
        Get scheduled value at given step.

        Args:
            step: Current training step

        Returns:
            Scheduled value
        """
        if step >= self.warmup_steps:
            return self.end_value

        progress = step / self.warmup_steps

        if self.schedule_type == "linear":
            alpha = progress
        elif self.schedule_type == "cosine":
            alpha = (1 - math.cos(progress * math.pi)) / 2
        elif self.schedule_type == "exponential":
            alpha = 1 - math.exp(-5 * progress)
        elif self.schedule_type == "quadratic":
            alpha = progress ** 2
        elif self.schedule_type == "sqrt":
            alpha = math.sqrt(progress)
        else:
            alpha = progress

        return self.start_value + (self.end_value - self.start_value) * alpha


class TwoStageSchedule:
    """
    Two-stage training schedule from TCM.

    Stage 1: Focus on boundary region [t_high, 1.0]
    Stage 2: Full range [t_min, 1.0]
    """

    def __init__(
        self,
        stage1_t_min: float = 0.7,
        stage1_t_max: float = 1.0,
        stage2_t_min: float = 0.001,
        stage2_t_max: float = 1.0,
        stage1_steps: int = 20000,
        transition_steps: int = 5000,  # Smooth transition between stages
    ):
        self.stage1_t_min = stage1_t_min
        self.stage1_t_max = stage1_t_max
        self.stage2_t_min = stage2_t_min
        self.stage2_t_max = stage2_t_max
        self.stage1_steps = stage1_steps
        self.transition_steps = transition_steps

    def get_time_range(self, step: int) -> Tuple[float, float]:
        """
        Get time range for current step.

        Args:
            step: Current training step

        Returns:
            (t_min, t_max) for current training phase
        """
        if step < self.stage1_steps:
            # Stage 1: Boundary focus
            return self.stage1_t_min, self.stage1_t_max

        elif step < self.stage1_steps + self.transition_steps:
            # Transition: Gradually expand range
            progress = (step - self.stage1_steps) / self.transition_steps
            # Smooth cosine transition
            alpha = (1 - math.cos(progress * math.pi)) / 2

            t_min = self.stage1_t_min + (self.stage2_t_min - self.stage1_t_min) * alpha
            t_max = self.stage1_t_max + (self.stage2_t_max - self.stage1_t_max) * alpha

            return t_min, t_max

        else:
            # Stage 2: Full range
            return self.stage2_t_min, self.stage2_t_max


# ============================================================================
# Learning Rate Schedules
# ============================================================================

class LRScheduler:
    """
    Learning rate scheduler with warmup.
    """

    def __init__(
        self,
        base_lr: float,
        schedule_type: str = "cosine",
        warmup_steps: int = 10000,
        total_steps: int = 100000,
        min_lr_ratio: float = 0.01,
    ):
        self.base_lr = base_lr
        self.schedule_type = schedule_type
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = base_lr * min_lr_ratio

    def get_lr(self, step: int) -> float:
        """
        Get learning rate at given step.

        Args:
            step: Current training step

        Returns:
            Learning rate
        """
        # Warmup phase
        if step < self.warmup_steps:
            return self.base_lr * step / self.warmup_steps

        # After warmup
        if self.schedule_type == "constant":
            return self.base_lr

        # Progress after warmup
        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        progress = min(1.0, progress)

        if self.schedule_type == "cosine":
            # Cosine annealing
            lr = self.min_lr + (self.base_lr - self.min_lr) * (1 + math.cos(progress * math.pi)) / 2

        elif self.schedule_type == "linear":
            # Linear decay
            lr = self.base_lr - (self.base_lr - self.min_lr) * progress

        elif self.schedule_type == "exponential":
            # Exponential decay
            decay_rate = math.log(self.min_lr / self.base_lr)
            lr = self.base_lr * math.exp(decay_rate * progress)

        elif self.schedule_type == "polynomial":
            # Polynomial decay (power=2)
            lr = (self.base_lr - self.min_lr) * (1 - progress) ** 2 + self.min_lr

        else:
            lr = self.base_lr

        return lr

    def step(self, optimizer: torch.optim.Optimizer, step: int):
        """Update optimizer learning rate."""
        lr = self.get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr


# ============================================================================
# OT Schedule (Epsilon annealing)
# ============================================================================

class OTEpsilonSchedule:
    """
    Epsilon (entropic regularization) annealing for OT solver.

    Start with large epsilon (smooth OT) and anneal to small epsilon (sharp OT).
    """

    def __init__(
        self,
        epsilon_init: float = 0.1,
        epsilon_final: float = 0.01,
        decay_steps: int = 50000,
        schedule_type: str = "exponential",
    ):
        self.epsilon_init = epsilon_init
        self.epsilon_final = epsilon_final
        self.decay_steps = decay_steps
        self.schedule_type = schedule_type

    def get_epsilon(self, step: int) -> float:
        """Get epsilon value at current step."""
        if step >= self.decay_steps:
            return self.epsilon_final

        progress = step / self.decay_steps

        if self.schedule_type == "exponential":
            # Exponential decay
            log_ratio = math.log(self.epsilon_final / self.epsilon_init)
            epsilon = self.epsilon_init * math.exp(log_ratio * progress)

        elif self.schedule_type == "linear":
            epsilon = self.epsilon_init + (self.epsilon_final - self.epsilon_init) * progress

        elif self.schedule_type == "cosine":
            alpha = (1 - math.cos(progress * math.pi)) / 2
            epsilon = self.epsilon_init + (self.epsilon_final - self.epsilon_init) * alpha

        else:
            epsilon = self.epsilon_init

        return epsilon


# ============================================================================
# FGW Lambda Schedule
# ============================================================================

class FGWLambdaSchedule:
    """
    Schedule for FGW lambda (structure vs feature weight).

    Start with emphasis on features, gradually increase structure weight.
    """

    def __init__(
        self,
        lambda_start: float = 0.1,
        lambda_end: float = 0.5,
        warmup_steps: int = 20000,
        schedule_type: str = "linear",
    ):
        self.lambda_start = lambda_start
        self.lambda_end = lambda_end
        self.warmup_steps = warmup_steps
        self.schedule_type = schedule_type

    def get_lambda(self, step: int) -> float:
        """Get FGW lambda at current step."""
        if step >= self.warmup_steps:
            return self.lambda_end

        progress = step / self.warmup_steps

        if self.schedule_type == "linear":
            alpha = progress
        elif self.schedule_type == "cosine":
            alpha = (1 - math.cos(progress * math.pi)) / 2
        elif self.schedule_type == "step":
            # Step function at midpoint
            alpha = 1.0 if progress > 0.5 else 0.0
        else:
            alpha = progress

        return self.lambda_start + (self.lambda_end - self.lambda_start) * alpha


# ============================================================================
# Comprehensive Training Scheduler
# ============================================================================

class TrainingScheduler:
    """
    Unified training scheduler combining all scheduling components.
    """

    def __init__(self, config):
        """Initialize from config."""
        self.config = config

        # Time sampler
        self.time_sampler = TimeSampler(
            sampling_type=config.schedule.time_sampling,
            t_min=config.schedule.t_min,
            t_max=config.schedule.t_max,
            logit_mean=config.schedule.logit_mean,
            logit_std=config.schedule.logit_std,
            student_t_df=config.schedule.student_t_df,
        )

        # Two-stage schedule
        if config.schedule.use_two_stage:
            self.two_stage = TwoStageSchedule(
                stage1_t_min=config.schedule.stage1_t_min,
                stage1_t_max=config.schedule.stage1_t_max,
                stage2_t_min=config.schedule.t_min,
                stage2_t_max=config.schedule.t_max,
                stage1_steps=config.schedule.stage1_kimg * 1000,
            )
        else:
            self.two_stage = None

        # Annealing schedules
        if config.schedule.use_annealing:
            self.annealing = AnnealingSchedule(
                schedule_type=config.schedule.annealing_type,
                start_value=config.schedule.ot_annealing_start,
                end_value=config.schedule.ot_annealing_end,
                warmup_steps=config.schedule.annealing_warmup_kimg * 1000,
            )
        else:
            self.annealing = None

        # LR scheduler
        if config.training.use_lr_schedule:
            self.lr_scheduler = LRScheduler(
                base_lr=config.training.lr,
                schedule_type=config.training.lr_schedule,
                warmup_steps=config.training.lr_warmup_kimg * 1000,
                total_steps=config.training.total_kimg * 1000,
                min_lr_ratio=config.training.lr_min_ratio,
            )
        else:
            self.lr_scheduler = None

        # Epsilon schedule
        if config.ot.use_adaptive_epsilon:
            self.epsilon_schedule = OTEpsilonSchedule(
                epsilon_init=config.ot.epsilon_init,
                epsilon_final=config.ot.epsilon_final,
                decay_steps=config.ot.epsilon_decay_kimg * 1000,
            )
        else:
            self.epsilon_schedule = None

        # FGW lambda schedule
        if config.schedule.fgw_lambda_annealing:
            self.fgw_lambda_schedule = FGWLambdaSchedule(
                lambda_start=config.schedule.fgw_lambda_start,
                lambda_end=config.schedule.fgw_lambda_end,
                warmup_steps=config.schedule.annealing_warmup_kimg * 1000,
            )
        else:
            self.fgw_lambda_schedule = None

    def sample_time(
        self,
        batch_size: int,
        device: torch.device,
        step: int,
    ) -> torch.Tensor:
        """Sample time values with current schedule."""
        # Get time range based on training stage
        if self.two_stage is not None:
            t_min, t_max = self.two_stage.get_time_range(step)
            self.time_sampler.t_min = t_min
            self.time_sampler.t_max = t_max

        return self.time_sampler.sample(batch_size, device)

    def get_lr(self, step: int) -> float:
        """Get learning rate for current step."""
        if self.lr_scheduler is not None:
            return self.lr_scheduler.get_lr(step)
        return self.config.training.lr

    def get_epsilon(self, step: int) -> float:
        """Get OT epsilon for current step."""
        if self.epsilon_schedule is not None:
            return self.epsilon_schedule.get_epsilon(step)
        return self.config.ot.epsilon

    def get_fgw_lambda(self, step: int) -> float:
        """Get FGW lambda for current step."""
        if self.fgw_lambda_schedule is not None:
            return self.fgw_lambda_schedule.get_lambda(step)
        return self.config.ot.fgw_lambda

    def get_annealing_value(self, step: int) -> float:
        """Get general annealing value."""
        if self.annealing is not None:
            return self.annealing.get_value(step)
        return 1.0

    def step_optimizer(self, optimizer: torch.optim.Optimizer, step: int) -> float:
        """Update optimizer LR and return current LR."""
        if self.lr_scheduler is not None:
            return self.lr_scheduler.step(optimizer, step)
        return self.config.training.lr

    def get_state(self, step: int) -> dict:
        """Get current state of all schedules."""
        return {
            'step': step,
            'lr': self.get_lr(step),
            'epsilon': self.get_epsilon(step),
            'fgw_lambda': self.get_fgw_lambda(step),
            'annealing': self.get_annealing_value(step),
            'time_range': self.two_stage.get_time_range(step) if self.two_stage else (self.config.schedule.t_min, self.config.schedule.t_max),
        }
