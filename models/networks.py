"""Neural network architectures for GFGW-FM one-step generator."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any


def weight_init(shape, mode, fan_in, fan_out):
    """Initialize weights with various methods."""
    if mode == 'xavier_uniform':
        return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == 'xavier_normal':
        return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == 'kaiming_uniform':
        return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == 'kaiming_normal':
        return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')


class Linear(nn.Module):
    """Fully-connected layer with custom initialization."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        init_mode: str = 'kaiming_normal',
        init_weight: float = 1.0,
        init_bias: float = 0.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = nn.Parameter(
            weight_init([out_features, in_features], **init_kwargs) * init_weight
        )
        self.bias = nn.Parameter(
            weight_init([out_features], **init_kwargs) * init_bias
        ) if bias else None

    def forward(self, x):
        x = x @ self.weight.to(x.dtype).t()
        if self.bias is not None:
            x = x.add_(self.bias.to(x.dtype))
        return x


class Conv2d(nn.Module):
    """Convolutional layer with optional up/downsampling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: int,
        bias: bool = True,
        up: bool = False,
        down: bool = False,
        resample_filter: List[int] = [1, 1],
        fused_resample: bool = False,
        init_mode: str = 'kaiming_normal',
        init_weight: float = 1.0,
        init_bias: float = 0.0,
    ):
        assert not (up and down)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.fused_resample = fused_resample
        init_kwargs = dict(
            mode=init_mode,
            fan_in=in_channels * kernel * kernel,
            fan_out=out_channels * kernel * kernel
        )
        self.weight = nn.Parameter(
            weight_init([out_channels, in_channels, kernel, kernel], **init_kwargs) * init_weight
        ) if kernel else None
        self.bias = nn.Parameter(
            weight_init([out_channels], **init_kwargs) * init_bias
        ) if kernel and bias else None
        f = torch.as_tensor(resample_filter, dtype=torch.float32)
        f = f.ger(f).unsqueeze(0).unsqueeze(1) / f.sum().square()
        self.register_buffer('resample_filter', f if up or down else None)

    def forward(self, x):
        w = self.weight.to(x.dtype) if self.weight is not None else None
        b = self.bias.to(x.dtype) if self.bias is not None else None
        f = self.resample_filter.to(x.dtype) if self.resample_filter is not None else None
        w_pad = w.shape[-1] // 2 if w is not None else 0
        f_pad = (f.shape[-1] - 1) // 2 if f is not None else 0

        if self.fused_resample and self.up and w is not None:
            x = F.conv_transpose2d(
                x, f.mul(4).tile([self.in_channels, 1, 1, 1]),
                groups=self.in_channels, stride=2, padding=max(f_pad - w_pad, 0)
            )
            x = F.conv2d(x, w, padding=max(w_pad - f_pad, 0))
        elif self.fused_resample and self.down and w is not None:
            x = F.conv2d(x, w, padding=w_pad + f_pad)
            x = F.conv2d(
                x, f.tile([self.out_channels, 1, 1, 1]),
                groups=self.out_channels, stride=2
            )
        else:
            if self.up:
                x = F.conv_transpose2d(
                    x, f.mul(4).tile([self.in_channels, 1, 1, 1]),
                    groups=self.in_channels, stride=2, padding=f_pad
                )
            if self.down:
                x = F.conv2d(
                    x, f.tile([self.in_channels, 1, 1, 1]),
                    groups=self.in_channels, stride=2, padding=f_pad
                )
            if w is not None:
                x = F.conv2d(x, w, padding=w_pad)
        if b is not None:
            x = x.add_(b.reshape(1, -1, 1, 1))
        return x


class GroupNorm(nn.Module):
    """Group normalization."""

    def __init__(
        self,
        num_channels: int,
        num_groups: int = 32,
        min_channels_per_group: int = 4,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.num_groups = min(num_groups, num_channels // min_channels_per_group)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        x = F.group_norm(
            x,
            num_groups=self.num_groups,
            weight=self.weight.to(x.dtype),
            bias=self.bias.to(x.dtype),
            eps=self.eps
        )
        return x


class AttentionOp(torch.autograd.Function):
    """Attention weight computation with FP32 precision."""

    @staticmethod
    def forward(ctx, q, k):
        w = torch.einsum(
            'ncq,nck->nqk',
            q.to(torch.float32),
            (k / np.sqrt(k.shape[1])).to(torch.float32)
        ).softmax(dim=2).to(q.dtype)
        ctx.save_for_backward(q, k, w)
        return w

    @staticmethod
    def backward(ctx, dw):
        q, k, w = ctx.saved_tensors
        db = torch._softmax_backward_data(
            grad_output=dw.to(torch.float32),
            output=w.to(torch.float32),
            dim=2,
            input_dtype=torch.float32
        )
        dq = torch.einsum('nck,nqk->ncq', k.to(torch.float32), db).to(q.dtype) / np.sqrt(k.shape[1])
        dk = torch.einsum('ncq,nqk->nck', q.to(torch.float32), db).to(k.dtype) / np.sqrt(k.shape[1])
        return dq, dk


class UNetBlock(nn.Module):
    """Unified U-Net block with optional up/downsampling and self-attention."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_channels: int,
        up: bool = False,
        down: bool = False,
        attention: bool = False,
        num_heads: Optional[int] = None,
        channels_per_head: int = 64,
        dropout: float = 0.0,
        skip_scale: float = 1.0,
        eps: float = 1e-5,
        resample_filter: List[int] = [1, 1],
        resample_proj: bool = False,
        adaptive_scale: bool = True,
        init: Dict = None,
        init_zero: Dict = None,
        init_attn: Dict = None,
    ):
        super().__init__()
        init = init or dict()
        init_zero = init_zero or dict(init_weight=0)
        init_attn = init_attn or init

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.num_heads = 0 if not attention else (
            num_heads if num_heads is not None else out_channels // channels_per_head
        )
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale

        self.norm0 = GroupNorm(num_channels=in_channels, eps=eps)
        self.conv0 = Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel=3,
            up=up, down=down, resample_filter=resample_filter, **init
        )
        self.affine = Linear(
            in_features=emb_channels,
            out_features=out_channels * (2 if adaptive_scale else 1),
            **init
        )
        self.norm1 = GroupNorm(num_channels=out_channels, eps=eps)
        self.conv1 = Conv2d(
            in_channels=out_channels, out_channels=out_channels, kernel=3,
            **init_zero
        )

        self.skip = None
        if out_channels != in_channels or up or down:
            kernel = 1 if resample_proj or out_channels != in_channels else 0
            self.skip = Conv2d(
                in_channels=in_channels, out_channels=out_channels, kernel=kernel,
                up=up, down=down, resample_filter=resample_filter, **init
            )

        if self.num_heads:
            self.norm2 = GroupNorm(num_channels=out_channels, eps=eps)
            self.qkv = Conv2d(
                in_channels=out_channels, out_channels=out_channels * 3, kernel=1,
                **init_attn
            )
            self.proj = Conv2d(
                in_channels=out_channels, out_channels=out_channels, kernel=1,
                **init_zero
            )

    def forward(self, x, emb):
        orig = x
        x = self.conv0(F.silu(self.norm0(x)))

        params = self.affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype)
        if self.adaptive_scale:
            scale, shift = params.chunk(chunks=2, dim=1)
            x = F.silu(torch.addcmul(shift, self.norm1(x), scale + 1))
        else:
            x = F.silu(self.norm1(x.add_(params)))

        x = self.conv1(F.dropout(x, p=self.dropout, training=self.training))
        x = x.add_(self.skip(orig) if self.skip is not None else orig)
        x = x * self.skip_scale

        if self.num_heads:
            q, k, v = self.qkv(self.norm2(x)).reshape(
                x.shape[0] * self.num_heads, x.shape[1] // self.num_heads, 3, -1
            ).unbind(2)
            w = AttentionOp.apply(q, k)
            a = torch.einsum('nqk,nck->ncq', w, v)
            x = self.proj(a.reshape(*x.shape)).add_(x)
            x = x * self.skip_scale
        return x


class PositionalEmbedding(nn.Module):
    """Timestep embedding used in DDPM++/ADM architectures."""

    def __init__(
        self,
        num_channels: int,
        max_positions: int = 10000,
        endpoint: bool = False,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(
            start=0, end=self.num_channels // 2,
            dtype=torch.float32, device=x.device
        )
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class FourierEmbedding(nn.Module):
    """Fourier embedding used in NCSN++ architecture."""

    def __init__(self, num_channels: int, scale: float = 0.02):
        super().__init__()
        self.register_buffer('freqs', torch.randn(num_channels // 2) * scale)

    def forward(self, x):
        x = x.ger((2 * np.pi * self.freqs).to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class SongUNet(nn.Module):
    """
    U-Net architecture based on DDPM++/NCSN++.

    This is the backbone network for the one-step generator in GFGW-FM.
    """

    def __init__(
        self,
        img_resolution: int,
        in_channels: int,
        out_channels: int,
        label_dim: int = 0,
        augment_dim: int = 0,
        model_channels: int = 128,
        channel_mult: Tuple[int, ...] = (1, 2, 2, 2),
        channel_mult_emb: int = 4,
        num_blocks: int = 4,
        attn_resolutions: Tuple[int, ...] = (16,),
        dropout: float = 0.10,
        label_dropout: float = 0,
        embedding_type: str = 'positional',
        channel_mult_noise: int = 1,
        encoder_type: str = 'standard',
        decoder_type: str = 'standard',
        resample_filter: List[int] = [1, 1],
    ):
        assert embedding_type in ['fourier', 'positional']
        assert encoder_type in ['standard', 'skip', 'residual']
        assert decoder_type in ['standard', 'skip']

        super().__init__()
        self.label_dropout = label_dropout
        self.img_resolution = img_resolution
        self.img_channels = in_channels
        self.label_dim = label_dim
        emb_channels = model_channels * channel_mult_emb
        noise_channels = model_channels * channel_mult_noise
        init = dict(init_mode='xavier_uniform')
        init_zero = dict(init_mode='xavier_uniform', init_weight=1e-5)
        init_attn = dict(init_mode='xavier_uniform', init_weight=np.sqrt(0.2))
        block_kwargs = dict(
            emb_channels=emb_channels, num_heads=1, dropout=dropout,
            skip_scale=np.sqrt(0.5), eps=1e-6, resample_filter=resample_filter,
            resample_proj=True, adaptive_scale=False,
            init=init, init_zero=init_zero, init_attn=init_attn,
        )

        # Mapping network
        if embedding_type == 'positional':
            self.map_noise = PositionalEmbedding(num_channels=noise_channels, endpoint=True)
        else:
            self.map_noise = FourierEmbedding(num_channels=noise_channels)

        self.map_label = Linear(
            in_features=label_dim, out_features=noise_channels, **init
        ) if label_dim else None
        self.map_augment = Linear(
            in_features=augment_dim, out_features=noise_channels, bias=False, **init
        ) if augment_dim else None
        self.map_layer0 = Linear(in_features=noise_channels, out_features=emb_channels, **init)
        self.map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)

        # Encoder
        self.enc = nn.ModuleDict()
        cout = in_channels
        caux = in_channels
        for level, mult in enumerate(channel_mult):
            res = img_resolution >> level
            if level == 0:
                cin = cout
                cout = model_channels
                self.enc[f'{res}x{res}_conv'] = Conv2d(
                    in_channels=cin, out_channels=cout, kernel=3, **init
                )
            else:
                self.enc[f'{res}x{res}_down'] = UNetBlock(
                    in_channels=cout, out_channels=cout, down=True, **block_kwargs
                )
                if encoder_type == 'skip':
                    self.enc[f'{res}x{res}_aux_down'] = Conv2d(
                        in_channels=caux, out_channels=caux, kernel=0,
                        down=True, resample_filter=resample_filter
                    )
                    self.enc[f'{res}x{res}_aux_skip'] = Conv2d(
                        in_channels=caux, out_channels=cout, kernel=1, **init
                    )
                if encoder_type == 'residual':
                    self.enc[f'{res}x{res}_aux_residual'] = Conv2d(
                        in_channels=caux, out_channels=cout, kernel=3,
                        down=True, resample_filter=resample_filter, fused_resample=True, **init
                    )
                    caux = cout
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                attn = (res in attn_resolutions)
                self.enc[f'{res}x{res}_block{idx}'] = UNetBlock(
                    in_channels=cin, out_channels=cout, attention=attn, **block_kwargs
                )
        skips = [block.out_channels for name, block in self.enc.items() if 'aux' not in name]

        # Decoder
        self.dec = nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = img_resolution >> level
            if level == len(channel_mult) - 1:
                self.dec[f'{res}x{res}_in0'] = UNetBlock(
                    in_channels=cout, out_channels=cout, attention=True, **block_kwargs
                )
                self.dec[f'{res}x{res}_in1'] = UNetBlock(
                    in_channels=cout, out_channels=cout, **block_kwargs
                )
            else:
                self.dec[f'{res}x{res}_up'] = UNetBlock(
                    in_channels=cout, out_channels=cout, up=True, **block_kwargs
                )
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = model_channels * mult
                attn = (idx == num_blocks and res in attn_resolutions)
                self.dec[f'{res}x{res}_block{idx}'] = UNetBlock(
                    in_channels=cin, out_channels=cout, attention=attn, **block_kwargs
                )
            if decoder_type == 'skip' or level == 0:
                if decoder_type == 'skip' and level < len(channel_mult) - 1:
                    self.dec[f'{res}x{res}_aux_up'] = Conv2d(
                        in_channels=out_channels, out_channels=out_channels,
                        kernel=0, up=True, resample_filter=resample_filter
                    )
                self.dec[f'{res}x{res}_aux_norm'] = GroupNorm(num_channels=cout, eps=1e-6)
                self.dec[f'{res}x{res}_aux_conv'] = Conv2d(
                    in_channels=cout, out_channels=out_channels, kernel=3, **init_zero
                )

    def forward(self, x, noise_labels, class_labels=None, augment_labels=None):
        # Mapping
        emb = self.map_noise(noise_labels)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)  # swap sin/cos
        if self.map_label is not None:
            tmp = class_labels
            if self.training and self.label_dropout:
                tmp = tmp * (torch.rand([x.shape[0], 1], device=x.device) >= self.label_dropout).to(tmp.dtype)
            emb = emb + self.map_label(tmp * np.sqrt(self.map_label.in_features))
        if self.map_augment is not None and augment_labels is not None:
            emb = emb + self.map_augment(augment_labels)
        emb = F.silu(self.map_layer0(emb))
        emb = F.silu(self.map_layer1(emb))

        # Encoder
        skips = []
        aux = x
        for name, block in self.enc.items():
            if 'aux_down' in name:
                aux = block(aux)
            elif 'aux_skip' in name:
                x = skips[-1] = x + block(aux)
            elif 'aux_residual' in name:
                x = skips[-1] = aux = (x + block(aux)) / np.sqrt(2)
            else:
                x = block(x, emb) if isinstance(block, UNetBlock) else block(x)
                skips.append(x)

        # Decoder
        aux = None
        tmp = None
        for name, block in self.dec.items():
            if 'aux_up' in name:
                aux = block(aux)
            elif 'aux_norm' in name:
                tmp = block(x)
            elif 'aux_conv' in name:
                tmp = block(F.silu(tmp))
                aux = tmp if aux is None else tmp + aux
            else:
                if x.shape[1] != block.in_channels:
                    x = torch.cat([x, skips.pop()], dim=1)
                x = block(x, emb)
        return aux


class OneStepGenerator(nn.Module):
    """
    One-step generator for GFGW-FM.

    This wraps the U-Net backbone with preconditioning for one-step generation.
    """

    def __init__(
        self,
        img_resolution: int,
        img_channels: int,
        label_dim: int = 0,
        sigma_data: float = 0.5,
        model_type: str = 'SongUNet',
        use_fp16: bool = False,
        **model_kwargs,
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.sigma_data = sigma_data
        self.use_fp16 = use_fp16

        # Build backbone network
        self.model = SongUNet(
            img_resolution=img_resolution,
            in_channels=img_channels,
            out_channels=img_channels,
            label_dim=label_dim,
            **model_kwargs
        )

    def forward(
        self,
        z: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        augment_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate images from latent codes in one step.

        Args:
            z: Latent codes (noise) of shape (B, C, H, W)
            class_labels: Optional class labels of shape (B, label_dim)
            augment_labels: Optional augmentation labels

        Returns:
            Generated images of shape (B, C, H, W)
        """
        z = z.to(torch.float32)

        # Use sigma_data as the "time" for conditioning
        sigma = torch.ones(z.shape[0], device=z.device) * self.sigma_data
        dtype = torch.float16 if (self.use_fp16 and z.device.type == 'cuda') else torch.float32

        # Preconditioning (from EDM paper)
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        # Reshape for broadcasting
        c_skip = c_skip.view(-1, 1, 1, 1)
        c_out = c_out.view(-1, 1, 1, 1)
        c_in = c_in.view(-1, 1, 1, 1)

        # Forward pass
        x_in = (c_in * z * sigma.view(-1, 1, 1, 1)).to(dtype)
        F_x = self.model(x_in, c_noise, class_labels, augment_labels)
        assert F_x.dtype == dtype
        x = c_skip * z + c_out * F_x.to(torch.float32)

        return x

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        device: torch.device,
        class_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Sample images from the generator.

        Args:
            batch_size: Number of images to generate
            device: Device for generation
            class_labels: Optional class labels

        Returns:
            Generated images
        """
        z = torch.randn(
            batch_size, self.img_channels, self.img_resolution, self.img_resolution,
            device=device
        )
        return self.forward(z, class_labels)


# ============================================================================
# Enhanced Generator with Boundary Conditions (from Boundary RF paper)
# ============================================================================

class BoundaryConditionedGenerator(nn.Module):
    """
    One-step generator with boundary condition enforcement.

    From "Improving Rectified Flow with Boundary Conditions":
    Enforces v(x, 1) = x to ensure trajectory straightness.

    Supports two parameterization types:
    - mask: v(x,t) = g(t)*(C-x) + f(t)*x + h(t)*m_theta(x,t)
    - subtraction: v(x,t) = x - f_theta(x,t)
    """

    def __init__(
        self,
        img_resolution: int,
        img_channels: int,
        label_dim: int = 0,
        sigma_data: float = 0.5,
        use_fp16: bool = False,
        boundary_type: str = "mask",  # "mask" or "subtraction"
        **model_kwargs,
    ):
        super().__init__()

        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.sigma_data = sigma_data
        self.use_fp16 = use_fp16
        self.boundary_type = boundary_type

        # Backbone network
        self.model = SongUNet(
            img_resolution=img_resolution,
            in_channels=img_channels,
            out_channels=img_channels,
            label_dim=label_dim,
            **model_kwargs
        )

    def _mask_parameterization(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        raw_output: torch.Tensor,
    ) -> torch.Tensor:
        """
        Mask-based boundary parameterization.

        v(x,t) = g(t)*(C-x) + f(t)*x + h(t)*m_theta(x,t)

        where:
        - g(t) = 0 at t=1 (no movement towards constant C)
        - f(t) = 1 at t=1 (identity at boundary)
        - h(t) = 0 at t=1 (no network contribution at boundary)
        """
        t = t.view(-1, 1, 1, 1)

        # Boundary functions (satisfy v(x,1) = x)
        g_t = 1.0 - t  # g(1) = 0
        f_t = t  # f(1) = 1
        h_t = (1.0 - t) * t  # h(0) = h(1) = 0, peak at t=0.5

        # Constant C (can be learned or fixed)
        C = torch.zeros_like(z)  # Target is "clean" image

        # Parameterized output
        output = g_t * (C - z) + f_t * z + h_t * raw_output

        return output

    def _subtraction_parameterization(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        raw_output: torch.Tensor,
    ) -> torch.Tensor:
        """
        Subtraction-based boundary parameterization.

        v(x,t) = x + (1-t) * f_theta(x,t)

        At t=1: v(x,1) = x (identity)
        """
        t = t.view(-1, 1, 1, 1)

        # Scale network output by (1-t)
        output = z + (1.0 - t) * raw_output

        return output

    def forward(
        self,
        z: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        augment_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate images with boundary condition enforcement.

        Args:
            z: Input noise (B, C, H, W)
            class_labels: Optional class labels
            t: Time values (B,), defaults to sigma_data
            augment_labels: Optional augmentation labels

        Returns:
            Generated images with boundary conditions satisfied
        """
        z = z.to(torch.float32)
        batch_size = z.shape[0]
        device = z.device

        # Default time to sigma_data (one-step generation)
        if t is None:
            t = torch.ones(batch_size, device=device) * self.sigma_data

        # Preconditioning
        sigma = t * self.sigma_data
        dtype = torch.float16 if (self.use_fp16 and device.type == 'cuda') else torch.float32

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        c_skip = c_skip.view(-1, 1, 1, 1)
        c_out = c_out.view(-1, 1, 1, 1)
        c_in = c_in.view(-1, 1, 1, 1)

        # Network forward pass
        x_in = (c_in * z * sigma.view(-1, 1, 1, 1)).to(dtype)
        raw_output = self.model(x_in, c_noise, class_labels, augment_labels)
        raw_output = raw_output.to(torch.float32)

        # Apply boundary parameterization
        if self.boundary_type == "mask":
            x = self._mask_parameterization(z, t, c_skip * z + c_out * raw_output)
        elif self.boundary_type == "subtraction":
            x = self._subtraction_parameterization(z, t, c_out * raw_output)
        else:
            # Fallback to standard EDM preconditioning
            x = c_skip * z + c_out * raw_output

        return x

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        device: torch.device,
        class_labels: Optional[torch.Tensor] = None,
        num_steps: int = 1,
    ) -> torch.Tensor:
        """
        Sample images with optional multi-step generation.

        Args:
            batch_size: Number of images
            device: Target device
            class_labels: Optional labels
            num_steps: Number of generation steps (1 for one-step)

        Returns:
            Generated images
        """
        z = torch.randn(
            batch_size, self.img_channels, self.img_resolution, self.img_resolution,
            device=device
        )

        if num_steps == 1:
            return self.forward(z, class_labels)

        # Multi-step Euler integration
        x = z
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.ones(batch_size, device=device) * (1.0 - i * dt)
            v = self.forward(x, class_labels, t) - x
            x = x + v * dt

        return x


class FlowGuidedGenerator(nn.Module):
    """
    Generator with flow-guided distillation support.

    From SlimFlow: Uses 2-step teacher to guide 1-step student.
    """

    def __init__(
        self,
        img_resolution: int,
        img_channels: int,
        label_dim: int = 0,
        sigma_data: float = 0.5,
        use_fp16: bool = False,
        **model_kwargs,
    ):
        super().__init__()

        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.sigma_data = sigma_data
        self.use_fp16 = use_fp16

        # Student (1-step) network
        self.student = SongUNet(
            img_resolution=img_resolution,
            in_channels=img_channels,
            out_channels=img_channels,
            label_dim=label_dim,
            **model_kwargs
        )

    def forward(
        self,
        z: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        augment_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """One-step generation."""
        z = z.to(torch.float32)

        sigma = torch.ones(z.shape[0], device=z.device) * self.sigma_data
        dtype = torch.float16 if (self.use_fp16 and z.device.type == 'cuda') else torch.float32

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        c_skip = c_skip.view(-1, 1, 1, 1)
        c_out = c_out.view(-1, 1, 1, 1)
        c_in = c_in.view(-1, 1, 1, 1)

        x_in = (c_in * z * sigma.view(-1, 1, 1, 1)).to(dtype)
        F_x = self.student(x_in, c_noise, class_labels, augment_labels)
        x = c_skip * z + c_out * F_x.to(torch.float32)

        return x

    def forward_with_teacher(
        self,
        z: torch.Tensor,
        teacher: nn.Module,
        class_labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both student output and teacher 2-step output.

        For flow-guided distillation training.
        """
        # Student 1-step output
        student_output = self.forward(z, class_labels)

        # Teacher 2-step Euler
        with torch.no_grad():
            t_mid = 0.5
            t1 = torch.ones(z.shape[0], device=z.device)
            t2 = torch.ones(z.shape[0], device=z.device) * t_mid

            # Step 1: z -> x_mid
            x_mid = teacher(z, class_labels)
            x_mid = z + (x_mid - z) * t_mid

            # Step 2: x_mid -> x_final
            x_final = teacher.forward(x_mid, class_labels)

        return student_output, x_final

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        device: torch.device,
        class_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample images."""
        z = torch.randn(
            batch_size, self.img_channels, self.img_resolution, self.img_resolution,
            device=device
        )
        return self.forward(z, class_labels)


# ============================================================================
# Factory Function
# ============================================================================

def create_generator(config) -> nn.Module:
    """
    Factory function to create generator from config.

    Args:
        config: Configuration object

    Returns:
        Generator module
    """
    model_kwargs = {
        'model_channels': config.model.model_channels,
        'channel_mult': config.model.channel_mult,
        'num_blocks': config.model.num_blocks,
        'attn_resolutions': config.model.attn_resolutions,
        'dropout': config.model.dropout,
    }

    if hasattr(config.model, 'use_boundary_condition') and config.model.use_boundary_condition:
        return BoundaryConditionedGenerator(
            img_resolution=config.model.img_resolution,
            img_channels=config.model.img_channels,
            label_dim=config.model.label_dim,
            sigma_data=config.model.sigma_data,
            use_fp16=config.model.use_fp16,
            boundary_type=config.model.boundary_type,
            **model_kwargs,
        )
    else:
        return OneStepGenerator(
            img_resolution=config.model.img_resolution,
            img_channels=config.model.img_channels,
            label_dim=config.model.label_dim,
            sigma_data=config.model.sigma_data,
            use_fp16=config.model.use_fp16,
            **model_kwargs,
        )
