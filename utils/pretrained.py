"""Pretrained model loading utilities for GFGW-FM.

Implements loading strategies from top-tier papers:
- ECM: Load EDM models for fine-tuning
- TCM: Load EDM/EDM2 models with two-stage training
- SlimFlow: Load rectified flow models

Core principle: Initialize from pretrained diffusion/flow models
while keeping the GFGW-FM core innovation (FGW OT + Global Memory Bank) intact.
"""

import os
import pickle
import copy
import urllib.request
from typing import Optional, Dict, Any, Union
import torch
import torch.nn as nn


def open_url(url: str, cache_dir: str = None, verbose: bool = True) -> Any:
    """
    Download file from URL and cache locally.
    
    Based on EDM/TCM dnnlib.util.open_url implementation.
    """
    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser('~'), '.cache', 'gfgw-fm')
    
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create cache filename from URL
    url_hash = hash(url) % (10 ** 8)
    cache_file = os.path.join(cache_dir, f'cached_{url_hash}.pkl')
    
    if os.path.exists(cache_file):
        if verbose:
            print(f'Loading from cache: {cache_file}')
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    if verbose:
        print(f'Downloading: {url}')
    
    # Download
    try:
        with urllib.request.urlopen(url) as response:
            data = pickle.load(response)
    except Exception as e:
        raise RuntimeError(f'Failed to download {url}: {e}')
    
    # Cache
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)
    
    return data


def copy_params_and_buffers(
    src_module: nn.Module,
    dst_module: nn.Module,
    require_all: bool = False,
) -> None:
    """
    Copy parameters and buffers from source to destination module.
    
    Based on EDM misc.copy_params_and_buffers implementation.
    This allows flexible loading even when architectures slightly differ.
    """
    assert isinstance(src_module, nn.Module)
    assert isinstance(dst_module, nn.Module)
    
    src_tensors = dict(src_module.named_parameters())
    src_tensors.update(dict(src_module.named_buffers()))
    
    for name, tensor in dst_module.named_parameters():
        if name in src_tensors:
            if tensor.shape == src_tensors[name].shape:
                tensor.copy_(src_tensors[name].detach()).requires_grad_(tensor.requires_grad)
            else:
                print(f'Warning: Shape mismatch for {name}: '
                      f'{tensor.shape} vs {src_tensors[name].shape}, skipping')
        elif require_all:
            raise KeyError(f'Missing parameter: {name}')
        else:
            print(f'Warning: Missing parameter {name} in source, keeping random init')
    
    for name, buffer in dst_module.named_buffers():
        if name in src_tensors:
            if buffer.shape == src_tensors[name].shape:
                buffer.copy_(src_tensors[name].detach())
            else:
                print(f'Warning: Shape mismatch for buffer {name}: '
                      f'{buffer.shape} vs {src_tensors[name].shape}, skipping')


class PretrainedModelLoader:
    """
    Unified pretrained model loader for GFGW-FM.
    
    Supports loading from:
    - EDM format (.pkl with 'ema' key)
    - EDM2 format (.pkl with different structure)
    - ECT format (similar to EDM)
    - TCM format (similar to EDM/EDM2)
    - SlimFlow format (.pth PyTorch checkpoints)
    - Standard PyTorch format (.pt/.pth)
    
    Key insight: All top papers use EDM/EDM2 pretrained models as initialization.
    This dramatically reduces training time while maintaining quality.
    """
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        verbose: bool = True,
    ):
        self.cache_dir = cache_dir or os.path.join(
            os.path.expanduser('~'), '.cache', 'gfgw-fm', 'pretrained'
        )
        self.verbose = verbose
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def load(
        self,
        path_or_url: str,
        model_format: str = 'auto',
    ) -> Dict[str, Any]:
        """
        Load pretrained checkpoint.
        
        Args:
            path_or_url: Local path or URL to checkpoint
            model_format: 'edm', 'edm2', 'pytorch', or 'auto'
            
        Returns:
            Dictionary with 'model_state_dict' and optional 'ema_state_dict'
        """
        # Determine if URL or local path
        is_url = path_or_url.startswith('http://') or path_or_url.startswith('https://')
        
        # Load checkpoint
        if is_url:
            data = self._load_from_url(path_or_url)
        else:
            data = self._load_from_path(path_or_url)
        
        # Detect format if auto
        if model_format == 'auto':
            model_format = self._detect_format(data, path_or_url)
        
        if self.verbose:
            print(f'Detected format: {model_format}')
        
        # Parse based on format
        if model_format in ['edm', 'ect', 'tcm']:
            return self._parse_edm_format(data)
        elif model_format == 'edm2':
            return self._parse_edm2_format(data)
        elif model_format in ['pytorch', 'slimflow']:
            return self._parse_pytorch_format(data)
        else:
            raise ValueError(f'Unknown model format: {model_format}')
    
    def _load_from_url(self, url: str) -> Any:
        """Download and load from URL."""
        # Create cache filename
        url_basename = os.path.basename(url).split('?')[0]
        cache_path = os.path.join(self.cache_dir, url_basename)
        
        if os.path.exists(cache_path):
            if self.verbose:
                print(f'Loading from cache: {cache_path}')
        else:
            if self.verbose:
                print(f'Downloading: {url}')
            urllib.request.urlretrieve(url, cache_path)
        
        return self._load_from_path(cache_path)
    
    def _load_from_path(self, path: str) -> Any:
        """Load from local path."""
        if path.endswith('.pkl'):
            with open(path, 'rb') as f:
                return pickle.load(f)
        elif path.endswith('.pt') or path.endswith('.pth'):
            return torch.load(path, map_location='cpu', weights_only=False)
        else:
            # Try pickle first, then torch
            try:
                with open(path, 'rb') as f:
                    return pickle.load(f)
            except:
                return torch.load(path, map_location='cpu', weights_only=False)
    
    def _detect_format(self, data: Any, path: str) -> str:
        """Detect checkpoint format."""
        if isinstance(data, dict):
            if 'ema' in data and hasattr(data['ema'], 'state_dict'):
                # EDM format: data['ema'] is a Module
                return 'edm'
            elif 'model' in data and isinstance(data['model'], dict):
                # SlimFlow/PyTorch format
                return 'pytorch'
            elif 'state_dict' in data:
                return 'pytorch'
            elif 'ema' in data and isinstance(data['ema'], dict):
                # EDM2 format: data['ema'] is a state_dict
                return 'edm2'
        
        if hasattr(data, 'state_dict'):
            # Direct module object
            return 'edm'
        
        # Default to edm for .pkl files
        if path.endswith('.pkl'):
            return 'edm'
        
        return 'pytorch'
    
    def _parse_edm_format(self, data: Dict) -> Dict[str, Any]:
        """
        Parse EDM/ECT/TCM format checkpoint.
        
        EDM format: data['ema'] is an nn.Module with pretrained weights
        """
        result = {}
        
        if 'ema' in data:
            ema_module = data['ema']
            if hasattr(ema_module, 'state_dict'):
                result['ema_state_dict'] = ema_module.state_dict()
                result['ema_module'] = ema_module  # Keep module for copy_params_and_buffers
            elif isinstance(ema_module, dict):
                result['ema_state_dict'] = ema_module
        
        # Also check for 'net' key (training state dumps)
        if 'net' in data:
            net_module = data['net']
            if hasattr(net_module, 'state_dict'):
                result['model_state_dict'] = net_module.state_dict()
            elif isinstance(net_module, dict):
                result['model_state_dict'] = net_module
        
        # Use EMA as model if no 'net' key
        if 'model_state_dict' not in result and 'ema_state_dict' in result:
            result['model_state_dict'] = result['ema_state_dict']
        
        return result
    
    def _parse_edm2_format(self, data: Dict) -> Dict[str, Any]:
        """Parse EDM2 format checkpoint."""
        result = {}
        
        # EDM2 may have different structure
        if 'ema' in data:
            if isinstance(data['ema'], dict):
                result['ema_state_dict'] = data['ema']
            elif hasattr(data['ema'], 'state_dict'):
                result['ema_state_dict'] = data['ema'].state_dict()
                result['ema_module'] = data['ema']
        
        if 'model_state_dict' not in result and 'ema_state_dict' in result:
            result['model_state_dict'] = result['ema_state_dict']
        
        return result
    
    def _parse_pytorch_format(self, data: Dict) -> Dict[str, Any]:
        """Parse standard PyTorch format checkpoint."""
        result = {}
        
        if 'model' in data:
            if isinstance(data['model'], dict):
                result['model_state_dict'] = data['model']
            elif hasattr(data['model'], 'state_dict'):
                result['model_state_dict'] = data['model'].state_dict()
        
        if 'state_dict' in data:
            result['model_state_dict'] = data['state_dict']
        
        if 'ema' in data:
            if isinstance(data['ema'], dict):
                result['ema_state_dict'] = data['ema']
            elif hasattr(data['ema'], 'state_dict'):
                result['ema_state_dict'] = data['ema'].state_dict()
        
        return result
    
    def load_into_model(
        self,
        model: nn.Module,
        path_or_url: str,
        model_format: str = 'auto',
        strict: bool = False,
        use_ema: bool = True,
    ) -> nn.Module:
        """
        Load pretrained weights into model.
        
        Args:
            model: Target model to load weights into
            path_or_url: Path or URL to checkpoint
            model_format: Checkpoint format
            strict: Whether to require exact match
            use_ema: Whether to use EMA weights (recommended)
            
        Returns:
            Model with loaded weights
        """
        checkpoint = self.load(path_or_url, model_format)
        
        # Prefer EMA weights (they're typically better)
        if use_ema and 'ema_module' in checkpoint:
            # Use copy_params_and_buffers for flexible loading (from EDM)
            if self.verbose:
                print('Loading EMA weights using copy_params_and_buffers...')
            copy_params_and_buffers(
                src_module=checkpoint['ema_module'],
                dst_module=model,
                require_all=strict,
            )
        elif use_ema and 'ema_state_dict' in checkpoint:
            if self.verbose:
                print('Loading EMA state dict...')
            self._load_state_dict_flexible(
                model, checkpoint['ema_state_dict'], strict
            )
        elif 'model_state_dict' in checkpoint:
            if self.verbose:
                print('Loading model state dict...')
            self._load_state_dict_flexible(
                model, checkpoint['model_state_dict'], strict
            )
        else:
            raise ValueError('No loadable weights found in checkpoint')
        
        return model
    
    def _load_state_dict_flexible(
        self,
        model: nn.Module,
        state_dict: Dict[str, torch.Tensor],
        strict: bool = False,
    ):
        """Load state dict with flexible key matching."""
        model_state = model.state_dict()
        
        # Try direct loading first
        try:
            model.load_state_dict(state_dict, strict=strict)
            return
        except RuntimeError as e:
            if strict:
                raise e
            if self.verbose:
                print(f'Direct loading failed, trying flexible loading: {e}')
        
        # Flexible loading: match by key suffix
        loaded_keys = []
        missing_keys = []
        
        for key in model_state.keys():
            # Try exact match
            if key in state_dict:
                if model_state[key].shape == state_dict[key].shape:
                    model_state[key].copy_(state_dict[key])
                    loaded_keys.append(key)
                else:
                    if self.verbose:
                        print(f'Shape mismatch: {key}')
                continue
            
            # Try matching by suffix (handle 'model.' prefix differences)
            matched = False
            for src_key in state_dict.keys():
                if src_key.endswith(key) or key.endswith(src_key):
                    if model_state[key].shape == state_dict[src_key].shape:
                        model_state[key].copy_(state_dict[src_key])
                        loaded_keys.append(key)
                        matched = True
                        break
            
            if not matched:
                missing_keys.append(key)
        
        model.load_state_dict(model_state, strict=False)
        
        if self.verbose:
            print(f'Loaded {len(loaded_keys)}/{len(model_state)} parameters')
            if missing_keys:
                print(f'Missing keys (kept random init): {missing_keys[:5]}...')


def load_pretrained_edm_model(
    model: nn.Module,
    url_or_path: str,
    device: torch.device = torch.device('cuda'),
    verbose: bool = True,
) -> nn.Module:
    """
    Convenience function to load pretrained EDM model.
    
    This is the main function to use for GFGW-FM training with pretrained initialization.
    
    Example:
        from utils.pretrained import load_pretrained_edm_model
        
        generator = OneStepGenerator(...)
        generator = load_pretrained_edm_model(
            generator,
            url_or_path='https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vp.pkl',
        )
    """
    loader = PretrainedModelLoader(verbose=verbose)
    model = loader.load_into_model(model, url_or_path, strict=False, use_ema=True)
    return model.to(device)


def create_ema_from_pretrained(
    model: nn.Module,
    url_or_path: str,
    verbose: bool = True,
) -> nn.Module:
    """
    Create EMA model initialized from pretrained weights.
    
    Returns a deep copy of the model with pretrained weights loaded.
    """
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    loader = PretrainedModelLoader(verbose=verbose)
    ema = loader.load_into_model(ema, url_or_path, strict=False, use_ema=True)
    return ema

