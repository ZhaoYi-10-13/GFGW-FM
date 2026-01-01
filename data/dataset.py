"""Dataset classes for GFGW-FM training with index support."""

import os
import io
import numpy as np
import zipfile
import json
import torch
from torch.utils.data import Dataset
from typing import Optional, Tuple, List
import PIL.Image

try:
    import pyspng
except ImportError:
    pyspng = None


class IndexedImageFolderDataset(Dataset):
    """
    Dataset for loading images with index support for OT matching.

    This dataset returns (image, label, index) tuples to support
    fetching matched images during training.
    """

    def __init__(
        self,
        path: str,
        resolution: Optional[int] = None,
        use_labels: bool = False,
        max_size: Optional[int] = None,
        xflip: bool = False,
        random_seed: int = 0,
        cache: bool = False,
    ):
        self._path = path
        self._resolution = resolution
        self._use_labels = use_labels
        self._cache = cache
        self._cached_images = {}
        self._xflip = xflip

        # Determine if path is directory or ZIP
        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {
                os.path.relpath(os.path.join(root, fname), start=self._path)
                for root, _dirs, files in os.walk(self._path)
                for fname in files
            }
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._zipfile = None
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or ZIP file')

        # Filter to image files only
        PIL.Image.init()
        self._image_fnames = sorted(
            fname for fname in self._all_fnames
            if self._file_ext(fname) in PIL.Image.EXTENSION
        )

        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        # Get dataset info
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)

        if resolution is not None:
            if raw_shape[2] != resolution or raw_shape[3] != resolution:
                raise IOError(f'Image files do not match resolution {resolution}')

        self._raw_shape = raw_shape

        # Apply max_size
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if max_size is not None and self._raw_idx.size > max_size:
            np.random.RandomState(random_seed % (1 << 31)).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip
        self._xflip_flags = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip_flags = np.concatenate([
                self._xflip_flags,
                np.ones_like(self._xflip_flags)
            ])

        # Load labels
        self._raw_labels = None
        self._label_shape = None

    @staticmethod
    def _file_ext(fname: str) -> str:
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname: str):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def _load_raw_image(self, raw_idx: int) -> np.ndarray:
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            if pyspng is not None and self._file_ext(fname) == '.png':
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))

        if image.ndim == 2:
            image = image[:, :, np.newaxis]

        image = image.transpose(2, 0, 1)
        return image

    def _get_raw_labels(self) -> np.ndarray:
        if self._raw_labels is None:
            if self._use_labels:
                self._raw_labels = self._load_raw_labels()
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
        return self._raw_labels

    def _load_raw_labels(self) -> Optional[np.ndarray]:
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None

        with self._open_file(fname) as f:
            labels = json.load(f)['labels']

        if labels is None:
            return None

        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels

    def close(self):
        try:
            if hasattr(self, '_zipfile') and self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __len__(self) -> int:
        return self._raw_idx.size

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, int]:
        """Returns (image, label, index) tuple."""
        raw_idx = self._raw_idx[idx]

        # Load image
        image = self._cached_images.get(raw_idx, None)
        if image is None:
            image = self._load_raw_image(raw_idx)
            if self._cache:
                self._cached_images[raw_idx] = image

        # Apply xflip
        if self._xflip_flags[idx]:
            image = image[:, :, ::-1].copy()

        # Get label
        label = self._get_raw_labels()[raw_idx]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_dim, dtype=np.float32)
            onehot[label] = 1
            label = onehot

        # Return with index
        return image.copy(), label.copy(), idx

    def get_image_by_index(self, idx: int) -> np.ndarray:
        """Get image by dataset index (for OT matching)."""
        raw_idx = self._raw_idx[idx % len(self._raw_idx)]
        image = self._load_raw_image(raw_idx)
        if idx >= len(self._raw_idx) // (2 if self._xflip else 1):
            image = image[:, :, ::-1].copy()
        return image

    def get_images_by_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Get multiple images by indices (batch operation)."""
        images = []
        for idx in indices.cpu().numpy():
            img = self.get_image_by_index(int(idx))
            images.append(torch.from_numpy(img))
        return torch.stack(images, dim=0)

    @property
    def image_shape(self) -> List[int]:
        return list(self._raw_shape[1:])

    @property
    def num_channels(self) -> int:
        return self.image_shape[0]

    @property
    def resolution(self) -> int:
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self) -> List[int]:
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = list(raw_labels.shape[1:])
        return self._label_shape

    @property
    def label_dim(self) -> int:
        if len(self.label_shape) == 0:
            return 0
        return self.label_shape[0]

    @property
    def has_labels(self) -> bool:
        return any(x != 0 for x in self.label_shape)


class ImageFolderDataset(IndexedImageFolderDataset):
    """Backward compatible alias."""
    pass


class CIFAR10Dataset(Dataset):
    """CIFAR-10 dataset wrapper with index support."""

    def __init__(
        self,
        root: str,
        train: bool = True,
        download: bool = True,
    ):
        from torchvision.datasets import CIFAR10

        self.dataset = CIFAR10(root=root, train=train, download=download)
        self._images = None  # Lazy loaded

    def _load_all_images(self):
        """Load all images into memory for fast indexed access."""
        if self._images is None:
            self._images = []
            for i in range(len(self.dataset)):
                image, _ = self.dataset[i]
                image = np.array(image).transpose(2, 0, 1)
                self._images.append(image)
            self._images = np.stack(self._images, axis=0)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, int]:
        image, label = self.dataset[idx]
        image = np.array(image)

        if image.ndim == 2:
            image = image[:, :, np.newaxis]

        image = image.transpose(2, 0, 1)

        label_onehot = np.zeros(10, dtype=np.float32)
        label_onehot[label] = 1

        return image, label_onehot, idx

    def get_images_by_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Get multiple images by indices."""
        self._load_all_images()
        indices_np = indices.cpu().numpy()
        images = self._images[indices_np]
        return torch.from_numpy(images)

    @property
    def resolution(self) -> int:
        return 32

    @property
    def num_channels(self) -> int:
        return 3

    @property
    def label_dim(self) -> int:
        return 10


class LSUNDataset(Dataset):
    """
    LSUN dataset wrapper with index support.

    Supports LSUN categories: bedroom, church_outdoor, cat, etc.
    LSUN is stored in LMDB format.
    """

    def __init__(
        self,
        root: str,
        category: str = 'bedroom',
        split: str = 'train',
        resolution: int = 256,
        max_size: Optional[int] = None,
    ):
        try:
            import lmdb
        except ImportError:
            raise ImportError("LSUN requires lmdb. Install with: pip install lmdb")

        self.resolution = resolution
        self.category = category

        # LSUN path format: root/category_split_lmdb
        db_path = os.path.join(root, f'{category}_{split}_lmdb')
        if not os.path.exists(db_path):
            # Try alternative path format
            db_path = os.path.join(root, category, split)

        self.env = lmdb.open(
            db_path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False
        )

        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries']

        if max_size is not None:
            self.length = min(self.length, max_size)

        # Cache keys
        self._keys = None
        self._images = None

    def _load_keys(self):
        if self._keys is None:
            with self.env.begin(write=False) as txn:
                self._keys = list(txn.cursor().iternext(values=False))[:self.length]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, int]:
        self._load_keys()

        with self.env.begin(write=False) as txn:
            imgbuf = txn.get(self._keys[idx])

        # Decode image
        buf = np.frombuffer(imgbuf, dtype=np.uint8)
        img = PIL.Image.open(io.BytesIO(buf)).convert('RGB')

        # Resize if needed
        if img.size[0] != self.resolution or img.size[1] != self.resolution:
            img = img.resize((self.resolution, self.resolution), PIL.Image.LANCZOS)

        image = np.array(img).transpose(2, 0, 1)

        # LSUN is unconditional, return empty label
        label = np.zeros(0, dtype=np.float32)

        return image, label, idx

    def get_images_by_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Get multiple images by indices."""
        images = []
        for idx in indices.cpu().numpy():
            img, _, _ = self[int(idx)]
            images.append(torch.from_numpy(img))
        return torch.stack(images, dim=0)

    @property
    def label_dim(self) -> int:
        return 0


class ImageNetDataset(Dataset):
    """
    ImageNet dataset with index support for various resolutions.

    Supports ImageNet-64 and ImageNet-256.
    """

    def __init__(
        self,
        root: str,
        split: str = 'train',
        resolution: int = 64,
        use_labels: bool = True,
    ):
        from torchvision.datasets import ImageFolder
        from torchvision import transforms

        self.resolution = resolution
        self.use_labels = use_labels

        # Build transform
        self.transform = transforms.Compose([
            transforms.Resize(resolution),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
        ])

        split_path = os.path.join(root, split)
        if os.path.exists(split_path):
            self.dataset = ImageFolder(split_path)
        else:
            self.dataset = ImageFolder(root)

        self._images = None

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, int]:
        image, label = self.dataset[idx]

        # Apply transform and convert to numpy
        image = self.transform(image)
        image = (image.numpy() * 255).astype(np.uint8)

        if self.use_labels:
            label_onehot = np.zeros(1000, dtype=np.float32)
            label_onehot[label] = 1
        else:
            label_onehot = np.zeros(0, dtype=np.float32)

        return image, label_onehot, idx

    def get_images_by_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Get multiple images by indices."""
        images = []
        for idx in indices.cpu().numpy():
            img, _, _ = self[int(idx)]
            images.append(torch.from_numpy(img))
        return torch.stack(images, dim=0)

    @property
    def label_dim(self) -> int:
        return 1000 if self.use_labels else 0


class InfiniteSampler(torch.utils.data.Sampler):
    """Infinite sampler for training without epochs."""

    def __init__(
        self,
        dataset: Dataset,
        rank: int = 0,
        num_replicas: int = 1,
        shuffle: bool = True,
        seed: int = 0,
        window_size: float = 0.5,
    ):
        assert len(dataset) > 0
        assert num_replicas > 0
        assert 0 <= rank < num_replicas
        assert 0 <= window_size <= 1

        self.dataset = dataset
        self.rank = rank
        self.num_replicas = num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.window_size = window_size

    def __iter__(self):
        order = np.arange(len(self.dataset))
        rnd = None
        window = 0
        idx = 0

        while True:
            i = idx % order.size
            if idx % self.num_replicas == self.rank:
                yield order[i]
            if window >= 1:
                j = (i - rnd.randint(window)) % order.size
                order[i], order[j] = order[j], order[i]
            idx += 1
            if self.shuffle and idx % order.size == 0:
                rnd = np.random.RandomState(self.seed)
                self.seed += 1
                window = int(np.rint(order.size * self.window_size))
