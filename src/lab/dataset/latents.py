from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from lab.utils.io import get_root, read_json

PROBS_FILE = 'probs.npy'
AUGS_FILE = 'augs.npy'
META_FILE = 'meta.json'


def get_latents_root(name: str, split: str) -> Path:
    """
    Default directory of the precomputed latents, see 'scripts/precompute_latents.py'.
    """
    return get_root() / 'data' / name / 'latents' / split


class LatentDataset(Dataset):
    """
    Binary latents precomputed by the frozen binary autoencoder, one stored set of
    augmentations (crop and flip) per image.

    An element holds the Bernoulli probabilities of the latents ('probs' of shape (C, h, w),
    float16) of one randomly chosen augmentation of the image. The binary code is sampled from
    them in the trainer, exactly as the online encoder does, so the only difference from the
    online training is the finite number of augmentations per image.
    """

    def __init__(
        self,
        name: str = 'celebahq',
        split: str = 'train',
        root: Path | str | None = None,
        limit: int | None = None,
    ) -> None:
        """
        Parameters
        ----------
        name : str, optional
            Dataset name, a directory inside 'data'.
        split : str, optional
            Dataset split.
        root : Path or str or None, optional
            Directory with the latents relative to the project root.
            If None, 'data/{name}/latents/{split}'.
        limit : int or None, optional
            If not None, keep only the first 'limit' images.
        """
        root = get_latents_root(name, split) if root is None else get_root() / root
        meta_path = root / META_FILE
        if not meta_path.exists():
            raise FileNotFoundError(
                f'No latents in {root}, create them with: '
                f'uv run scripts/precompute_latents.py --split {split}'
            )
        meta: dict[str, Any] = read_json(meta_path)  # ty:ignore[invalid-assignment]

        self.meta = meta
        self.probs_path = root / PROBS_FILE
        self.num_augs: int = meta['num_augs']
        self.num_images: int = (
            meta['num_images'] if limit is None else min(limit, meta['num_images'])
        )
        # opened lazily: a memory map is not shared with the dataloader workers by pickling
        self._probs: np.ndarray | None = None

    def __len__(self) -> int:
        return self.num_images

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self._probs is None:
            self._probs = np.load(self.probs_path, mmap_mode='r')
        # the dataloader seeds torch differently in each worker and each epoch
        aug = int(torch.randint(self.num_augs, ()))
        probs = torch.from_numpy(np.array(self._probs[index, aug]))
        return {'probs': probs}
