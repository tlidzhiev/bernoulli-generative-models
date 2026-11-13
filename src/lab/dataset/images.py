import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar, Literal

import safetensors.torch
from datasets import load_dataset
from torchvision.transforms import v2
from tqdm.auto import tqdm

from lab.utils.io import get_root, read_json, write_json

from .base import BaseDataset

logger = logging.getLogger(__name__)


class HFImageDataset(BaseDataset):
    """
    Unconditional image dataset from the Hugging Face Hub.
    """

    name: str
    hf_path: str
    hf_splits: ClassVar[dict[str, str]]

    def __init__(
        self,
        root: Path | str | None = None,
        split: Literal['train', 'test'] = 'train',
        limit: int | None = None,
        shuffle_index: bool = False,
        instance_transforms: dict[str, Callable] | None = None,
        force_reindex: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        root : Path or str or None, optional
            Dataset directory relative to the project root. If None, 'data/{name}/{split}'.
        split : {'train', 'test'}, optional
            Dataset split.
        limit : int or None, optional
            Limit the number of samples.
        shuffle_index : bool, optional
            Whether to shuffle the index.
        instance_transforms : dict[str, Callable] or None, optional
            Instance transforms to apply.
        force_reindex : bool, optional
            Recreate the index file even if it exists.
        """
        if root is None:
            root = get_root() / 'data' / self.name / split
        else:
            root = get_root() / root

        index_path = root / 'index.json'
        if index_path.exists() and not force_reindex:
            index: list[dict[str, Any]] = read_json(str(index_path))  # ty:ignore[invalid-assignment]
        else:
            index: list[dict[str, Any]] = self._create_index(split, root)

        super().__init__(
            index=index,
            limit=limit,
            shuffle_index=shuffle_index,
            instance_transforms=instance_transforms,
        )

    def _create_index(
        self,
        split: Literal['train', 'test'],
        data_path: Path,
    ) -> list[dict[str, Any]]:
        """
        Download the dataset, save each image as a safetensors file and write index.json.

        Returns
        -------
        list[dict[str, Any]]
            Index with the 'path' of each element.
        """

        index: list[dict[str, Any]] = []
        data_path.mkdir(exist_ok=True, parents=True)

        transform = v2.Compose([v2.ToImage(), v2.Resize(268, antialias=True), v2.CenterCrop(268)])
        hf_dataset = load_dataset(self.hf_path, split=self.hf_splits[split])

        logger.info(f'Processing {self.name} data for {split} part...')
        for i, item in tqdm(enumerate(hf_dataset), total=len(hf_dataset)):
            img = transform(item['image'].convert('RGB')).contiguous()

            save_dict = {'img': img}
            save_path = data_path / f'{i:06}.safetensors'
            safetensors.torch.save_file(save_dict, save_path)

            index.append({'path': str(save_path)})

        write_json(index, str(data_path / 'index.json'))

        logger.info(f'Successfully processed {len(index)} {split} {self.name} images.')
        return index


class CelebAHQDataset(HFImageDataset):
    """
    CelebA-HQ at 256x256, see https://github.com/tkarras/progressive_growing_of_gans.
    """

    name = 'celebahq'
    hf_path = 'korexyz/celeba-hq-256x256'
    hf_splits: ClassVar[dict[str, str]] = {'train': 'train', 'test': 'validation'}


class LSUNChurchesDataset(HFImageDataset):
    """
    LSUN Churches (church_outdoor), see https://www.yf.io/p/lsun.
    """

    name = 'churches'
    hf_path = 'tglcourse/lsun_church_train'
    hf_splits: ClassVar[dict[str, str]] = {'train': 'train', 'test': 'test'}
