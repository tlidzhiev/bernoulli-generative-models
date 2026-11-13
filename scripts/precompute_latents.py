"""
Precompute the binary latents of the train images with the frozen binary autoencoder, so
that the training does not run its encoder at each step.

Augmentations of the training (random crop 256 of the image stored at 268, horizontal flip)
cannot be applied to the latents, and the encoder is not equivariant to them. Instead, for
each image `--num-augs` distinct augmentations (crop position, flip) are drawn and encoded;
the training picks one of them at random. The Bernoulli probabilities of the latents are
stored (not the sampled bits), so the training still samples the latents like the online one.

Disk space: num_images * num_augs * C * h * w * 2 bytes (float16), e.g. 28000 CelebA-HQ images
with 16 augmentations and C=64, h=w=16 take 14.7 GB.

Usage: uv run scripts/precompute_latents.py --dataset celebahq --split train
"""

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from lab.dataset.images import CelebAHQDataset, LSUNChurchesDataset
from lab.dataset.latents import AUGS_FILE, META_FILE, PROBS_FILE, get_latents_root
from lab.model.bae import BinaryAutoEncoder
from lab.training.runner import AMP_DTYPES
from lab.utils.io import get_root, write_json

DATASETS = {'celebahq': CelebAHQDataset, 'churches': LSUNChurchesDataset}


def sample_augmentations(
    num_images: int,
    num_augs: int,
    num_positions: int,
    seed: int,
) -> torch.Tensor:
    """
    Draw `num_augs` distinct augmentations for each image.

    An augmentation is a crop position (top, left) and a flip, `2 * num_positions**2`
    in total.

    Returns
    -------
    torch.Tensor
        Integer tensor (num_images, num_augs, 3) with (top, left, flip) of each augmentation.
    """
    num_total = 2 * num_positions**2
    generator = torch.Generator().manual_seed(seed)
    # a random subset of the augmentations for each image: the first of a random permutation
    combos = torch.rand(num_images, num_total, generator=generator).argsort(dim=1)[:, :num_augs]
    flip, pos = combos // num_positions**2, combos % num_positions**2
    return torch.stack([pos // num_positions, pos % num_positions, flip], dim=-1)


def extract_crops(imgs: torch.Tensor, augs: torch.Tensor, crop_size: int) -> torch.Tensor:
    """
    Crop and flip every image with each of its augmentations at once (a single gather).

    Parameters
    ----------
    imgs : torch.Tensor
        Images (B, 3, H, W).
    augs : torch.Tensor
        Augmentations (B, K, 3) with (top, left, flip) of each one.
    crop_size : int
        Size of the square crop.

    Returns
    -------
    torch.Tensor
        Crops (B * K, 3, crop_size, crop_size), the K crops of an image go one after another.
    """
    batch, _ = augs.shape[:2]
    top, left, flip = augs.unbind(-1)
    offset = torch.arange(crop_size, device=imgs.device)
    rows = top[..., None] + offset  # (B, K, crop)
    # a flipped crop reads the columns in the reversed order
    cols = left[..., None] + torch.where(flip[..., None] == 1, crop_size - 1 - offset, offset)
    img_idx = torch.arange(batch, device=imgs.device)[:, None, None, None, None]
    channel = torch.arange(imgs.shape[1], device=imgs.device)[None, None, :, None, None]
    crops = imgs[img_idx, channel, rows[:, :, None, :, None], cols[:, :, None, None, :]]
    return crops.flatten(0, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=list(DATASETS), default='celebahq')
    parser.add_argument('--split', choices=['train', 'test'], default='train')
    parser.add_argument('--num-augs', type=int, default=16, help='augmentations per image')
    parser.add_argument('--crop-size', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=16, help='images per step')
    parser.add_argument('--encode-batch-size', type=int, default=128, help='crops per forward')
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--limit', type=int, default=None, help='encode only the first images')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--amp', choices=['bfloat16', 'float16'], default='bfloat16')
    parser.add_argument('--no-amp', action='store_true', help='full precision encoder')
    parser.add_argument('--no-compile', action='store_true', help='do not compile the encoder')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--bae-checkpoint', default='checkpoints/BAE_C64/binaryae_ema_8100000.th')
    parser.add_argument('--codebook-size', type=int, default=64)
    parser.add_argument('--decoder-mult', type=int, default=1)
    parser.add_argument('--output', default=None, help='directory relative to the project root')
    parser.add_argument('--force', action='store_true', help='overwrite existing latents')
    args = parser.parse_args()

    dataset_cls = DATASETS[args.dataset]
    root = (
        get_root() / args.output if args.output else get_latents_root(dataset_cls.name, args.split)
    )
    if (root / META_FILE).exists() and not args.force:
        raise SystemExit(f'{root} already has latents, use --force to overwrite them.')
    # the latents are usable only when the meta is written, i.e. after the complete run
    (root / META_FILE).unlink(missing_ok=True)
    root.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    amp_dtype = None if args.no_amp else AMP_DTYPES[args.amp]

    bae = BinaryAutoEncoder.from_pretrained(
        args.bae_checkpoint,
        codebook_size=args.codebook_size,
        decoder_mult=args.decoder_mult,
    )
    bae = bae.to(device).eval().requires_grad_(False)

    @torch.inference_mode()
    def encode(crops: torch.Tensor) -> torch.Tensor:
        """
        Probabilities of the binary latents of images in [0, 1], as in the online training.
        """
        with torch.autocast(
            torch.device(device).type,
            dtype=amp_dtype,
            enabled=amp_dtype is not None,
        ):
            return bae.quantizer.proj(bae.encoder(crops.float() / 255))

    # images are stored as uint8 at the size used by the training (no instance transforms)
    dataset = dataset_cls(split=args.split, shuffle_index=False, limit=args.limit)
    num_images = len(dataset)
    source_size = dataset[0]['img'].shape[-1]
    num_positions = source_size - args.crop_size + 1
    if not 1 <= args.num_augs <= 2 * num_positions**2:
        raise SystemExit(f'--num-augs must be in [1, {2 * num_positions**2}] for this crop size.')

    shape = tuple(
        encode(torch.zeros(1, 3, args.crop_size, args.crop_size, device=device)).shape[1:]
    )
    # after the probe above, which would compile the encoder for a batch of one
    if not args.no_compile:
        bae.encoder.compile()
    size_gb = num_images * args.num_augs * int(np.prod(shape)) * 2 / 1e9
    print(
        f'{num_images} images x {args.num_augs} augmentations x latents {shape}: {size_gb:.1f} GB '
        f'to {root}'
    )

    augs = sample_augmentations(num_images, args.num_augs, num_positions, args.seed)
    probs_out = np.lib.format.open_memmap(
        root / PROBS_FILE,
        mode='w+',
        dtype=np.float16,
        shape=(num_images, args.num_augs, *shape),
    )

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True
    )
    start = 0
    for batch in tqdm(dataloader, desc=f'{args.dataset} {args.split}'):
        imgs = batch['img'].to(device)
        crops = extract_crops(imgs, augs[start : start + len(imgs)].to(device), args.crop_size)

        probs = torch.cat([encode(chunk) for chunk in crops.split(args.encode_batch_size)])
        probs = probs.to(torch.float16).view(len(imgs), args.num_augs, *shape)
        probs_out[start : start + len(imgs)] = probs.cpu().numpy()
        start += len(imgs)

    probs_out.flush()
    np.save(root / AUGS_FILE, augs.numpy().astype(np.int16))
    write_json(
        {
            'dataset': args.dataset,
            'split': args.split,
            'num_images': num_images,
            'num_augs': args.num_augs,
            'shape': list(shape),
            'dtype': 'float16',
            'crop_size': args.crop_size,
            'source_size': source_size,
            'seed': args.seed,
            'amp': None if amp_dtype is None else args.amp,
            'bae_checkpoint': args.bae_checkpoint,
        },
        root / META_FILE,
    )
    print(f'Done: {root}')


if __name__ == '__main__':
    main()
