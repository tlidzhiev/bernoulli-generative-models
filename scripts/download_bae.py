"""
Download the pretrained C64 binary autoencoder of the authors of
"Binary Latent Diffusion" (Wang et al., CVPR 2023) and check that it loads.

Needs the `download` dependency group: uv run --group download scripts/download_bae.py

Source: https://drive.google.com/drive/folders/1CFlixXLnEHZ0jaRLXLS4d_bJMZwG-Iih
"""

import argparse

from lab.model.bae import BinaryAutoEncoder
from lab.utils.io import get_root

# saved_models/binaryae_ema_8100000.th in the Google Drive folder of the authors
FILE_ID = '1DSS0gY2L4bvvK2vtoC6lwzwEJ39K_Z0O'
# must match configs/bae/c64.yaml
DEFAULT_PATH = 'checkpoints/BAE_C64/binaryae_ema_8100000.th'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--output',
        default=DEFAULT_PATH,
        help='checkpoint path relative to the project root (default: %(default)s)',
    )
    parser.add_argument('--force', action='store_true', help='download even if the file exists')
    parser.add_argument('--skip-check', action='store_true', help='do not load the weights')
    args = parser.parse_args()

    try:
        import gdown  # ty:ignore[import-not-found]
    except ImportError:
        raise SystemExit('gdown is not installed, run: uv sync --group data') from None

    path = get_root() / args.output
    if path.exists() and not args.force:
        print(f'{path} already exists, use --force to download it again.')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        if gdown.download(id=FILE_ID, output=str(path)) is None:
            raise SystemExit(
                'Download failed (Google Drive quota?). Download the file manually from '
                f'https://drive.google.com/file/d/{FILE_ID} to {path}.'
            )

    if not args.skip_check:
        # strict=True: raises on any missing or unexpected key of the converted state dict
        BinaryAutoEncoder.from_pretrained(args.output, codebook_size=64, decoder_mult=1)
        print('Checkpoint loaded into BinaryAutoEncoder (strict).')


if __name__ == '__main__':
    main()
