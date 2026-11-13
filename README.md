# Bernoulli Generative Models

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#examples">Examples</a> •
  <a href="#credits">Credits</a>
</p>

## About

Generative models over binary latents: Binary Latent Diffusion (`bld`, [paper](https://arxiv.org/abs/2304.04820)) and the Bernoulli Flow Model (`bfm`, [paper](https://arxiv.org/abs/2610.11362)), trained on the latents of a frozen pretrained binary autoencoder (C64) for CelebA-HQ and LSUN Churches.

## Installation

### Requirements
- Python 3.15.2
- CUDA (optional, for GPU support)

### Setup

```bash
# Clone the repository
git clone https://github.com/tlidzhiev/bernoulli-generative-models.git
cd bernoulli-generative-models

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv --python 3.15.2
source .venv/bin/activate

# Install dependencies via uv
uv sync

# Install pre-commit
pre-commit install

# Download the pretrained C64 binary autoencoder of the authors (~330 MB) into
# checkpoints/BAE_C64/ and check that it loads
uv run --group download scripts/download_bae.py
```

The binary autoencoder is frozen: during training it encodes the images into binary latents.
If the automatic download fails (Google Drive quota), download `binaryae_ema_8100000.th` from the
[authors' folder](https://drive.google.com/drive/folders/1CFlixXLnEHZ0jaRLXLS4d_bJMZwG-Iih)
and put it to `checkpoints/BAE_C64/`.

### Precomputing latents

To avoid running the autoencoder encoder during training, precompute the training latents:

```bash
uv run scripts/precompute_latents.py --dataset celebahq --split train
```

Latents are saved to `data/celebahq/latents/train/` (about 14.7 GB with the default 16
augmentations). Train on them with:

```bash
uv run train dataset=celebahq_latents
```

Use `--dataset churches` to precompute LSUN Churches latents. See
`uv run scripts/precompute_latents.py --help` for options such as the number of augmentations and
output directory. A completed output directory cannot be reused unless you pass `--force`.

### CometML Configuration

CometML is used by default for experiment tracking. Create a `.comet.config` file in the project root:

```ini
[comet]
api_key=YOUR_API_KEY
workspace=YOUR_WORKSPACE
project_name=YOUR_PROJECT_NAME
```

The API key is only needed for online logging: `writer.mode` is `offline` by default.

## How To Use

To run a script (training or inference), use the following command:

```bash
uv run <script> -cn=CONFIG_NAME HYDRA_CONFIG_ARGUMENTS
```

- `<script>`: Use `train` for training or `inference` for inference
- `CONFIG_NAME`: A config file from `src/lab/configs` (`train` and `inference` by default)
- `HYDRA_CONFIG_ARGUMENTS`: Optional Hydra overrides

By default the Binary Latent Diffusion (`model=bld`) is trained. To train the Bernoulli Flow Model instead
(the matching loss `criterion/<model>.yaml` is selected automatically):

```bash
uv run train model=bfm
```

## Examples

Train the Binary Latent Diffusion on the precomputed latents:

```bash
uv run train dataset=celebahq_latents writer.run_name=BLD_celeba_latents writer.mode=online \
  dataloader.train.batch_size=256 trainer.grad_accum_steps=1 trainer.compile={mode: default}
```

Train the Bernoulli Flow Model:

```bash
uv run train dataset=celebahq_latents model=bfm writer.run_name=BFM_celeba_latents writer.mode=online \
  dataloader.train.batch_size=256 trainer.grad_accum_steps=1 trainer.compile={mode: default}
```

Generate images with the EMA weights, compute FID and save the results to `data/saved/bld`:

```bash
uv run inference inferencer.from_pretrained=path/to/model-best.pth \
  inferencer.use_ema=true inferencer.save_path=bld
```

## Credits

The code of this project is based on the [PyTorch project template](https://github.com/Blinorot/pytorch_project_template).
