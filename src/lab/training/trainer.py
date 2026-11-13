from typing import Any

import torch
from torchvision.utils import make_grid

from lab.model.bae import BinaryAutoEncoder
from lab.model.base import BaseModel

from .base import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer of the binary latent diffusion: images are encoded into binary latents online
    by a frozen binary autoencoder, and the diffusion model is trained on them.
    """

    model: BaseModel

    def __init__(self, *args, bae: BinaryAutoEncoder, **kwargs) -> None:
        """
        Parameters
        ----------
        bae : BinaryAutoEncoder
            Frozen (and compiled) binary autoencoder that encodes the images into latents.
        *args, **kwargs
            Arguments of `BaseTrainer`.
        """
        super().__init__(*args, **kwargs)
        self.bae = bae
        cfg = self.cfg_trainer
        self.deterministic_latents = cfg.deterministic_latents
        # Generation of the images logged at the end of each evaluation epoch
        self.num_log_images = cfg.log_images
        self.sample_temperature = cfg.sample_temperature
        self.sample_num_steps = cfg.sample_num_steps
        # Fixed seed of the sampling noise: the same noise is denoised each epoch, so the
        # logged images differ only because the model has changed (None for a random noise)
        self.sample_seed = cfg.sample_seed

    def process_batch(self, batch: dict[str, Any], part: str) -> dict[str, Any]:
        """
        Run a batch through the model, compute loss and metrics, and backpropagate the loss.
        """
        is_train = part == 'train'
        metric_tracker = self.train_metric_tracker if is_train else self.eval_metric_tracker

        batch = self.prepare_batch(batch, is_train=is_train)

        # Precomputed latents (LatentDataset) or encoding of the images online
        if 'probs' in batch:
            batch['code'] = self.sample_code(batch['probs'])
        else:
            batch['code'] = self.encode(batch['img'])

        with self.autocast():
            outputs = self.model(**batch)
            batch.update(outputs)

            all_losses = self.criterion(**batch)
            batch.update(all_losses)

        if is_train:
            self.backward(batch['loss'])  # Sum of all losses is always called loss

        # Update metrics for each loss (in case of multiple losses)
        for loss_name in self.criterion.loss_names:
            metric_tracker.update(loss_name, batch[loss_name].detach())

        for met in self.metrics['train' if is_train else 'inference']:
            if is_train:
                metric_tracker.update(met.name, met(**batch))
            else:
                # Accumulated over the whole partition, computed in '_evaluate_part'
                met.update(**batch)

        return batch

    @torch.no_grad()
    def encode(self, img: torch.Tensor) -> torch.Tensor:
        """
        Encode images in [0, 1] of shape (B, 3, H, W) into binary latents (B, C, h, w).
        """
        with self.autocast():
            return self.bae.encode(img, deterministic=self.deterministic_latents, code_only=True)

    def sample_code(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Binary latents from their precomputed probabilities, as the online encoder does.
        """
        probs = probs.float()
        if self.deterministic_latents:
            return torch.gt(probs, 0.5).float()
        return torch.bernoulli(probs)

    @torch.no_grad()
    def _log_batch(self, batch: dict[str, Any], part: str) -> None:
        """
        At the end of an evaluation epoch, generate images (latents sampled by the model and
        decoded by the BAE) and log them as a grid. Nothing is logged during training.
        """
        if part == 'train' or self.num_log_images <= 0:
            return

        fake_img = self.generate(self.num_log_images, seed=self.sample_seed)
        grid = make_grid(fake_img, nrow=int(self.num_log_images**0.5), padding=2)
        grid = grid.mul(255).round().clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
        self.writer.add_image('generated', grid)

    @torch.no_grad()
    def generate(self, num_samples: int, seed: int | None = None) -> torch.Tensor:
        """
        Sample binary latents with the current model and decode them into images in [0, 1]
        of shape (B, 3, H, W).

        Parameters
        ----------
        num_samples : int
            Number of images.
        seed : int or None, optional
            Seed of the initial noise and of the sampling steps. The random state of the
            training is restored afterwards, so it does not depend on the logging.
        """
        devices = [torch.device(self.device)] if torch.device(self.device).type == 'cuda' else []
        with torch.random.fork_rng(devices=devices, enabled=seed is not None):
            if seed is not None:
                torch.manual_seed(seed)
            with self.autocast():
                code = self.model.sample(
                    num_samples,
                    temperature=self.sample_temperature,
                    num_steps=self.sample_num_steps,
                )
                img = self.bae.decode(code)
        return img.float()
