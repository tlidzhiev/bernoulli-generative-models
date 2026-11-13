"""
Training loss of the Bernoulli diffusion over binary latents.

Rewritten from the official implementation of
"Binary Latent Diffusion" (Wang et al., CVPR 2023), https://arxiv.org/abs/2304.04820.
Original code: https://github.com/ZeWang95/BinaryLatentDiffusion
(models/binarylatent.py, `BinaryDiffusion._train_loss` and `focal_loss`).
"""

from typing import Literal

import torch
from torch.nn import functional as F

from lab.model.diffusion import NoiseScheduler

from .base import BaseLoss


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = -1.0,
    gamma: float = 1.0,
) -> torch.Tensor:
    """
    Element-wise binary focal loss.

    Parameters
    ----------
    logits : torch.Tensor
        Logits of shape (B, ...).
    targets : torch.Tensor
        Binary targets of the same shape.
    alpha : float, optional
        Class weight of positives. -1 balances the classes per sample (each class is
        weighted by the share of the other one), values <= 0 other than -1 disable weighting.
    gamma : float, optional
        Focusing parameter, gamma = 0 gives (weighted) BCE.
    """
    ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p = torch.sigmoid(logits)
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * (1 - p_t).clamp(min=1e-6, max=1 - 1e-6) ** gamma
    if alpha == -1:
        dims = tuple(range(1, targets.ndim))
        neg_weight = targets.mean(dim=dims, keepdim=True)
        loss = ((1 - neg_weight) * targets + neg_weight * (1 - targets)) * loss
    elif alpha > 0:
        loss = (alpha * targets + (1 - alpha) * (1 - targets)) * loss
    return loss


class BinaryDiffusionLoss(BaseLoss):
    def __init__(
        self,
        num_timesteps: int = 64,
        p_flip: bool = True,
        weighting: Literal['mean', 'weighted'] = 'mean',
        focal: float | None = 0.0,
        aux: float = 0.0,
        schedule: str = 'linear',
    ) -> None:
        """
        Loss of `BinaryDiffusion`: BCE (or focal loss) on x_0 plus an optional
        auxiliary BCE on the posterior q(x_{t-1} | x_t, x_0).

        Defaults follow the released recipe (`--p_flip --loss_final mean`, focal = 0).

        Parameters
        ----------
        num_timesteps : int, optional
            Number of diffusion steps, must match the model.
        p_flip : bool, optional
            Whether the denoiser predicts bit flips x_0 xor x_t instead of x_0, must match the model.
        weighting : {'mean', 'weighted'}, optional
            'weighted' scales the loss at step t by 1 - (t - 1) / T.
        focal : float or None, optional
            Gamma of the focal loss, None for plain BCE. As in the original, the focal loss
            is class-balanced (alpha = -1) with `p_flip` and uses alpha = gamma without it.
        aux : float, optional
            Weight of the auxiliary posterior loss, 0 disables it.
        schedule : str, optional
            Noise schedule, must match the model (used only by the auxiliary loss).
        """
        super().__init__()
        self.num_timesteps = num_timesteps
        self.p_flip = p_flip
        self.weighting = weighting
        self.focal = focal
        self.aux = aux
        self.scheduler = NoiseScheduler(num_timesteps, schedule) if aux > 0 else None
        self.loss_names = ['loss', 'bce_loss', 'acc'] + (['aux_loss'] if aux > 0 else [])

    def forward(
        self,
        logits: torch.Tensor,
        x0: torch.Tensor,
        xt: torch.Tensor,
        t: torch.Tensor,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """
        Compute the losses from the outputs of `BinaryDiffusion.forward`.

        Parameters
        ----------
        logits : torch.Tensor
            Denoiser logits of shape (B, L, C).
        x0 : torch.Tensor
            Clean binary latents of shape (B, L, C).
        xt : torch.Tensor
            Noisy binary latents of shape (B, L, C).
        t : torch.Tensor
            Diffusion steps of shape (B,), in [1, num_timesteps].
        """
        logits = logits.float()
        target = torch.logical_xor(x0, xt).float() if self.p_flip else x0
        # logits of x_0 = 1
        x0_logits = torch.where(xt.bool(), -logits, logits) if self.p_flip else logits

        if self.focal is None:
            bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
        else:
            alpha = -1.0 if self.p_flip else self.focal
            bce = focal_loss(logits, target, alpha=alpha, gamma=self.focal)

        if self.weighting == 'weighted':
            weight = 1 - (t - 1).float() / self.num_timesteps
            weight = weight.view(-1, *([1] * (bce.ndim - 1)))
        else:
            weight = torch.ones((), device=bce.device)

        losses = {'loss': (weight * bce).mean(), 'bce_loss': bce.mean()}
        with torch.no_grad():
            losses['acc'] = ((x0_logits > 0).float() == x0).float().mean()

        if self.scheduler is not None:
            x0_probs = torch.sigmoid(x0_logits)
            pred = self.scheduler.posterior(x0_probs, xt, t)
            # x_{t-1} = x_0 at the last step
            pred = torch.where((t == 1).view(-1, *([1] * (pred.ndim - 1))), x0_probs, pred)
            gt = self.scheduler.posterior(x0, xt, t)
            aux_loss = F.binary_cross_entropy(
                pred.clamp(min=1e-6, max=1 - 1e-6), gt.clamp(min=0.0, max=1.0), reduction='none'
            )
            losses['aux_loss'] = (weight * aux_loss).mean()
            losses['loss'] = losses['loss'] + self.aux * losses['aux_loss']
        return losses
