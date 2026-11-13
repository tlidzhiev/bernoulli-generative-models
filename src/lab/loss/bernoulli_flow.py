"""
Training loss of the Bernoulli flow matching over binary latents.

Rewritten from the official implementation of
"Bernoulli Flow Models: Self-Consistent Generative Modeling for Binary Data" (Mo et al.,
NeurIPS 2026), https://arxiv.org/abs/2610.11362.
Original code: https://github.com/minimo6666/BFM_NeurIPS2026
(models/bernoulli_flow_models.py, `BinaryFlowMatching.training_loss` and `focal_loss`).
"""

import torch

from .base import BaseLoss
from .binary_diffusion import focal_loss


class BernoulliFlowLoss(BaseLoss):
    def __init__(self, focal: float = 0.0) -> None:
        """
        Loss of `BernoulliFlowMatching`: class-balanced focal loss on the bit flips x_0 xor x_t.

        Parameters
        ----------
        focal : float, optional
            Gamma of the focal loss, the paper uses 0 (class-balanced BCE).
        """
        super().__init__()
        assert focal >= 0, 'focal must be non-negative'
        self.focal = focal
        self.loss_names = ['loss', 'bce_loss', 'acc']

    def forward(
        self,
        logits: torch.Tensor,
        x0: torch.Tensor,
        xt: torch.Tensor,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """
        Compute the losses from the outputs of `BernoulliFlowMatching.forward`.

        Parameters
        ----------
        logits : torch.Tensor
            Logits of the bit flips of shape (B, L, C).
        x0 : torch.Tensor
            Clean binary latents of shape (B, L, C).
        xt : torch.Tensor
            Noisy binary latents of shape (B, L, C).
        """
        logits = logits.float()
        target = torch.logical_xor(x0, xt).float()
        loss = focal_loss(logits, target, alpha=-1.0, gamma=self.focal).mean()

        losses = {'loss': loss, 'bce_loss': loss.detach()}
        with torch.no_grad():
            # logits of x_0 = 1
            x0_logits = torch.where(xt.bool(), -logits, logits)
            losses['acc'] = ((x0_logits > 0).float() == x0).float().mean()
        return losses
