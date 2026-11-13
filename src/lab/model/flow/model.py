"""
Bernoulli flow matching over binary latents: linear Bernoulli path, flip prediction and
plug-in reverse transitions.

Rewritten from the official implementation of
"Bernoulli Flow Models: Self-Consistent Generative Modeling for Binary Data" (Mo et al.,
NeurIPS 2026), https://arxiv.org/abs/2610.11362.
Original code: https://github.com/minimo6666/BFM_NeurIPS2026 (models/bernoulli_flow_models.py).
"""

import torch
from einops import rearrange

from ..base import BaseModel
from ..denoiser import DenoisingTransformer


class BernoulliFlowMatching(BaseModel):
    denoiser: DenoisingTransformer

    def __init__(
        self,
        denoiser: DenoisingTransformer,
        num_timesteps: int = 64,
        eps: float = 1e-30,
    ) -> None:
        """
        Parameters
        ----------
        denoiser : DenoisingTransformer
            Unconditional network predicting the logits of the bit flips x_0 xor x_t.
        num_timesteps : int, optional
            Number of steps T of the time grid t / T, must match the denoiser.
        eps : float, optional
            Numerical guard of the posterior normalization.
        """
        super().__init__()
        assert denoiser.num_timesteps == num_timesteps, (
            f'Denoiser has {denoiser.num_timesteps} timesteps, expected {num_timesteps}.'
        )
        assert denoiser.condition == 'none', 'BernoulliFlowMatching is unconditional.'
        self.denoiser = denoiser
        self.num_timesteps = num_timesteps
        self.eps = eps
        # share of the clean signal kept at step t: 1 - t / T
        self.register_buffer(
            'retention', torch.linspace(1.0, 0.0, num_timesteps + 1), persistent=False
        )

    @staticmethod
    def _expand(coef: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return coef.view(-1, *([1] * (x.ndim - 1)))

    def marginal(self, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Probability q(x_t = 1 | x_0) = (1 + (2 x_0 - 1) k_t) / 2 for (soft) x_0 of shape (B, ...).
        """
        k = self._expand(self.retention[t], x0)  # ty:ignore[not-subscriptable]
        return (1 + (2 * x0 - 1) * k) / 2

    def posterior(
        self,
        x0: torch.Tensor,
        xt: torch.Tensor,
        t: torch.Tensor,
        s: torch.Tensor,
    ) -> torch.Tensor:
        """
        Plug-in reverse transition q(x_s = 1 | x_t, x_0) for (soft) x_0 and binary x_t.

        Parameters
        ----------
        x0 : torch.Tensor
            Probabilities of x_0 = 1, of shape (B, ...).
        xt : torch.Tensor
            Binary x_t of shape (B, ...).
        t : torch.Tensor
            Current steps of shape (B,), in [1, num_timesteps].
        s : torch.Tensor
            Target steps of shape (B,), s < t.
        """
        prior = self.marginal(torch.stack([x0, 1 - x0], dim=-1), s)
        # x_s -> x_t is a binary symmetric channel with flip probability (1 - k_t / k_s) / 2
        k_t, k_s = self.retention[t], self.retention[s]  # ty:ignore[not-subscriptable]
        flip = (0.5 * (k_s - k_t) / (k_s + self.eps)).clamp(0.0, 0.5)
        states = torch.stack([xt, 1 - xt], dim=-1)
        flip = self._expand(flip, states)
        likelihood = states * (1 - flip) + (1 - states) * flip
        probs = prior * likelihood
        probs = probs / probs.sum(-1, keepdim=True).clamp_min(self.eps)
        return probs[..., 0].clamp(0.0, 1.0)

    def forward(self, code: torch.Tensor, **kwargs) -> dict[str, torch.Tensor]:
        x0 = rearrange(code, 'b c h w -> b (h w) c').float()
        t = torch.randint(1, self.num_timesteps + 1, (x0.shape[0],), device=x0.device)
        xt = torch.bernoulli(self.marginal(x0, t))
        logits = self.denoiser(xt, t - 1)
        return {'logits': logits, 'x0': x0, 'xt': xt, 't': t}

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        temperature: float = 1.0,
        num_steps: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        num_steps = num_steps or self.num_timesteps
        assert 1 <= num_steps <= self.num_timesteps, (
            f'num_steps must be in [1, {self.num_timesteps}]'
        )
        device = next(self.denoiser.parameters()).device
        size, codebook_size = self.denoiser.latent_size, self.denoiser.codebook_size
        xt = torch.bernoulli(torch.full((num_samples, size**2, codebook_size), 0.5, device=device))

        # evenly spaced subset of the steps (integer arithmetic, no float rounding), descending
        idx = torch.arange(num_steps) * (self.num_timesteps - 1) // max(num_steps - 1, 1)
        steps = (idx + 1).flip(0).tolist()

        for i, step in enumerate(steps):
            t = torch.full((num_samples,), step, device=device, dtype=torch.long)
            flip_logits = self.denoiser(xt, t - 1).float() / temperature
            # probability of x_0 = 1
            x0_probs = torch.sigmoid(torch.where(xt.bool(), -flip_logits, flip_logits))

            if step == 1:
                xt = (x0_probs > 0.5).float()
            else:
                s = torch.full_like(t, steps[i + 1])
                xt = torch.bernoulli(self.posterior(x0_probs, xt, t, s))

        return rearrange(xt, 'b (h w) c -> b c h w', h=size, w=size)
