"""
Binary latent diffusion (Bernoulli diffusion over binary latents).

Rewritten from the official implementation of
"Binary Latent Diffusion" (Wang et al., CVPR 2023), https://arxiv.org/abs/2304.04820.
Original code: https://github.com/ZeWang95/BinaryLatentDiffusion (models/).
"""

from .model import BinaryDiffusion
from .scheduler import NoiseScheduler

__all__ = ['BinaryDiffusion', 'NoiseScheduler']
