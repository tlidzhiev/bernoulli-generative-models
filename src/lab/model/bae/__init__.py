"""
Binary autoencoder (VQGAN-style encoder/decoder with a Bernoulli quantizer).

Rewritten from the official implementation of
"Binary Latent Diffusion" (Wang et al., CVPR 2023), https://arxiv.org/abs/2304.04820.
Original code: https://github.com/ZeWang95/BinaryLatentDiffusion (models/binaryae.py).
"""

from .model import BinaryAutoEncoder

__all__ = ['BinaryAutoEncoder']
