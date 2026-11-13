"""
Denoising backbones shared by the generative models over binary latents.

Rewritten from the official implementations of
"Binary Latent Diffusion" (Wang et al., CVPR 2023), https://arxiv.org/abs/2304.04820, and
"Bernoulli Flow Models: Self-Consistent Generative Modeling for Binary Data" (Mo et al.,
NeurIPS 2026), https://arxiv.org/abs/2610.11362.
"""

from .transformer import DenoisingTransformer

__all__ = ['DenoisingTransformer']
