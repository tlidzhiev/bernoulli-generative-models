"""
Bernoulli flow matching over binary latents.

Rewritten from the official implementation of
"Bernoulli Flow Models: Self-Consistent Generative Modeling for Binary Data" (Mo et al.,
NeurIPS 2026), https://arxiv.org/abs/2610.11362.
Original code: https://github.com/minimo6666/BFM_NeurIPS2026 (models/).
"""

from .model import BernoulliFlowMatching

__all__ = ['BernoulliFlowMatching']
