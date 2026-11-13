import torch
from torch import nn
from torch.nn import functional as F
from torchmetrics.image.fid import FrechetInceptionDistance

from .base import BaseMetric


def to_uint8(img: torch.Tensor) -> torch.Tensor:
    """
    Convert images in [0, 1] to uint8 in [0, 255] (rounding, not truncating).
    """
    return (img.clamp(0, 1) * 255).round().to(torch.uint8)


class DINOv2Features(nn.Module):
    """
    Frozen DINOv2 feature extractor (CLS token) for the Fréchet distance (FD_DINOv2).

    Takes uint8 images in [0, 255] like the InceptionV3 of torchmetrics. The weights are
    loaded with `timm` (optional dependency: `uv sync --extra dinov2`).
    """

    mean: torch.Tensor
    std: torch.Tensor

    def __init__(
        self,
        model_name: str = 'vit_large_patch14_dinov2.lvd142m',
        image_size: int = 224,
    ) -> None:
        """
        Parameters
        ----------
        model_name : str, optional
            Name of the DINOv2 model in timm, ViT-L/14 by default as in FD_DINOv2.
        image_size : int, optional
            Images are resized to image_size x image_size (bicubic), must divide by 14.
        """
        super().__init__()
        try:
            import timm
        except ImportError:
            raise ImportError('DINOv2 features require timm, run: uv sync --extra dinov2') from None

        self.image_size = image_size
        self.model = timm.create_model(
            model_name, pretrained=True, num_classes=0, img_size=image_size
        )
        self.num_features: int = self.model.num_features  # ty:ignore[invalid-assignment]
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.eval().requires_grad_(False)

    def train(self, mode: bool = True) -> DINOv2Features:
        # always in eval mode, also when the metric is moved to another mode
        return super().train(False)

    @torch.no_grad()
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        x = img.float() / 255
        x = F.interpolate(
            x, size=(self.image_size, self.image_size), mode='bicubic', antialias=True
        ).clamp(0, 1)
        return self.model((x - self.mean) / self.std)


class FID(BaseMetric):
    """
    Fréchet distance between the real ('img') and the generated ('fake_img') images,
    accumulated over all batches with `torchmetrics.image.fid.FrechetInceptionDistance`.

    Images are expected in [0, 1] of shape (B, 3, H, W).
    """

    def __init__(
        self,
        features: str = 'inception',
        device: str = 'auto',
        dinov2_name: str = 'vit_large_patch14_dinov2.lvd142m',
        name: str | None = None,
    ) -> None:
        """
        Parameters
        ----------
        features : {'inception', 'dinov2'}, optional
            Feature extractor: InceptionV3 (the standard FID, torch-fidelity weights) or
            DINOv2 (FD_DINOv2).
        device : str, optional
            Device name or 'auto'.
        dinov2_name : str, optional
            timm model name of DINOv2, used only with features='dinov2'.
        name : str or None, optional
            Metric name, 'FID' for inception and 'FD_DINOv2' for DINOv2 by default.
        """
        match features:
            case 'inception':
                feature, default_name = 2048, 'FID'
            case 'dinov2':
                feature, default_name = DINOv2Features(dinov2_name), 'FD_DINOv2'
            case _:
                raise ValueError(f"features must be 'inception' or 'dinov2', got '{features}'.")
        super().__init__(name=name or default_name)

        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # images are passed as uint8 in [0, 255] (normalize=False)
        self.fid = FrechetInceptionDistance(feature=feature, normalize=False).to(device)
        self.device = device

    @torch.no_grad()
    def update(self, img: torch.Tensor, fake_img: torch.Tensor, **kwargs) -> None:  # ty:ignore[invalid-method-override]
        self.fid.update(to_uint8(img).to(self.device), real=True)
        self.fid.update(to_uint8(fake_img).to(self.device), real=False)

    def compute(self) -> float:
        return float(self.fid.compute())

    def reset(self) -> None:
        self.fid.reset()

    def __str__(self) -> str:
        return f'{type(self).__name__}(name={self.name}, device={self.device})'
