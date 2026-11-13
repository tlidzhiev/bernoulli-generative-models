import logging
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm.auto import tqdm

from lab.metrics.base import BaseMetric
from lab.metrics.tracker import MetricTracker
from lab.model.bae import BinaryAutoEncoder
from lab.model.base import BaseModel
from lab.utils.io import get_root

from .runner import BaseRunner

logger = logging.getLogger(__name__)


class Inferencer(BaseRunner):
    """
    Evaluates the model on the dataset and saves predictions, without optimizers and writers.
    """

    def __init__(
        self,
        config: DictConfig,
        device: str,
        model: BaseModel,
        bae: BinaryAutoEncoder,
        dataloaders: dict[str, DataLoader],
        metrics: dict[str, list[BaseMetric]],
        save_path: Path | str,
        batch_transforms: dict[str, dict[str, nn.Sequential]],
        skip_model_load: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        config : DictConfig
            Experiment config with the 'inferencer' section.
        device : str
            Device for tensors and model.
        model : BaseModel
            PyTorch model.
        dataloaders : dict[str, DataLoader]
            Dataloader of each partition.
        metrics : dict[str, list[BaseMetric]]
            Metrics for inference ('inference').
        save_path : Path or str
            Directory to save predictions to.
        batch_transforms : dict[str, dict[str, nn.Sequential]]
            Transforms applied to the whole batch, by tensor name.
        skip_model_load : bool, optional
            Skip loading model weights from the checkpoint.
        """
        assert skip_model_load or config.inferencer.get('from_pretrained') is not None, (
            'Provide checkpoint or set skip_model_load=True'
        )

        self.config = config
        self.cfg_inferencer = self.config.inferencer
        super().__init__(
            device=device,
            model=model,
            metrics=metrics,
            batch_transforms=batch_transforms,
            device_tensors=list(self.cfg_inferencer.device_tensors),
            use_bfloat16=self.cfg_inferencer.get('use_bfloat16', False),
        )

        self.bae = bae

        self.temperature = self.cfg_inferencer.get('temperature', 1.0)
        self.num_steps = self.cfg_inferencer.get('num_steps')
        # number of generated images to save per partition, None to save all of them
        self.save_images = self.cfg_inferencer.get('save_images', 0)

        self.eval_dataloaders = dataloaders

        # Path definition
        self.save_path = get_root() / save_path

        # Define metrics
        self.eval_metric_tracker = MetricTracker(
            *[m.name for m in self.metrics['inference']],
        )

        if not skip_model_load:
            # inferencer.use_ema loads the EMA (shadow) weights of the checkpoint
            self.load_weights(
                self.cfg_inferencer.from_pretrained,
                use_ema=self.cfg_inferencer.get('use_ema', False),
            )

        self.compile_model(self.cfg_inferencer.get('compile'))

    def compile_model(self, compile_cfg: DictConfig | dict[str, Any] | None) -> None:
        """
        Compile the denoiser instead of the whole model: sampling calls the denoiser, not the
        forward of the diffusion model.
        """
        if compile_cfg is None:
            return
        logger.info(f'Compiling denoiser with options: {compile_cfg}')
        self.model.denoiser.compile(**dict(compile_cfg))

    def run_inference(self) -> dict[str, Any]:
        """
        Run inference on each partition.

        Returns
        -------
        dict[str, Any]
            Metrics of each partition.
        """
        part_logs = {}
        for part, dataloader in self.eval_dataloaders.items():
            logs = self._inference_part(part, dataloader)
            part_logs[part] = logs
        return part_logs

    @torch.no_grad()
    def _inference_part(self, part: str, dataloader: DataLoader) -> dict[str, Any]:
        """
        Run inference on a partition and save predictions.

        Returns
        -------
        dict[str, Any]
            Metrics of the partition.
        """
        self.model.eval()
        self.eval_metric_tracker.reset()

        self._num_generated = 0
        self._num_saved = 0

        # Create save dir
        (self.save_path / part).mkdir(exist_ok=True, parents=True)

        for batch in tqdm(dataloader, desc=part):
            self.process_batch(batch=batch, part=part)

        for met in self.metrics['inference']:
            self.eval_metric_tracker.update(met.name, met.compute())
            met.reset()

        logs = self.eval_metric_tracker.result()
        logs['num_generated'] = self._num_generated
        return logs

    @torch.no_grad()
    def generate(self, num_samples: int) -> torch.Tensor:
        """
        Sample binary latents and decode them into images in [0, 1] of shape (B, 3, H, W).
        """
        with self.autocast():
            code = self.model.sample(
                num_samples,
                temperature=self.temperature,
                num_steps=self.num_steps,
            )
            img = self.bae.decode(code)
        return img.float()

    def process_batch(self, batch: dict[str, Any], part: str) -> dict[str, Any]:
        """
        Run a batch through the model, update metrics and save predictions to disk.
        """
        batch = self.prepare_batch(batch, is_train=False)

        batch_size = batch['img'].shape[0]
        batch['fake_img'] = self.generate(batch_size)

        for met in self.metrics['inference']:
            met.update(**batch)

        # Save the first generated images of the partition
        num_to_save = batch_size
        if self.save_images is not None:
            num_to_save = max(0, min(batch_size, self.save_images - self._num_saved))
        for i in range(num_to_save):
            save_image(
                batch['fake_img'][i], self.save_path / part / f'{self._num_generated + i:06}.png'
            )
        self._num_saved += num_to_save
        self._num_generated += batch_size

        return batch
