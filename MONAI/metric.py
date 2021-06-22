import torch

from monai.handlers.iteration_metric import IterationMetric


class LossMetric(IterationMetric):

    def __init__(
        self,
        metric_fn,
        output_transform=lambda x: x,
        device="cpu",
        save_details=True,
    ):
        super().__init__(
            metric_fn=metric_fn,
            output_transform=output_transform,
            device=device,
            save_details=save_details,
        )

    def compute(self):
        return torch.Tensor(self._scores).mean()