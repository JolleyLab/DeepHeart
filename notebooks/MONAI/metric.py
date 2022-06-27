import torch

# from monai.metrics.metric import IterationMetric
from monai.handlers.ignite_metric import IgniteMetric


class LossMetric(IgniteMetric):

    def __init__(
        self,
        metric_fn,
        output_transform=lambda x: x,
        save_details=True,
    ):
        super().__init__(
            metric_fn=metric_fn,
            output_transform=output_transform,
            save_details=save_details,
        )

    def compute(self):
        return torch.Tensor(self._scores).mean()
    
    def reset(self):
        pass