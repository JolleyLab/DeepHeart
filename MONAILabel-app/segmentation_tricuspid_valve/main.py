import json
import logging
import os

from lib import MyInfer, MyStrategy, VNet

from monailabel.interfaces import MONAILabelApp
from monailabel.utils.activelearning import Random

logger = logging.getLogger(__name__)


class MyApp(MONAILabelApp):
    def __init__(self, app_dir, studies):
        self.model_dir = os.path.join(app_dir, "model")
        # TODO: depending on selected model, a different network needs to be selected
        self.network = VNet(
            n_channels=2,
            n_classes=4,
            n_filters=16,
            normalization="batchnorm"
        )

        self.pretrained_model = os.path.join(self.model_dir, "segmentation_tricuspid_valve.pt")
        self.final_model = os.path.join(self.model_dir, "final.pt")
        self.train_stats_path = os.path.join(self.model_dir, "train_stats.json")

        path = [self.pretrained_model, self.final_model]
        infers = {
            "segmentation_tricuspid_valve": MyInfer(path, self.network),
        }

        strategies = {
            "random": Random(),
            "first": MyStrategy(),
        }

        resources = [
            (
                self.pretrained_model,
                # "https://api.ngc.nvidia.com/v2/models/nvidia/med"
                # "/clara_pt_liver_and_tumor_ct_segmentation/versions/1/files/models/model.pt",
            ),
        ]

        super().__init__(
            app_dir=app_dir,
            studies=studies,
            infers=infers,
            strategies=strategies,
            resources=resources,
        )

    def train(self, request):
        pass

    def train_stats(self):

        if os.path.exists(self.train_stats_path):
            with open(self.train_stats_path, "r") as fc:
                return json.load(fc)
        return super().train_stats()
