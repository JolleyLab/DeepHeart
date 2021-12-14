import logging
import os
from pathlib import Path

from lib import VNet
from lib.infer import *

from monailabel.interfaces.app import MONAILabelApp
from distutils.util import strtobool

logger = logging.getLogger(__name__)


class MyApp(MONAILabelApp):

    def __init__(self, app_dir, studies, conf):
        self.model_dir = os.path.join(app_dir, "model")
        import torch
        self.device = 'GPU' if torch.cuda.is_available() else 'CPU'
        print(f"Using {self.device}")

        description = "TODO"

        super().__init__(
            app_dir=app_dir,
            studies=studies,
            conf=conf,
            name="Valve Segmentation",
            version="0.1",
            description=description
        )

    def init_infers(self):
        model_dir = Path(self.model_dir)

        network_params = dict(
            n_classes=4,
            n_filters=16,
            normalization="batchnorm"
        )

        model_paths = [
            str(model_dir / f"tricuspid_ms_ann_{self.device}.pt"),
            str(model_dir / f"tricuspid_ms_md_ann_{self.device}.pt"),
            str(model_dir / f"tricuspid_ms_ann_com_{self.device}.pt"),
            str(model_dir / f"tricuspid_ms_md_ann_com_{self.device}.pt"),
            str(model_dir / f"tricuspid_ms_md_ann_com_single_label_{self.device}.pt"),
        ]

        if strtobool(self.conf.get("use_pretrained_model", "true")) is True:
            # logger.info(f"Pretrained Model Path: {pretrained_model_uri}")
            self.download(model_paths)

        return {
            "Mid-Systole__Annulus":
                TricuspidInferenceTaskSinglePhaseAnn(model_paths[0], VNet(**network_params, n_channels=2)),
            # "Mid-Systole__Mid-Diastole__Annulus":
            #     TricuspidInferenceTaskTwoPhaseAnn(model_paths[1], VNet(**network_params, n_channels=3)),
            "Mid-Systole__Annulus__Commissures":
                TricuspidInferenceTaskSinglePhaseAnnCom(model_paths[2], VNet(**network_params, n_channels=5)),
            "Mid-Systole__Mid-Diastole__Annulus__Commissures":
                TricuspidInferenceTaskTwoPhaseAnnCom(model_paths[3], VNet(**network_params, n_channels=6)),
            # "Mid-Systole__Mid-Diastole__Annulus__Commissures__Alt":
            #     TricuspidInferenceTaskTwoPhaseAnnComOneLabel(model_paths[4], VNet(**network_params, n_channels=4)),
        }

    def init_strategies(self):
        return {}

    def train(self, request):
        return {}

