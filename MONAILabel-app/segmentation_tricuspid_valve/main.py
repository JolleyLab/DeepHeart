import logging
import os
from pathlib import Path

from lib import VNet
from monai.networks.layers.factories import Norm
from monai.networks.nets import UNet
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
            str(model_dir / f"tricuspid_diastolic_ann_com_{self.device}.pt"),
        ]

        if strtobool(self.conf.get("use_pretrained_model", "true")) is True:
            # logger.info(f"Pretrained Model Path: {pretrained_model_uri}")
            self.download(model_paths)

        tricuspid_labels = ("tricuspid anterior leaflet", "tricuspid posterior leaflet", "tricuspid septal leaflet")
        lavv_labels = ("superior bridging leaflet", "inferior bridging leaflet", "left mural leaflet")
        return {
            "Tricuspid__MS__Annulus":
                SlicerHeartInferenceTaskSinglePhaseAnn(
                    model_paths[0],
                    network=VNet(**network_params, n_channels=2),
                    labels=tricuspid_labels,
                    valve_type="tricuspid",
                    cardiac_phase_frames=["MS"],
                    landmark_labels=["APC", "ASC", "PSC"],
                    export_keys=[
                        'mid-systolic-images',
                        'mid-systolic-annulus'
                    ]
                ),
            "Tricuspid__MS__Annulus__Commissures":
                TricuspidInferenceTaskSinglePhaseAnnCom(
                    model_paths[2],
                    VNet(**network_params, n_channels=5),
                    labels=tricuspid_labels,
                    valve_type="tricuspid",
                    cardiac_phase_frames=["MS"],
                    landmark_labels=["APC", "ASC", "PSC"],
                    export_keys= [
                        'mid-systolic-images',
                        'mid-systolic-annulus',
                        'mid-systolic-APC',
                        'mid-systolic-ASC',
                        'mid-systolic-PSC'
                    ]
                ),
            "Tricuspid__MS__MD__Annulus__Commissures":
                TricuspidInferenceTaskTwoPhaseAnnCom(
                    model_paths[3],
                    network=VNet(**network_params, n_channels=6),
                    labels=tricuspid_labels,
                    valve_type="tricuspid",
                    cardiac_phase_frames=["MS", "MD"],
                    landmark_labels=["APC", "ASC", "PSC"],
                    export_keys=[
                        'mid-systolic-images',
                        'mid-diastolic-images',
                        'mid-systolic-annulus',
                        'mid-systolic-APC',
                        'mid-systolic-ASC',
                        'mid-systolic-PSC'
                    ]
                ),
            "LAVV__MS__Annulus":
                SlicerHeartInferenceTaskSinglePhaseAnn(
                    model_paths[0],
                    network=VNet(**network_params, n_channels=2),
                    labels=lavv_labels,
                    valve_type="lavv",
                    cardiac_phase_frames=["MS"],
                    landmark_labels=["SIC", "ALC", "PMC"],
                    export_keys=[
                        'mid-systolic-images',
                        'mid-systolic-annulus'
                    ]
                ),
            "LAVV__MS__Annulus__Commissures":
                LavvInferenceTaskSinglePhaseAnnCom(
                    model_paths[2],
                    VNet(**network_params, n_channels=5),
                    labels=lavv_labels,
                    valve_type="lavv",
                    cardiac_phase_frames=["MS"],
                    landmark_labels=["SIC", "ALC", "PMC"],
                    export_keys=[
                        'mid-systolic-images',
                        'mid-systolic-annulus',
                        'mid-systolic-SIC',
                        'mid-systolic-ALC',
                        'mid-systolic-PMC'
                    ]
                ),
            "LAVV__MS__MD__Annulus__Commissures":
                LavvInferenceTaskTwoPhaseAnnCom(
                    model_paths[3],
                    network=VNet(**network_params, n_channels=6),
                    labels=lavv_labels,
                    valve_type="lavv",
                    cardiac_phase_frames=["MS", "MD"],
                    landmark_labels=["SIC", "ALC", "PMC"],
                    export_keys=[
                        'mid-systolic-images',
                        'mid-diastolic-images',
                        'mid-systolic-annulus',
                        'mid-systolic-SIC',
                        'mid-systolic-ALC',
                        'mid-systolic-PMC'
                    ]
                ),
            "Tricuspid_MD__Annulus__Commissures":
                TricuspidDiastolicInferenceTaskSinglePhaseAnnCom(
                    model_paths[5],
                    network=UNet(
                        spatial_dims=3,
                        in_channels=5,
                        out_channels=4,
                        channels=(16, 32, 64, 128, 256),
                        strides=(2, 2, 2, 2),
                        num_res_units=2,
                        norm=Norm.BATCH,
                        adn_ordering="NDA"
                    ),
                    labels=tricuspid_labels,
                    valve_type="tricuspid",
                    cardiac_phase_frames=["MD"],
                    landmark_label_phases=["MD"],
                    annulus_phases=["MD"],
                    landmark_labels=["APC", "ASC", "PSC"],
                    export_keys=[
                        'mid-diastolic-images',
                        'mid-diastolic-annulus',
                        'mid-diastolic-APC',
                        'mid-diastolic-ASC',
                        'mid-diastolic-PSC'
                    ]
                ),
        }

    def init_strategies(self):
        return {}

    def train(self, request):
        return {}

