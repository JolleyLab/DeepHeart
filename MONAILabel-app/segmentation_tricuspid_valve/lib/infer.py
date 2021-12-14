from monai.inferers import SimpleInferer
from monai.engines.utils import CommonKeys as Keys
import torch
from abc import abstractmethod

from monai.transforms import (
    LoadImaged,
    ToTensord,
    ScaleIntensityd,
    AsDiscreted,
    ConcatItemsd,
    ToNumpyd,
    SqueezeDimd,
    Activationsd,
    SplitChanneld,
    KeepLargestConnectedComponentd
)

from monailabel.transform.post import Restored
from monailabel.interfaces.tasks.infer import InferTask, InferType

from .transforms import DistanceTransformd, MergeLabelsd


class TricuspidInference(InferTask):
    """
    This provides Inference Engine for pre-trained tricuspid valve segmentation (VNet) model.
    """


    def __init__(
        self,
        path,
        network=None,
        type="DeepHeartSegmentation",
        labels=("tricuspid anterior leaflet",
                "tricuspid posterior leaflet",
                "tricuspid septal leaflet"), # NB: label names correspond to SlicerHeart terminology
        dimension=3,
        description="A pre-trained model for volumetric (3D) segmentation of tricuspid valve from 3DE image",
    ):
        config = self.get_config()

        super().__init__(
            path=path,
            network=network,
            type=type,
            labels=labels,
            dimension=dimension,
            description=description,
            config=config
        )

    def info(self):
        return {
            "type": self.type,
            "labels": self.labels,
            "dimension": self.dimension,
            "description": self.description,
            "config": self.config(),
            "valve_type": self.getValveType(),
        }

    def get_config(self):
        return {
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "model_attributes": {
                "valve_type": self.getValveType(),
                "phases": self.getCardiacPhases(),
                "landmark_labels": self.getLandmarkLabels(),
                "volume_dimensions": self.getVolumeDimensions(),
                "voxel_spacing": self.getVoxelSpacing()
            },
            "export_keys": self.getExportKeys()
        }

    def getValveType(self):
        return "tricuspid"

    def getVolumeDimensions(self):
        return [224] * 3

    def getVoxelSpacing(self):
        return 0.25

    @abstractmethod
    def getCardiacPhases(self):
        pass

    @abstractmethod
    def getLandmarkLabels(self):
        return None

    @abstractmethod
    def getExportKeys(self):
        pass

    def inferer(self):
        return SimpleInferer()

    def post_transforms(self):
        return [
            Activationsd(keys=Keys.PRED, softmax=True),
            AsDiscreted(keys=Keys.PRED, argmax=True),
            KeepLargestConnectedComponentd(keys=Keys.PRED, applied_labels=[1, 2, 3]),
            SqueezeDimd(keys=Keys.PRED, dim=0),
            ToNumpyd(keys=Keys.PRED),
            Restored(keys=Keys.PRED, ref_image=Keys.IMAGE),
        ]


class TricuspidInferenceTaskSinglePhaseAnn(TricuspidInference):
    """ This model requires input of:
      - mid systolic frame
      - annulus label
    """

    def getValveType(self):
        return "tricuspid"

    def getCardiacPhases(self):
        return ["MS"]

    def getExportKeys(self):
        return [
            'mid-systolic-images',
            'mid-systolic-annulus'
        ]

    def pre_transforms(self):
        all_keys = ["image_ms", "image_annulus"]
        return [
            LoadImaged(keys=Keys.IMAGE, reader="NibabelReader"),
            SplitChanneld(keys=Keys.IMAGE, output_postfixes=["ms", "annulus"], channel_dim=0),
            DistanceTransformd(keys=["image_annulus"]),
            ScaleIntensityd(
                keys=["image_ms"]
            ),
            ToTensord(keys=all_keys),
            ConcatItemsd(keys=all_keys, name=Keys.IMAGE, dim=0)
        ]


class TricuspidInferenceTaskTwoPhaseAnn(TricuspidInference):
    """ This model requires input of:
      - mid systolic frame
      - mid diastolic frame
      - annulus label
    """

    def getCardiacPhases(self):
        return ["MS", "MD"]

    def getExportKeys(self):
        return [
            'mid-systolic-images',
            'mid-diastolic-images',
            'mid-systolic-annulus'
        ]

    def pre_transforms(self):
        all_keys = ["image_ms", "image_md", "image_annulus"]
        return [
            LoadImaged(keys=Keys.IMAGE, reader="NibabelReader"),
            SplitChanneld(keys=Keys.IMAGE, output_postfixes=["ms", "md", "annulus"], channel_dim=0),
            DistanceTransformd(keys=["image_annulus"]),
            ScaleIntensityd(
                keys=["image_ms", "image_md"]
            ),
            ToTensord(keys=all_keys),
            ConcatItemsd(keys=all_keys, name=Keys.IMAGE, dim=0)
        ]


class TricuspidInferenceTaskSinglePhaseAnnCom(TricuspidInference):
    """ This model requires input of:
      - mid systolic frame
      - annulus label
      - commissure labels: "APC", "ASC", "PSC"
    """

    def getCardiacPhases(self):
        return ["MS"]

    def getLandmarkLabels(self):
        return ["APC", "ASC", "PSC"]

    def getExportKeys(self):
        return [
            'mid-systolic-images',
            'mid-systolic-annulus',
            'mid-systolic-APC',
            'mid-systolic-ASC',
            'mid-systolic-PSC'
        ]

    def pre_transforms(self):
        all_keys = ["image_ms", "image_annulus", "image_apc", "image_asc", "image_psc"]
        return [
            LoadImaged(keys=Keys.IMAGE, reader="NibabelReader"),
            SplitChanneld(keys=Keys.IMAGE, output_postfixes=["ms", "annulus", "apc", "asc", "psc"], channel_dim=0),
            DistanceTransformd(keys=["image_annulus", "image_apc", "image_asc", "image_psc"]),
            ScaleIntensityd(
                keys=["image_ms"],
                minv=0.0,
                maxv=1.0
            ),
            ToTensord(keys=all_keys),
            ConcatItemsd(keys=all_keys, name=Keys.IMAGE, dim=0)
        ]


class TricuspidInferenceTaskTwoPhaseAnnCom(TricuspidInference):
    """ This model requires input of:
      - mid systolic frame
      - mid diastolic frame
      - annulus label
      - commissures labels: "APC", "ASC", "PSC"
    """

    def getCardiacPhases(self):
        return ["MS", "MD"]

    def getLandmarkLabels(self):
        return ["APC", "ASC", "PSC"]

    def getExportKeys(self):
        return [
            'mid-systolic-images',
            'mid-diastolic-images',
            'mid-systolic-annulus',
            'mid-systolic-APC',
            'mid-systolic-ASC',
            'mid-systolic-PSC'
        ]

    def pre_transforms(self):
        all_keys = ["image_ms", "image_md", "image_annulus", "image_apc", "image_asc", "image_psc"]
        return [
            LoadImaged(keys=Keys.IMAGE, reader="NibabelReader"),
            SplitChanneld(keys=Keys.IMAGE, output_postfixes=["ms", "md", "annulus", "apc", "asc", "psc"], channel_dim=0),
            DistanceTransformd(keys=["image_annulus", "image_apc", "image_asc", "image_psc"]),
            ScaleIntensityd(
                keys=["image_ms", "image_md"]
            ),
            ToTensord(keys=all_keys),
            ConcatItemsd(keys=all_keys, name=Keys.IMAGE, dim=0)
        ]


class TricuspidInferenceTaskTwoPhaseAnnComOneLabel(TricuspidInferenceTaskTwoPhaseAnnCom):
    """ This model requires input of:
      - mid systolic frame
      - mid diastolic frame
      - annulus label
      - commissures labels: "APC", "ASC", "PSC"
    """

    def pre_transforms(self):
        all_keys = ["image_ms", "image_md", "image_annulus", "image_commissures"]
        return [
            LoadImaged(keys=Keys.IMAGE, reader="NibabelReader"),
            SplitChanneld(keys=Keys.IMAGE, output_postfixes=["ms", "md", "annulus", "apc", "asc", "psc"], channel_dim=0),
            MergeLabelsd(keys=["image_apc", "image_asc", "image_psc"], name="image_commissures"),
            DistanceTransformd(keys=["image_annulus", "image_commissures"]),
            ScaleIntensityd(
                keys=["image_ms", "image_md"]
            ),
            ToTensord(keys=all_keys),
            ConcatItemsd(keys=all_keys, name=Keys.IMAGE, dim=0)
        ]

