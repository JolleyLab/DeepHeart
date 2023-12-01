from monai.inferers import SimpleInferer
from monai.engines.utils import CommonKeys as Keys
import torch

from monai.transforms import (
    LoadImaged,
    EnsureTyped,
    ToTensord,
    ScaleIntensityd,
    AsDiscreted,
    ConcatItemsd,
    ToNumpyd,
    SqueezeDimd,
    Activationsd,
    SplitChanneld,
    KeepLargestConnectedComponentd,
    NormalizeIntensityd
)

from monailabel.transform.post import Restored
from monailabel.interfaces.tasks.infer import InferTask

from .transforms import DistanceTransformd


class SlicerHeartInference(InferTask):
    """
    This provides Inference Engine for pre-trained tricuspid valve segmentation (VNet) model.
    """

    def __init__(
        self,
        path,
        network=None,
        type="DeepHeartSegmentation",
        labels=None, # NB: label names correspond to SlicerHeart terminology
        dimension=3,
        description="A pre-trained model for volumetric (3D) segmentation of SlicerHeart valve from 3DE image",
        spatial_size=(224,224,224),
        voxel_spacing=0.25,
        landmark_label_phases=None,
        annulus_phases=None,
        valve_type=None,
        cardiac_phase_frames=None,
        landmark_labels=None,
        export_keys=None
    ):
        self._spatial_size = spatial_size
        self._voxel_spacing = voxel_spacing

        if landmark_label_phases is None:
            landmark_label_phases = ["MS"]
        self._landmark_label_phases = landmark_label_phases

        if annulus_phases is None:
            annulus_phases = ["MS"]
        self._annulus_phases = annulus_phases

        self._valve_type = valve_type

        self._cardiac_phase_frames = cardiac_phase_frames
        self._landmark_labels = landmark_labels

        # NB: SlicerHeart ExportHeartData export keys
        self._export_keys = export_keys

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
            "valve_type": self._valve_type,
        }

    def get_config(self):
        return {
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "model_attributes": {
                "valve_type": self._valve_type,
                "cardiac_phase_frames": self._cardiac_phase_frames,
                "landmark_labels": self._landmark_labels,
                "landmark_label_phases": self._landmark_label_phases,
                "annulus_phases": self._annulus_phases,
                "volume_dimensions": self._spatial_size,
                "voxel_spacing": self._voxel_spacing
            },
            "export_keys": self._export_keys
        }

    def inferer(self, data=None):
        return SimpleInferer()

    def post_transforms(self, data=None):
        return [
            Activationsd(keys=Keys.PRED, softmax=True),
            AsDiscreted(keys=Keys.PRED, argmax=True),
            KeepLargestConnectedComponentd(keys=Keys.PRED, applied_labels=[1, 2, 3]), # TODO: change to len labels
            SqueezeDimd(keys=Keys.PRED, dim=0),
            ToNumpyd(keys=Keys.PRED),
            Restored(keys=Keys.PRED, ref_image=Keys.IMAGE),
        ]


class SlicerHeartInferenceTaskSinglePhaseAnn(SlicerHeartInference):
    """ This model requires input of:
      - mid systolic frame
      - annulus label
    """

    def pre_transforms(self, data=None):
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


class TricuspidInferenceTaskSinglePhaseAnnCom(SlicerHeartInference):
    """ This model requires input of:
      - mid systolic frame
      - annulus label
      - commissure labels: "APC", "ASC", "PSC"
    """

    def pre_transforms(self, data=None):
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


class TricuspidInferenceTaskTwoPhaseAnnCom(SlicerHeartInference):
    """ This model requires input of:
      - mid systolic frame
      - mid diastolic frame
      - annulus label
      - commissures labels: "APC", "ASC", "PSC"
    """

    def pre_transforms(self, data=None):
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


class LavvInferenceTaskSinglePhaseAnnCom(SlicerHeartInference):
    """ This model requires input of:
      - mid systolic frame
      - annulus label
      - commissure labels: "APC", "ASC", "PSC"
    """

    def pre_transforms(self, data=None):
        all_keys = ["image_ms", "image_annulus", "image_sic", "image_alc", "image_pmc"]
        return [
            LoadImaged(keys=Keys.IMAGE, reader="NibabelReader"),
            SplitChanneld(keys=Keys.IMAGE, output_postfixes=["ms", "annulus", "sic", "alc", "pmc"], channel_dim=0),
            DistanceTransformd(keys=["image_annulus", "image_sic", "image_alc", "image_pmc"]),
            ScaleIntensityd(
                keys=["image_ms"],
                minv=0.0,
                maxv=1.0
            ),
            ToTensord(keys=all_keys),
            ConcatItemsd(keys=all_keys, name=Keys.IMAGE, dim=0)
        ]


class LavvInferenceTaskTwoPhaseAnnCom(SlicerHeartInference):
    """ This model requires input of:
      - mid systolic frame
      - mid diastolic frame
      - annulus label
      - commissures labels: "APC", "ASC", "PSC"
    """

    def pre_transforms(self, data=None):
        all_keys = ["image_ms", "image_md", "image_annulus", "image_sic", "image_alc", "image_pmc"]
        return [
            LoadImaged(keys=Keys.IMAGE, reader="NibabelReader"),
            SplitChanneld(keys=Keys.IMAGE, output_postfixes=["ms", "md", "annulus", "sic", "alc", "pmc"], channel_dim=0),
            DistanceTransformd(keys=["image_annulus", "image_sic", "image_alc", "image_pmc"]),
            ScaleIntensityd(
                keys=["image_ms", "image_md"]
            ),
            ToTensord(keys=all_keys),
            ConcatItemsd(keys=all_keys, name=Keys.IMAGE, dim=0)
        ]


class TricuspidDiastolicInferenceTaskSinglePhaseAnnCom(SlicerHeartInference):
    """ This model requires input of:
      - mid systolic frame
      - annulus label
      - commissure labels: "APC", "ASC", "PSC"
    """

    def pre_transforms(self, data=None):
        all_keys = ["image_md", "image_annulus", "image_apc", "image_asc", "image_psc"]
        return [
            LoadImaged(keys=Keys.IMAGE, reader="NibabelReader"),
            SplitChanneld(keys=Keys.IMAGE, output_postfixes=["md", "annulus", "apc", "asc", "psc"], channel_dim=0),
            DistanceTransformd(keys=all_keys[1:]),
            NormalizeIntensityd(keys="image_md", nonzero=True),
            ScaleIntensityd(
                keys=all_keys,
                minv=0.0,
                maxv=1.0
            ),
            ToTensord(keys=all_keys),
            ConcatItemsd(keys=all_keys, name=Keys.IMAGE, dim=0),
            EnsureTyped(keys=Keys.IMAGE)
        ]
