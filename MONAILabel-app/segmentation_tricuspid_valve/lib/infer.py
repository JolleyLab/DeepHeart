from monai.inferers import SimpleInferer
from monai.engines.utils import CommonKeys as Keys

from monai.transforms import (
    AddChanneld,
    LoadImaged,
    ToTensord,
    ScaleIntensityd,
    AsDiscreted,
    ConcatItemsd,
    ToNumpyd,
    SqueezeDimd
)

from monailabel.utils.others.post import Restored
from monailabel.interfaces.tasks import InferTask, InferType

from .transforms import DistanceTransformd


class MyInfer(InferTask):
    """
    This provides Inference Engine for pre-trained tricuspid valve segmentation (VNet) model.
    """

    def __init__(
        self,
        path,
        network=None,
        type=InferType.SEGMENTATION,
        labels=("anterior", "posterior", "septal"),
        dimension=3,
        description="A pre-trained model for volumetric (3D) segmentation of tricuspid valve from 3DE image",
    ):
        super().__init__(
            path=path,
            network=network,
            type=type,
            labels=labels,
            dimension=dimension,
            description=description,
        )

    def pre_transforms(self):
        all_keys = [Keys.IMAGE, Keys.LABEL]
        return [
            LoadImaged(keys=all_keys, reader="NibabelReader"),
            AddChanneld(keys=all_keys),
            DistanceTransformd(keys=[Keys.LABEL]),
            ScaleIntensityd(
                keys=[Keys.IMAGE],
                minv=0.0,
                maxv=1.0
            ),
            ToTensord(keys=all_keys),
            ConcatItemsd(keys=all_keys, name=Keys.IMAGE, dim=0)
        ]

    def inferer(self):
        return SimpleInferer()

    def post_transforms(self):
        return [
            AddChanneld(keys="pred"),
            AsDiscreted(keys="pred", argmax=True),
            SqueezeDimd(keys="pred", dim=0),
            ToNumpyd(keys="pred"),
            Restored(keys="pred", ref_image="image"),
        ]