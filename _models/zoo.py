from abc import ABC, abstractmethod

import torchvision

from _models.utils import download_weights_to_model


class AbstractModel(ABC):
    @abstractmethod
    def forward(self, x):
        raise NotImplementedError

    def __call__(self, x):
        self.forward()

    @abstractmethod
    def to(self, device):
        pass


class DeepLabV3(AbstractModel):
    def __init__(self, backbone, weights_url, weights_hash):
        super().__init__()

        if backbone == "resnet50":
            self.model = torchvision.models.segmentation.deeplabv3_resnet50(
                progress=False, num_classes=1, weights_backbone=None,
            )
        elif backbone == "resnet101":
            self.model = torchvision.models.segmentation.deeplabv3_resnet101(
                progress=False, num_classes=1, weights_backbone=None,
            )
        else:
            raise RuntimeError(f"unknown DeepLabV3 backbone - {backbone}")

        download_weights_to_model(model=self.model, url=weights_url, hash_=weights_hash)
        self.model.eval()

    def to(self, device):
        self.model = self.model.to(device)
        return self

    def forward(self, x):
        return self.model.forward(x)['out']
