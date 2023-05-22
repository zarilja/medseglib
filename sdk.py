import torch.cuda
import numpy as np

from _models.zoo import AbstractModel, DeepLabV3


MODEL_NAMES = ("deeplab_v3_resnet50", "deeplab_v3_resnet101")
MODEL_TYPES = ("brain", "liver")


SETTINGS = {
    "deeplab_v3_resnet50:brain": (
        DeepLabV3,
        {
            "backbone": "resnet50",
            "weights_url": "https://getfile.dokpub.com/yandex/get/https://disk.yandex.ru/d/oASL_tgLzJEjGg",
            "weights_hash": "521055c34e714597159fb59d2bfa9659920d29d7fc7ee6ec6b8014c21d5d21b1",
        }
    ),
    "deeplab_v3_resnet50:liver": (
        DeepLabV3,
        {
            "backbone": "resnet50",
            "weights_url": "https://getfile.dokpub.com/yandex/get/https://disk.yandex.ru/d/uSjraehaEi8raA",
            "weights_hash": "e0e44653869995b8ee5b0055c692d26d4702cb89399c16e61bbe356613c1a4e5",
        }
    ),
    "deeplab_v3_resnet101:brain": (
        DeepLabV3,
        {
            "backbone": "resnet101",
            "weights_url": "https://getfile.dokpub.com/yandex/get/https://disk.yandex.ru/d/O1WZ4EUJsjM1XQ",
            "weights_hash": "6d0328b604d1a0ee4a21f2baeda195a27260ae69f998391acd5da44954a0775f",
        }
    ),
    "deeplab_v3_resnet101:liver": (
        DeepLabV3,
        {
            "backbone": "resnet101",
            "weights_url": "https://getfile.dokpub.com/yandex/get/https://disk.yandex.ru/d/r98QV2PLXBEgEw",
            "weights_hash": "dd71ef48baf71896cf490460f5a81ef5a828f1ac78c34be888bf1a53a01ddef9",
        }
    )
}


class Model:
    def __init__(self, model_name, model_type):
        if model_name not in MODEL_NAMES:
            raise NotImplementedError(f"model {model_name} not implemented")

        if model_type not in MODEL_TYPES:
            raise NotImplementedError(f"model {model_name} not implemented")

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model_cls, kwargs = SETTINGS[f"{model_name}:{model_type}"]
        self.model: AbstractModel = model_cls(**kwargs)
        self.model = self.model.to(self.device)

    @staticmethod
    def _preprocess(x: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(x)
        x[x > 1200] = 1200
        x = (x + 1000) / (1200 + 1000)
        x[x < 0] = 0

        return x

    @staticmethod
    def _postprocess(x):
        return torch.where(x > 0.5, 1.0, 0.0)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = self._preprocess(x)
        x = x.to(self.device)
        y = self.model.forward(x)
        return self._postprocess(y).cpu().numpy()

    def __call__(self, x):
        return self.forward(x)
