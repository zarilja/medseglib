# MEDSEGLIB

This package designed for segmentation of oncology by CT and MRI images.

### Models
Currently, the models DeepLabV3(bb: resnet50), DeepLabV3(bb: resnet101) are used.
During the training, these models were used, as well as UNet and TransUnet, but DeepLabV3 showed the best results (dice and iou).

### Supported types of research
* Brain
* Liver

### How to use
```python
import numpy as np
from megseglib.sdk import Model

ct = np.zeros((1, 3, 512, 512), dtype=np.single)
model = Model(model_name="deeplab_v3_resnet101", model_type="liver")

result = model.forward(ct)
# or
result = model(ct)

print(type(result), result.shape) # <class 'numpy.ndarray'> (1, 1, 512, 512) - segmentation mask 
```