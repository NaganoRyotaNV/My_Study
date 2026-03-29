import json
import sys

import cv2
import numpy as np
import ot
import PIL
import scipy
import sklearn
import torch
import torchvision
import yaml

info = {
    "python": sys.version,
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_count": torch.cuda.device_count(),
    "numpy": np.__version__,
    "scipy": scipy.__version__,
    "sklearn": sklearn.__version__,
    "opencv": cv2.__version__,
    "pillow": PIL.__version__,
    "pyyaml": yaml.__version__,
    "pot": ot.__version__,
}

print(json.dumps(info, indent=2, ensure_ascii=False))