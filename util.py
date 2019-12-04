import os, sys
from PIL import Image
import numpy as np
import torch


def img_to_tensor(imgs):
    return torch.tensor(np.array(imgs))
