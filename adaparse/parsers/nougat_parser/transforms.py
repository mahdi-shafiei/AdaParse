"""
Upgrade of albumentations for inference-only Nougat build

Source:
https://github.com/facebookresearch/nougat/blob/main/nougat/transforms.py
"""
# Implements image augmentation

import albumentations as alb
from albumentations.pytorch import ToTensorV2
import numpy as np
from typing import Callable

from adaparse.parsers.nougat_parser.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

def alb_wrapper(transform: alb.Compose) -> Callable:
    """Wrap an Albumentations transform so it accepts a PIL.Image and returns the 'image' field."""
    def f(im):
        return transform(image=np.asarray(im))["image"]
    return f

# training of Nougat disabled in AdaParse (inference-only build)
train_transform = None

# test-time transform (normalization)
test_transform = alb_wrapper(
    alb.Compose(
        [
            alb.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ToTensorV2(),
        ]
    )
)
