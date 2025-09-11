from __future__ import annotations

from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple

import albumentations as alb
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from PIL import ImageOps

from .nougat_parser.constants import IMAGENET_DEFAULT_MEAN
from .nougat_parser.constants import IMAGENET_DEFAULT_STD

from torchvision.transforms.functional import resize
from torchvision.transforms.functional import rotate

def alb_wrapper_sc(transform) -> Callable[[Image.Image], torch.Tensor]:
    """
    Albumations pipeline wrapper self-contained
    """

    def f(im):
        return transform(image=np.asarray(im))['image']

    return f

# test Transformation
test_transform_sc = alb_wrapper_sc(
    alb.Compose(
        [
            alb.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ToTensorV2(),
        ]
    )
)


def to_tensor_sc(training_flag: bool):
    """
    Wrapper to convert image to tensor for Nougat inference.
    """
    if training_flag:
        raise NotImplementedError(
            'The AdaParse pipeline is designed for inference at scale - not training of Nougat.\n'
            'Fine-tune Nougat weights within *their framework* and replace weights path in `parser_settings.checkpoint`.\n'
            'Reference: https://github.com/facebookresearch/nougat?tab=readme-ov-file#training'
        )
    else:
        return test_transform_sc


def crop_margin_sc_without_cv2(img: Image.Image) -> Image.Image:
    """
    Self-contained crop margin (no OpenCV); outside of Swin Encoder.
    Mirrors:
        gray = 255 * (data < 200)
        # replaced lines:
        #coords = cv2.findNonZero(gray)
        #a, b, w, h = cv2.boundingRect(coords)
        return img.crop((a, b, a + w, b + h))

    # confirmed equivalent to self.encoder.crop_margin()
    """
    # grayscale to uint8
    data = np.array(img.convert('L'), dtype=np.uint8)
    max_val = int(data.max())
    min_val = int(data.min())

    # uniform image → nothing to crop
    if max_val == min_val:
        return img

    # normalize to [0, 255] like the original
    data = ((data.astype(np.float32) - min_val) / (max_val - min_val) * 255.0).astype(
        np.uint8
    )

    # foreground mask (True where "text" is)
    # original makes gray = 255 * (data < 200)
    mask = data < 200

    # no foreground → return original image
    if not mask.any():
        return img

    # Equivalent of findNonZero + boundingRect without constructing coords
    ys, xs = np.nonzero(mask)  # y=row, x=col
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())

    # width/height in OpenCV are inclusive of both ends → +1
    w = x_max - x_min + 1
    h = y_max - y_min + 1

    # PIL crop box is (left, upper, right, lower) and right/lower are exclusive
    left, upper, right, lower = x_min, y_min, x_min + w, y_min + h
    return img.crop((left, upper, right, lower))


def prepare_input_sc(
    img: Image.Image,
    prep_args: Tuple[bool, List[int], bool],
) -> Optional[torch.Tensor]:
    """Standalone CPU-only `prepare_input()` that augments Nougat code

    Standalone implementation of `Nougat.encoder.prepare_input()` method that no longer inherits from nn.Module.
    Allows easy handling of dataloader since CPU-only.
    """
    # disentagle inputs (bool, list[int], bool)
    align_long_axis, input_size, random_padding = prep_args

    if img is None:
        return None

    try:
        img = crop_margin_sc_without_cv2(img.convert('RGB'))
    except OSError:
        # might throw an error for broken files
        return
    if img.height == 0 or img.width == 0:
        return
    # formerly: self.align_long_axis, self.input_size
    if align_long_axis and (  # self.align_long_axis, attr
        (input_size[0] > input_size[1] and img.width > img.height)
        or (input_size[0] < input_size[1] and img.width < img.height)
    ):
        img = rotate(img, angle=-90, expand=True)
    img = resize(img, min(input_size))
    img.thumbnail((input_size[1], input_size[0]))
    delta_width = input_size[1] - img.width
    delta_height = input_size[0] - img.height

    if random_padding:
        pad_width = np.random.randint(low=0, high=delta_width + 1)
        pad_height = np.random.randint(low=0, high=delta_height + 1)
    else:
        pad_width = delta_width // 2
        pad_height = delta_height // 2
    padding = (
        pad_width,
        pad_height,
        delta_width - pad_width,
        delta_height - pad_height,
    )
    transform_sc = to_tensor_sc(training_flag=False)
    return transform_sc(ImageOps.expand(img, padding))
