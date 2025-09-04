import os
from PIL import Image, ImageOps
import torch
import cv2
from pathlib import Path
import numpy as np
from typing import Optional, List
import argparse
from collections import defaultdict

import albumentations as alb
from albumentations.pytorch import ToTensorV2

from transformers import VisionEncoderDecoderModel, NougatProcessor
from transformers import PreTrainedTokenizerFast
from transformers import StoppingCriteria, StoppingCriteriaList
from transformers.file_utils import ModelOutput

from torchvision.transforms.functional import resize, rotate

from postprocessing import postprocess

from device_utils import move_to_custom_device, resolve_dtype

# --------------------------------------------------------
# timm-0.5.4
# timm/data/constants.py
# --------------------------------------------------------
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

def crop_margin(img: Image.Image) -> Image.Image:
        """
        Crop
        """
        data = np.array(img.convert("L"))
        data = data.astype(np.uint8)
        max_val = data.max()
        min_val = data.min()
        if max_val == min_val:
            return img
        data = (data - min_val) / (max_val - min_val) * 255
        gray = 255 * (data < 200).astype(np.uint8)

        coords = cv2.findNonZero(gray)  # Find all non-zero points (text)
        a, b, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box

        return img.crop((a, b, w + a, h + b))

def prepare_input(img: Image.Image,
                  input_size:List[int] = [896, 672],
                  align_long_axis:bool=False,
                  random_padding: bool = False) -> torch.Tensor:
        """
        Convert PIL Image to tensor according to specified input_size after following steps below:
            - resize
            - rotate (if align_long_axis is True and image is not aligned longer axis with canvas)
            - pad
        """
        if img is None:
            print('img is None -> None')
            return
        # crop margins
        try:
            img = crop_margin(img.convert("RGB"))
        except OSError:
            # might throw an error for broken files
            print('OSError -> None')
            return
        if img.height == 0 or img.width == 0:
            return
        if align_long_axis and (
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

        # check
        print(f'type(img) : {type(img)}')
        print(f'img.size : {img.size}')

        transform = to_tensor()
        return transform(ImageOps.expand(img, padding))

def alb_wrapper(transform):
    def f(im):
        return transform(image=np.asarray(im))["image"]

    return f

test_transform = alb_wrapper(
    alb.Compose(
        [
            alb.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ToTensorV2(),
        ]
    )
)

def to_tensor():
    return test_transform

# Running VarTorch
class RunningVarTorch:
    def __init__(self, L=15, norm=False):
        self.values = None
        self.L = L
        self.norm = norm

    def push(self, x: torch.Tensor):
        assert x.dim() == 1
        if self.values is None:
            self.values = x[:, None]
        elif self.values.shape[1] < self.L:
            self.values = torch.cat((self.values, x[:, None]), 1)
        else:
            self.values = torch.cat((self.values[:, 1:], x[:, None]), 1)

    def variance(self):
        if self.values is None:
            return
        if self.norm:
            return torch.var(self.values, 1) / self.values.shape[1]
        else:
            return torch.var(self.values, 1)

# class RunningVarTorch
class StoppingCriteriaScores(StoppingCriteria):
    def __init__(self, threshold: float = 0.015, window_size: int = 200):
        super().__init__()
        self.threshold = threshold
        self.vars = RunningVarTorch(norm=True)
        self.varvars = RunningVarTorch(L=window_size)
        self.stop_inds = defaultdict(int)
        self.stopped = defaultdict(bool)
        self.size = 0
        self.window_size = window_size

    @torch.no_grad()
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        last_scores = scores[-1]
        self.vars.push(last_scores.max(1)[0].float().cpu())
        self.varvars.push(self.vars.variance())
        self.size += 1
        if self.size < self.window_size:
            return False

        varvar = self.varvars.variance()
        for b in range(len(last_scores)):
            if varvar[b] < self.threshold:
                if self.stop_inds[b] > 0 and not self.stopped[b]:
                    self.stopped[b] = self.stop_inds[b] >= self.size
                else:
                    self.stop_inds[b] = int(
                        min(max(self.size, 1) * 1.15 + 150 + self.window_size, 4095)
                    )
            else:
                self.stop_inds[b] = 0
                self.stopped[b] = False
        return all(self.stopped.values()) and len(self.stopped) > 0

# helpers
def batch(l, b=15):
    subs = []
    for i in range(len(l) - b):
        subs.append(l[i : i + b])
    return subs


def subdiv(l, b=10):
    subs = []
    for i in range(len(l) - b):
        subs.append(l[: i + b])
    return subs


# ========= Your CLI script, now calling the above =========

def main(args):
    # load model/processor
    model_dir = '/home/siebenschuh/AdaParse/models/facebook__nougat-base'
    processor = NougatProcessor.from_pretrained(model_dir, use_fast=True, local_files_only=True)
    model = VisionEncoderDecoderModel.from_pretrained(model_dir, local_files_only=True)

    model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
    model.config.pad_token_id           = processor.tokenizer.pad_token_id
    model.config.eos_token_id           = processor.tokenizer.eos_token_id

    # tokenizer
    tokenizer_file_path = '/lus/flare/projects/FoundEpidem/siebenschuh/adaparse_data/meta/nougat/checkpoint/tokenizer.json'
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_file_path))

    # device management
    model = move_to_custom_device(model)

    # Debug: confirm HF pre-processing matches expectations
    image = Image.open(args.image_path).convert('RGB')

    #inputs = processor(images=image, return_tensors="pt").to(device)
    inputs = processor(images=image, return_tensors="pt").to(device=model.device, dtype=model.dtype)

    encoded = processor(images=image, return_tensors="pt")
    pixel_values = encoded.pixel_values

    # If processor returned uint8, make it float
    if pixel_values.dtype == torch.uint8:
        pixel_values = pixel_values.float()

    # DEBUG
    print(f'pixel_values.size() : {pixel_values.size()}')

    # If values still look like 0..255, rescale & normalize like Nougat expects
    ip = processor.image_processor
    do_rescale   = getattr(ip, "do_rescale", True)
    do_normalize = getattr(ip, "do_normalize", True)

    # Rescale only if it looks un-rescaled
    if do_rescale and pixel_values.max() > 1.5:
        pixel_values = pixel_values / 255.0

    if do_normalize:
        mean = torch.tensor(getattr(ip, "image_mean", IMAGENET_DEFAULT_MEAN)).view(1, 3, 1, 1)
        std  = torch.tensor(getattr(ip, "image_std",  IMAGENET_DEFAULT_STD)).view(1, 3, 1, 1)
        pixel_values = (pixel_values - mean) / std

    # Final device/dtype cast
    pixel_values = pixel_values.to(device=model.device, dtype=model.dtype, non_blocking=False)

    # Sanity check (remove later)
    #print("pixel_values:", pixel_values.dtype, float(pixel_values.min()), float(pixel_values.max()))
    assert pixel_values.is_floating_point(), "pixel_values must be float"

    # output
    output = {
            "predictions": list(),
            "sequences": list(),
            "repeats": list(),
            "repetitions": list(),
        }

    # generate (full model, not decoder)
    decoder_output = model.generate(
        pixel_values=pixel_values,
        return_dict_in_generate=True,
        output_scores=True,
        do_sample=False,
        max_new_tokens=1024,              # or use max_length if you prefer
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        stopping_criteria=StoppingCriteriaList([StoppingCriteriaScores()])  # if you want your early stopping
    )

    # VARS
    early_stopping = True

    output["repetitions"] = decoder_output.sequences.clone()
    output["sequences"] = decoder_output.sequences.clone()
    batch_size = len(decoder_output.sequences)

    logits = torch.stack(decoder_output.scores, 1).cpu().max(-1)
    values = logits.values
    indices = logits.indices

    for b in range(batch_size):
        mask = indices[b] != tokenizer.pad_token_id
        N = mask.sum().item()
        var = np.array(
            [np.var(s) / len(s) for s in batch(values[b, mask].float().numpy())]
        )
        if len(var) < 10:
            output["repeats"].append(None)
            continue
        varvar = np.array([np.var(v) for v in subdiv(var[::-1])][::-1])
        minlen = 120
        #if (indices[b] == tokenizer.eos_token_id).any() and (N + 1 < indices.shape[1]):
        if (indices[b] == tokenizer.eos_token_id) and (N + 1 < indices.shape[1]):
            # there is an end to the generation, likely no repetitions
            output["repeats"].append(None)
            continue
        small_var = np.where(varvar < 0.045)[0]
        if early_stopping and len(small_var) > 1:
            if np.all(np.diff(small_var) < 2):
                idx = int(min(max(small_var[0], 1) * 1.08 + minlen, 4095))
                if idx / N > 0.9:  # at most last bit
                    output["repeats"].append(None)
                    continue
                elif small_var[0] < 30:
                    idx = 0
                #logging.warn("Found repetitions in sample %i" % b)
                output["repeats"].append(idx)
                output["sequences"][b, idx:] = tokenizer.pad_token_id
                output["repetitions"][b, :idx] = tokenizer.pad_token_id
            else:
                output["repeats"].append(None)
        else:
            output["repeats"].append(None)
    output["repetitions"] = tokenizer.batch_decode(
        output["repetitions"], skip_special_tokens=True
    )
    output["predictions"] = postprocess(
        tokenizer.batch_decode(
            output["sequences"], skip_special_tokens=True
        ),
        markdown_fix=False,
    )

    return

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Extract text from PDF images using Nougat')
    parser.add_argument('image_path', help='Path to the image file (e.g., ./data/page_1.png)')
    parser.add_argument('--output', '-o', help='Output file to save the extracted text (optional)')
    parser.add_argument('--dtype', '-d', default='bfloat16', help='Torch dtype (e.g. float32, bfloat16)')
    args = parser.parse_args()
    main(args)
