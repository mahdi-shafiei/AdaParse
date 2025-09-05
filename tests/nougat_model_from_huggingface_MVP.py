import os

# BLOCK WEB REQUESTS
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1") # crucial
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# torch on aurora
import torch
if torch.xpu.is_available():
    import intel_extension_for_pytorch as ipex

import cv2
from PIL import Image, ImageOps

from pathlib import Path
import numpy as np
from typing import Optional, List, Dict, Any, Iterable
import argparse
from collections import defaultdict
from functools import partial
from contextlib import ExitStack, nullcontext

import albumentations as alb
from albumentations.pytorch import ToTensorV2

from transformers import VisionEncoderDecoderModel, NougatProcessor
from transformers import PreTrainedTokenizerFast
from transformers import StoppingCriteria, StoppingCriteriaList

from torchvision.transforms.functional import resize, rotate

from adaparse.parsers.nougat_parser.postprocessing import postprocess
from adaparse.device_utils import move_to_custom_device
from adaparse.parsers.nougat_inference_utils import prepare_input_sc

# Nougat-specific imports
#from nougat.utils.dataset import LazyDataset # legacy

from adaparse.parsers.nougat_parser.utils.dataset import LazyDataset
from adaparse.parsers.nougat_parser.postprocessing import markdown_compatible
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader

# -----------------------------------
# timm-0.5.4
# timm/data/constants.py
# -----------------------------------
from adaparse.parsers.nougat_parser.legacy_timm.data.constants import IMAGENET_DEFAULT_MEAN
from adaparse.parsers.nougat_parser.legacy_timm.data.constants import IMAGENET_DEFAULT_STD


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# --- add near imports ---
from pathlib import Path
from typing import Iterable

# --- tiny helpers ---
IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

def iter_images(path: str | Path) -> list[Path]:
    p = Path(path)
    if p.is_file():
        return [p]
    if p.is_dir():
        files = []
        for ext in IMG_EXTS:
            files += sorted(p.rglob(f"*{ext}"))
        return files
    raise FileNotFoundError(f"{p} not found")

def chunked(seq: list, n: int) -> Iterable[list]:
    for i in range(0, len(seq), n):
        yield seq[i:i+n]
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# ARGS
def process_decoder_output(decoder_output,
                           tokenizer,
                           early_stopping:bool=True,
                           min_len:int=120,
                           repeat_threshold:int=10,
                           variance_threshold:int=0.045,
                           small_var_threshold:int=30,
                           variance_mult:int=1.08,
                           max_id:int=4095):
    """
    Helper to post-process generated output
    """

    # output
    output = {
            "predictions": list(),
            "sequences": list(),
            "repeats": list(),
            "repetitions": list(),
        }

    output["repetitions"] = decoder_output.sequences.clone()
    output["sequences"] = decoder_output.sequences.clone()
    batch_size = len(decoder_output.sequences)

    # infer logits
    logits = torch.stack(decoder_output.scores, 1).cpu().max(-1)
    values = logits.values
    indices = logits.indices

    # loop
    for b in range(batch_size):
        mask = indices[b] != tokenizer.pad_token_id
        N = mask.sum().item()
        var = np.array(
            [np.var(s) / len(s) for s in batch(values[b, mask].float().numpy())]
        )
        if len(var) < repeat_threshold:
            output["repeats"].append(None)
            continue
        varvar = np.array([np.var(v) for v in subdiv(var[::-1])][::-1])

        #minlen = 120
        #if (indices[b] == tokenizer.eos_token_id).any() and (N + 1 < indices.shape[1]): # bug of .any()
        if (indices[b] == tokenizer.eos_token_id) and (N + 1 < indices.shape[1]):
            # there is an end to the generation, likely no repetitions
            output["repeats"].append(None)
            continue
        small_var = np.where(varvar < variance_threshold)[0]

        # early stopping
        if early_stopping and len(small_var) > 1:
            if np.all(np.diff(small_var) < 2):
                idx = int(min(max(small_var[0], 1) * variance_mult + min_len, max_id))
                if idx / N > 0.9:  # at most last bit
                    output["repeats"].append(None)
                    continue
                elif small_var[0] < small_var_threshold:
                    idx = 0
                #logging.warn("Found repetitions in sample %i" % b)
                output["repeats"].append(idx)
                output["sequences"][b, idx:] = tokenizer.pad_token_id
                output["repetitions"][b, :idx] = tokenizer.pad_token_id
            else:
                output["repeats"].append(None)
        else:
            output["repeats"].append(None)
    # loop done - - - -

    # batch decode
    output["repetitions"] = tokenizer.batch_decode(
        output["repetitions"], skip_special_tokens=True
    )
    # postprocess predictions
    output["predictions"] = postprocess(
        tokenizer.batch_decode(
            output["sequences"], skip_special_tokens=True
        ),
        markdown_fix=False,
    )

    return output



def amp_infer_context(model, *, no_grad=True):
    """
    Helper to set context
    """
    p = next(model.parameters(), None)
    dev = getattr(p, "device", torch.device("cpu"))
    dt  = getattr(p, "dtype", torch.float32)

    cm = ExitStack()
    if no_grad:
        cm.enter_context(torch.inference_mode())  # faster than no_grad for inference

    if dev.type == "cuda" and dt in (torch.float16, torch.bfloat16):
        cm.enter_context(torch.amp.autocast("cuda", dtype=dt))
    elif dev.type == "cpu" and dt == torch.bfloat16:
        cm.enter_context(torch.amp.autocast("cpu", dtype=torch.bfloat16))
    elif dev.type == "xpu" and dt in (torch.float16, torch.bfloat16):
        #cm.enter_context(torch.xpu.amp.autocast(dtype=dt, cache_enabled=False)) # bad style
        cm.enter_context(torch.amp.autocast("xpu", dtype=torch.bfloat16))
    else:
        cm.enter_context(nullcontext())

    return cm
#  - - - - - -


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
        # values has shape (batch, num_samples_in_window)
        n = 0 if self.values is None else self.values.shape[1]
        if n < 2:
            # Not enough samples yet → define variance as zeros (no early stopping impact)
            return torch.zeros(
                self.values.shape[0],
                dtype=self.values.dtype,
                device=self.values.device,
            )

        # Use sample variance (ddof=1) once we have >=2 samples (preserves your behavior)
        try:
            v = torch.var(self.values, dim=1, correction=1)  # PyTorch ≥1.10
        except TypeError:
            v = torch.var(self.values, dim=1, unbiased=True)  # older API

        return v / n if self.norm else v

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

# class StoppingCriteriaScores(StoppingCriteria)
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

def main(args):
    # load model/processor
    checkpoint = '/home/siebenschuh/AdaParse/models/facebook__nougat-base'
    processor = NougatProcessor.from_pretrained(checkpoint, use_fast=True, local_files_only=True)
    model = VisionEncoderDecoderModel.from_pretrained(checkpoint, local_files_only=True)
    print('\nAfter model = ... \n')

    # = = = = = = =
    #    MODEL
    # = = = = = = =

    model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
    model.config.pad_token_id           = processor.tokenizer.pad_token_id
    model.config.eos_token_id           = processor.tokenizer.eos_token_id

    # tokenizer
    tokenizer_file_path = Path(checkpoint) / 'tokenizer.json'
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_file_path))

    # device management
    model = move_to_custom_device(model, bf16=False)

    # DEBUG
    if 'ipex' in globals():
        model = ipex.optimize(model, dtype=model.dtype)

    # = = = = = =
    #  DATASET
    # = = = = = =
    # 1. destination
    args.pdf_dir = Path(args.pdf_dir)
    args.out_dir = Path(args.out_dir)
    # - create
    if not args.pdf_dir.is_dir():
        args.pdf_dir.mkdir(parents=True, exist_ok=True)
    if not args.out_dir.is_dir():
        args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1. load PDFs
    assert args.pdf_dir.is_dir(), "PDF directory does not exist"
    # TODO
    pdf_files = [args.pdf_dir / f for f in os.listdir(args.pdf_dir) if f.endswith('.pdf')]

    # DEBUG
    print('pdf_files')
    print(pdf_files)

    assert len(pdf_files) > 0, "No PDFs in `pdf_dir`"
    pdfs = [Path(pdf_file) for pdf_file in pdf_files]

    # 2. Fix arguments
    align_long_axis = False # hack, from SwinEncoder config
    input_size = [896, 672] # hack, from SwinEncoder config
    random_padding = False
    # - combine
    prepared_arg_triplet = (align_long_axis, input_size, random_padding)

    # dataset
    datasets: List[LazyDataset] = []
    for pdf in pdfs:
        if not pdf.exists():
            #self.logger.warning(f'Could not find {pdf}. Skipping.')
            print(f'Could not find {pdf}. Skipping.') # help
            continue

        # TODO: fix dir to dump MMD into
        #if self.config.mmd_out is not None:
        if True:
            out_path = args.out_dir / 'mmd' / pdf.with_suffix('.mmd').name
            if out_path.exists() and not self.config.recompute:
                #logger.info(
                #    f'Skipping {pdf.name}: already extracted. '
                #    ' Use --recompute config to override extraction.'
                #)
                print(f'Skipping {pdf.name}: already extracted. ') # help
                continue

        try:
            # dataset
            dataset = LazyDataset(
                pdf,
                partial(prepare_input_sc, prep_args=prepared_arg_triplet),
            )

        # PdfStreamError, ValueError, KeyError, pypdf.errors.PdfReadError,
        # and potentially other exceptions can be raised here.
        except Exception:
            #self.logger.info(f'Could not load file {pdf!s}.')
            print(f'Could not load file {pdf!s}.') # help
            continue
        datasets.append(dataset)

    # If there are no PDFs to process, return None
    if len(datasets) == 0:
        return None

    # = = = = = = = =
    #  DATALOADER
    # = = = = = = = =

    # dataloader arguments
    dl_kwargs = dict(
        batch_size=args.batch_size, # self.config.batchsize,
        pin_memory=True,
        num_workers=args.num_workers, # self.config.num_workers,
        shuffle=False,
        collate_fn=LazyDataset.ignore_none_collate,
    )
    #if self.config.num_workers > 0:
    if args.num_workers > 0:
        dl_kwargs['prefetch_factor'] = 2 #self.config.prefetch_factor

    # dataloader
    dataloader = DataLoader(ConcatDataset(datasets), **dl_kwargs)

    documents = [] # List[Dict[str, Any]] = []
    predictions: List[str] = []
    file_index = 0
    page_num = 0
    model_outputs = []

    # single image inference
    # Debug: confirm HF pre-processing matches expectations
    #image = Image.open(args.image_path).convert('RGB')

    # = = = = = = = =
    #  INFERENCE
    #   (Nougat)
    # = = = = = = = =
    for sample, is_last_page in dataloader:
        # DEBUG
        #print(f"type(sample) : {type(sample)}")
        #print(f"type(pixel_values) : {type(pixel_values)}")
        #print(f"pixel_values.size() : {pixel_values.size()}")

        with amp_infer_context(model=model):
            # encoder:
            encoded = processor(images=sample, return_tensors="pt").to(device=model.device, dtype=model.dtype)
            pixel_values = encoded.pixel_values

            # decoder: generate from full model
            decoder_output = model.generate(
                pixel_values=pixel_values,
                return_dict_in_generate=True,
                output_scores=True,
                do_sample=False,
                max_new_tokens=1024,              # or use max_length if you prefer
                bad_words_ids=[[processor.tokenizer.unk_token_id]],
                stopping_criteria=StoppingCriteriaList([StoppingCriteriaScores()])  # if you want your early stopping
            )

            #filter
            output = process_decoder_output(decoder_output=decoder_output, tokenizer=tokenizer)

            # print
            preds = output['predictions']

            # DEBUG
            print('\n\n')
            print(preds)

    return

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Extract text from PDF images using Nougat')
    parser.add_argument('pdf_dir', help='Path to the image file (e.g., ./data/)')
    parser.add_argument('--out_dir', '-o', help='Output file to save the extracted text (optional)')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='Batch size')
    parser.add_argument('--num_workers', '-w', type=int, default=4, help='Number of workers')
    args = parser.parse_args()
    main(args)
