# model.py (NEW)
from __future__ import annotations

import os
os.environ.setdefault("ALBUMENTATIONS_DISABLE_VERSION_CHECK", "1")

from pathlib import Path
from typing import List, Optional, Union
from collections import defaultdict
import contextlib
import logging
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import rotate, resize
import timm
from PIL import Image
from PIL import ImageOps
import cv2

from transformers import GenerationConfig
from transformers.modeling_utils import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.modeling_outputs import BaseModelOutput

# from timm.models.swin_transformer import SwinTransformer # LEGACY
#from legacy_timm.models.swin_transformer import SwinTransformer # NEW but dysfunctional
from adaparse.parsers.nougat_parser.legacy_timm.models.swin_transformer import SwinTransformer

from transformers import (
    PreTrainedTokenizerFast,
    StoppingCriteria,
    StoppingCriteriaList,
    MBartConfig,
    MBartForCausalLM,
    MBartForConditionalGeneration
)

# from nougat.postprocessing import postprocess # LEGACY
from adaparse.parsers.nougat_parser.postprocessing import postprocess
# from nougat.transforms import train_transform, test_transform # LEGACY
from adaparse.parsers.nougat_parser.transforms import train_transform, test_transform

class SwinEncoder(nn.Module):
    r"""
    NEW IMPLEMENTATION

    Encoder based on SwinTransformer
    Set the initial weights and configuration with a pretrained SwinTransformer and then
    modify the detailed configurations

    Args:
        input_size: Input image size (width, height)
        align_long_axis: Whether to rotate image if height is greater than width
        window_size: Window size(=patch size) of SwinTransformer
        encoder_layer: Number of layers of SwinTransformer encoder
        name_or_path: Name of a pretrained model name either registered in huggingface.co. or saved in local.
                      otherwise, `swin_base_patch4_window12_384` will be set (using `timm`).
    """

    def __init__(
        self,
        input_size: List[int],
        align_long_axis: bool,
        window_size: int,
        encoder_layer: List[int],
        patch_size: int,
        embed_dim: int,
        num_heads: List[int],
        name_or_path: str | bytes | Path | None = None
    ):
        super().__init__()
        self.input_size = input_size
        self.align_long_axis = align_long_axis
        self.window_size = window_size
        self.encoder_layer = encoder_layer
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.model = SwinTransformer(
            img_size=self.input_size,
            depths=self.encoder_layer,
            window_size=self.window_size,
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            num_classes=0,
        )

        # ---- weight loading policy -------------------------------------------------
        # 1) Nougat checkpoint dir -> read encoder.model.* from pytorch_model.bin
        # 2) Local .pth file       -> direct load (keys match timm SwinTransformer)
        # 3) timm id string        -> timm.create_model(..., pretrained=True), copy
        # 4) None                  -> leave randomly init; parent may load later
        # ---------------------------------------------------------------------------

        # print
        print('In model.py ... ')
        print(f'name_or_path : {name_or_path}')

        # DEBUG: block in `if False` maybe completely misguided
        # - .bin contains both encoder and decoeder weights
        # - torch.load deprecation warning
        # - empty meta model

        # default: name_or_path = None hence weights from Nougat checkpoint
        # timm.create fairly broken doesn't work
        # swin_state_dict = timm.create_model('timm/swin_base_patch4_window12_384_in22k', pretrained=True).state_dict()
        if False:
            # DEBUG

            if False:
                print('Indeed, `name_or_path` is not None!')
                # LEGACY: timm 0.5.4
                # - - - - - - - - - -
                #swin_state_dict = timm.create_model(
                #    "swin_base_patch4_window12_384", pretrained=True
                #).state_dict()

                # NEW: timm 1.0.19
                # - - - - - - - - - -
                swin_ckpt_pretrain = Path(name_or_path) / "swin_base_patch4_window12_384_22kto1k.pth"
                if swin_ckpt_pretrain.is_file():
                    swin_state_dict = torch.load(swin_ckpt_pretrain, map_location="cpu")['model']
                    # DEBUG
                    print('loaded ..w12_384_22kto1k.pth into SWIN!')
                else:
                    raise FileNotFoundError("Pre-trained Swin checkpoint (.pth) path not found")

            #
            new_swin_state_dict = self.model.state_dict()

            # DEBUG
            #print('\nnew_swin_state_dict.keys()')
            #print(new_swin_state_dict.keys())
            #print("\nswin_state_dict['model'].keys()")
            #print(swin_state_dict['model'].keys())
            # - - - - -

            for x in new_swin_state_dict:
                if x.endswith("relative_position_index") or x.endswith("attn_mask"):
                    pass
                elif (
                    x.endswith("relative_position_bias_table")
                    and self.model.layers[0].blocks[0].attn.window_size[0] != 12
                ):
                    pos_bias = swin_state_dict[x].unsqueeze(0)[0]
                    old_len = int(math.sqrt(len(pos_bias)))
                    new_len = int(2 * window_size - 1)
                    pos_bias = pos_bias.reshape(1, old_len, old_len, -1).permute(
                        0, 3, 1, 2
                    )
                    pos_bias = F.interpolate(
                        pos_bias,
                        size=(new_len, new_len),
                        mode="bicubic",
                        align_corners=False,
                    )
                    new_swin_state_dict[x] = (
                        pos_bias.permute(0, 2, 3, 1)
                        .reshape(1, new_len**2, -1)
                        .squeeze(0)
                    )
                else:
                    print(f'key name. x={x}')
                    new_swin_state_dict[x] = swin_state_dict[x]   # <- bug here
            self.model.load_state_dict(new_swin_state_dict)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, num_channels, height, width)
        """
        x = self.model.patch_embed(x)
        x = self.model.pos_drop(x)
        x = self.model.layers(x)
        return x

    @staticmethod
    def crop_margin(img: Image.Image) -> Image.Image:
        data = np.array(img.convert("L"))
        data = data.astype(np.uint8)
        max_val = data.max()
        min_val = data.min()
        if max_val == min_val:
            return img
        data = (data - min_val) / (max_val - min_val) * 255
        gray = 255 * (data < 200).astype(np.uint8)

        # Likely slow
        coords = cv2.findNonZero(gray)  # Find all non-zero points (text)
        a, b, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box
        return img.crop((a, b, w + a, h + b))

    @property
    def to_tensor(self):
        if self.training:
            return train_transform
        else:
            return test_transform

    def prepare_input(
        self, img: Image.Image, random_padding: bool = False
    ) -> torch.Tensor:
        """
        Convert PIL Image to tensor according to specified input_size after following steps below:
            - resize
            - rotate (if align_long_axis is True and image is not aligned longer axis with canvas)
            - pad
        """
        if img is None:
            return
        # crop margins
        try:
            img = self.crop_margin(img.convert("RGB"))
        except OSError:
            # might throw an error for broken files
            return
        if img.height == 0 or img.width == 0:
            return
        if self.align_long_axis and (
            (self.input_size[0] > self.input_size[1] and img.width > img.height)
            or (self.input_size[0] < self.input_size[1] and img.width < img.height)
        ):
            img = rotate(img, angle=-90, expand=True)
        img = resize(img, min(self.input_size))
        img.thumbnail((self.input_size[1], self.input_size[0]))
        delta_width = self.input_size[1] - img.width
        delta_height = self.input_size[0] - img.height
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
        return self.to_tensor(ImageOps.expand(img, padding))

class BARTDecoder(nn.Module):
    """
    Decoder based on Multilingual BART (facebook/mbart-large-50)
    mBART-based decoder (decoder-only usage w/ cross-attention).

    Exposes:
      - self.tokenizer (PreTrainedTokenizerFast)
      - self.model (MBartForCausalLM or MBartForConditionalGeneration)
      - .add_special_tokens(...)
      - .prepare_inputs_for_inference(...)
      - .resize_bart_abs_pos_emb(...)
    """

    def __init__(
        self,
        decoder_layer: int,
        max_position_embeddings: int,
        name_or_path: str | Path,
        hidden_dimension: int = 1024,
        cond_gen: bool = False,  # new arg
        dropout=0.0, attention_dropout=0.0, activation_dropout=0.0
    ):
        super().__init__()
        self.decoder_layer = int(decoder_layer)
        self.max_position_embeddings = int(max_position_embeddings)

        # always load weights locally
        p = Path(name_or_path)
        if not p.exists():
            raise FileNotFoundError(f"[BARTDecoder] name_or_path does not exist: {p}")

        # tokenizer (fail-fast)
        p = Path(name_or_path)
        if p.exists() and (p / "tokenizer.json").exists():
            tokenizer_file_path = p / "tokenizer.json"
        elif (Path(__file__).parent / "dataset" / "tokenizer.json").exists():
            tokenizer_file_path = Path(__file__).parent / "dataset" / "tokenizer.json"
        else:
            raise FileNotFoundError(f"Could not load tokenizer; no such file at {p / 'tokenizer.json'} "
                                    f"or {(Path(__file__).parent / 'dataset' / 'tokenizer.json')}")
        # - load
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_file_path))
        tokenizer.pad_token = "<pad>"
        tokenizer.bos_token = "<s>"
        tokenizer.eos_token = "</s>"
        tokenizer.unk_token = "<unk>"
        # - assign
        self.tokenizer = tokenizer

        # model config
        cfg = MBartConfig(
            is_decoder=True,
            is_encoder_decoder=False,  # flip later for CausalLM to enable X-attn
            add_cross_attention=True,
            decoder_layers=self.decoder_layer,
            max_position_embeddings=self.max_position_embeddings,
            vocab_size=len(self.tokenizer),
            scale_embedding=True,
            add_final_layer_norm=True,
            d_model=int(hidden_dimension),
            # NEW: carry through the configured dropouts for stability-equivalence
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
        )

        # MBartForCausalLM (original)
        ModelCls = MBartForConditionalGeneration if cond_gen else MBartForCausalLM

        # load weights from path (local, fine-tuned)
        #self.model = ModelCls.from_pretrained(str(p), low_cpu_mem_usage=True)  # changed
        #self.model = ModelCls(config=cfg, low_cpu_mem_usage=True) # new low_cpu_mem_usage unknown even under transformers 4.55.0
        self.model = ModelCls(config=cfg) # new

        if not cond_gen:
            self.model.config.is_encoder_decoder = True  # ensure X-attn flag still flipped
            self.model.prepare_inputs_for_generation = self.prepare_inputs_for_inference
        #self.model.model.decoder.embed_tokens.padding_idx = self.tokenizer.pad_token_id  # out now?
        self.model.model.decoder.embed_tokens.padding_idx = self.tokenizer.pad_token_id


    def add_special_tokens(self, list_of_tokens: List[str]) -> None:
        """
        Add extra special tokens & resize embeddings.
        """
        n_new = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": sorted(set(list_of_tokens))}
        )
        if n_new > 0:
            self.model.resize_token_embeddings(len(self.tokenizer))

        pass

    def prepare_inputs_for_inference(
        self,
        input_ids: torch.Tensor,
        encoder_outputs: torch.Tensor,
        past: Optional[tuple] = None,
        past_key_values: Optional[tuple] = None,
        use_cache: Optional[bool] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        Prepare inputs for text generation (decoder-only step).
        """
        pad_id = self.tokenizer.pad_token_id
        attn = attention_mask if attention_mask is not None else input_ids.ne(pad_id)
        pkv = past_key_values if past is None else past
        if pkv is not None:
            input_ids = input_ids[:, -1:]

        # assemble
        output =  {
            "input_ids": input_ids,
            "attention_mask": attn.long(),
            "past_key_values": pkv,
            "use_cache": use_cache,
            "encoder_hidden_states": encoder_outputs.last_hidden_state,
        }

        return output

    @staticmethod
    def resize_bart_abs_pos_emb(weight: torch.Tensor, max_length: int) -> torch.Tensor:
        """
        Resize (absolute) position embeddings (truncate or 1D linear interpolate).
        Expects weight shape [old_len, dim].
        """
        old_len, dim = weight.shape
        if old_len == max_length:
            return weight
        if old_len > max_length:
            return weight[:max_length, ...]

        # in: [1, dim, old_len]
        w = weight.t().unsqueeze(0)
        w = F.interpolate(w, size=max_length, mode="linear", align_corners=False)
        # out: [max_length, dim]
        w = w.squeeze(0).t()
        return w

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        past_key_values: Optional[tuple] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        Passthrough to underlying HF model; signature kept for HF compatibility.
        """
        return self.model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            encoder_hidden_states=encoder_hidden_states,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

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

class NougatConfigInference(PretrainedConfig):
    """
    Configuration of a [`NougatModelInference`], the inference-only variant of
    ['NougatModel'] defining the model architecture.

    Args:
        input_size:
            Input image size (canvas size) of Nougat.encoder, SwinTransformer in this codebase
        align_long_axis:
            Whether to rotate image if height is greater than width
        window_size:
            Window size of Nougat.encoder, SwinTransformer in this codebase
        encoder_layer:
            Depth of each Nougat.encoder Encoder layer, SwinTransformer in this codebase
        decoder_layer:
            Number of hidden layers in the Nougat.decoder, such as MBART
        max_position_embeddings
            Trained max position embeddings in the Nougat decoder,
            if not specified, it will have same value with max_length
        max_length:
            Max position embeddings(=maximum sequence length) you want to train
        name_or_path:
            Name of a pretrained model name either registered in huggingface.co. or saved in local
        deterministic:
            If True, runs pipeline explicitely as deterministic
        full_precision:
            If True, float32 else bf16
    """
    model_type = "nougat_inference"

    def __init__(
        self,
        input_size: List[int] = [896, 672],
        align_long_axis: bool = False,
        window_size: int = 7,
        encoder_layer: List[int] = [2, 2, 14, 2],
        decoder_layer: int = 10,
        max_position_embeddings: int = None,
        max_length: int = 4096,
        name_or_path: str|Path = "",
        patch_size: int = 4,
        embed_dim: int = 128,
        num_heads: List[int] = [4, 8, 16, 32],
        hidden_dimension: int = 1024,
        cond_gen:bool = False,
        decoder_dropout: float = 0.0,
        decoder_attention_dropout: float = 0.0,
        decoder_activation_dropout: float = 0.0,
        decoder_use_cache: bool = True,
        decoder_do_sample: bool = False,
        deterministic: bool = True,
        full_precision: bool = False,
        seed: int | None = 1234,
        # new
        compile_encoder: bool = True,             # compile after loading
        bootstrap_swin_from_pth: bool = False,    # only use local .pth if True
        pretrained_swin_pth: str | None = None,   # explicit .pth path, overrides default
        **kwargs,
    ):
        super().__init__()
        # existing assignments...
        self.compile_encoder = compile_encoder
        self.bootstrap_swin_from_pth = bootstrap_swin_from_pth
        self.pretrained_swin_pth = pretrained_swin_pth
        # args
        self.input_size = input_size
        self.align_long_axis = align_long_axis
        self.window_size = window_size
        self.encoder_layer = encoder_layer
        self.decoder_layer = decoder_layer
        self.max_position_embeddings = (
            max_length if max_position_embeddings is None else max_position_embeddings
        )
        self.max_length = max_length
        self.name_or_path = name_or_path
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.hidden_dimension = hidden_dimension
        # accelerated decoder config
        self.cond_gen = cond_gen
        self.decoder_dropout = decoder_dropout
        self.decoder_attention_dropout = decoder_attention_dropout
        self.decoder_activation_dropout = decoder_activation_dropout
        self.decoder_use_cache = decoder_use_cache
        self.decoder_do_sample = decoder_do_sample

class NougatModelInference(PreTrainedModel):
    """
    Nougat (arxiv|2308.13418) implementation for accelerated inference within AdaParse
    """

    def __init__(self, config: NougatConfigInference):
        """
        new init
        """
        super().__init__(config)
        self.config = config

        # DEBUG
        #print(f'self.config.name_or_path : {self.config.name_or_path}')


        # encoder (ViT)
        self.encoder = SwinEncoder(
            input_size=self.config.input_size,
            align_long_axis=self.config.align_long_axis,
            window_size=self.config.window_size,
            encoder_layer=self.config.encoder_layer,
            name_or_path=self.config.name_or_path,
            patch_size=self.config.patch_size,
            embed_dim=self.config.embed_dim,
            num_heads=self.config.num_heads,
        )

        # carry decoder dropouts from config into instance (so BARTDecoder sees them)
        for k in ("decoder_dropout","decoder_attention_dropout","decoder_activation_dropout"):
            setattr(self.decoder if hasattr(self, "decoder") else self, k, getattr(self.config, k, 0.0))


        # decoder (mBART)
        self.decoder = BARTDecoder(
            decoder_layer=self.config.decoder_layer,
            max_position_embeddings=self.config.max_position_embeddings,
            name_or_path=self.config.name_or_path,
            hidden_dimension=self.config.hidden_dimension,
            cond_gen=self.config.cond_gen, # new
            dropout=self.config.decoder_dropout,
            attention_dropout=self.config.decoder_attention_dropout,
            activation_dropout=self.config.decoder_activation_dropout,
        )

        # inference by default
        self.eval()
        self.requires_grad_(False)

        # IMPORTANT: DO NOT COMPILE HERE. We’ll compile in from_pretrained after load.
        self._compiled = False

        # device/dtype management
        from adaparse.parsers.device_utils import resolve_device
        from adaparse.parsers.device_utils import resolve_dtype

        device_str = resolve_device()
        self._device = torch.device(device_str)
        target_dtype = resolve_dtype(getattr(self.config, "full_precision", False), device_str=device_str)

        # - move then cast
        self.to(self._device)
        if target_dtype != next(self.parameters()).dtype:
            self.to(dtype=target_dtype)

        # - cache
        self._dtype = next(self.parameters()).dtype
        self._non_blocking = self._device.type in ("cuda", "xpu")
        self._use_channels_last = (self._device.type == "cuda")
        self.deterministic = getattr(config, "deterministic", False)

        # deterministic (portable)
        self._sdpa_ctx = self._make_sdpa_ctx(self.deterministic)

        # - device switch
        torch.use_deterministic_algorithms(self.deterministic)

        # decoder defaults
        gen = GenerationConfig.from_model_config(self.decoder.model.config)
        gen.do_sample = False
        gen.use_cache = self.config.decoder_use_cache
        gen.pad_token_id = self.decoder.tokenizer.pad_token_id # new
        gen.eos_token_id = self.decoder.tokenizer.eos_token_id # new
        gen.max_length = config.max_length
        # overrides from config
        if getattr(config, "generation_overrides", None):
            gen.update(config.generation_overrides)
        self._gen_cfg = gen

        # -------- compile encoder only (safe win) --------
        #try:
        #    self.encoder = torch.compile(self.encoder, fullgraph=True, mode="reduce-overhead")
        #except Exception:
        #    pass

    @staticmethod
    def _make_sdpa_ctx(deterministic: bool):
        """Prefer fast SDPA on accelerators; fall back cleanly and support a deterministic mode."""
        try:
            from torch.nn.attention import sdpa_kernel, SDPBackend
            if deterministic:
                return sdpa_kernel(SDPBackend.MATH)
            # fast path: FLASH/EFFICIENT when available, MATH as fallback
            return sdpa_kernel(SDPBackend.FLASH_ATTENTION,
                               SDPBackend.EFFICIENT_ATTENTION,
                               SDPBackend.MATH)
        except Exception:
            return contextlib.nullcontext()

    def forward(
        self,
        image_tensors: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> Seq2SeqLMOutput:
        """
        Calculate a loss given an input image and a desired token sequence,
        the model will be trained in a teacher-forcing manner

        Args:
            image_tensors: (batch_size, num_channels, height, width)
            decoder_input_ids: (batch_size, sequence_length, embedding_dim)
        """
        encoder_outputs = self.encoder(image_tensors)
        # .contiguous() copies removed
        attn = attention_mask[:, :-1] if attention_mask is not None else None
        decoder_outputs = self.decoder(input_ids=decoder_input_ids[:, :-1],
            encoder_hidden_states=encoder_outputs,
            attention_mask=attn,
            labels=decoder_input_ids[:, 1:],
        )
        return decoder_outputs

    def _init_weights(self, *args, **kwargs):
            return

    @staticmethod
    def _batch_chunks(arr: np.ndarray, b: int = 15):
        return [arr[i:i+b] for i in range(0, len(arr), b)]

    @staticmethod
    def _subdiv_sliding(arr: np.ndarray, w: int = 10):
        return [arr[i:i+w] for i in range(0, max(0, len(arr)-w+1))]

    def _build_gen_defaults(self):
        # Call once in __init__; no per-call churn
        gen = GenerationConfig.from_model_config(self.decoder.model.config)
        gen.do_sample = False
        gen.use_cache = True
        gen.max_length = self.config.max_length
        return gen

    @torch.inference_mode()
    def inference(
        self,
        image: Image.Image = None,
        image_tensors: torch.Tensor | None = None,
        return_attentions: bool = False,
        early_stopping: bool = True,
    ):
        """
        Generate a token sequence (greedily) in an auto-regressive manner.
        Lower peak memory and less overhead than NougatModel.inference()

        Args:
            image: input document image (PIL.Image)
            image_tensors: (1, num_channels, height, width)
                convert prompt to tensor if image_tensor is not fed
        """
        # inputs/outputs
        output = {
            "predictions": [],
            "sequences": [],
            "repeats": [],
            "repetitions": [],
        }

        if image_tensors is None and image is None:
            logging.warning("inference(): neither image nor image_tensors provided")
            return output

        if image_tensors is None:
            image_tensors = self.encoder.prepare_input(image).unsqueeze(0)

        if self._device.type != "mps":
            image_tensors = image_tensors.to(self._dtype, copy=False)
        x = image_tensors.to(self._device, non_blocking=self._non_blocking)
        if self._use_channels_last:
            x = x.to(memory_format=torch.channels_last)

        with self._sdpa_ctx:
            encoded_x = self.encoder(x)
            enc_out = BaseModelOutput(last_hidden_state=encoded_x)
            dec_out = self.decoder.model.generate(encoder_outputs=enc_out,
                                                  generation_config=self._gen_cfg,
                                                  min_length=1,
                                                  pad_token_id=self.decoder.tokenizer.pad_token_id,
                                                  eos_token_id=self.decoder.tokenizer.eos_token_id,
                                                  bad_words_ids=[[self.decoder.tokenizer.unk_token_id]],
                                                  return_dict_in_generate=True,
                                                  output_scores=True,           # keep if you use the original repetition heuristic
                                                  output_attentions=return_attentions,
                                                  do_sample=False,
                                                  stopping_criteria=StoppingCriteriaList(
                                                      [StoppingCriteriaScores()] if early_stopping else []
                                                  ),
                                                 )

        # clone
        sequences = dec_out.sequences             # alias/view
        repetitions = None                        # defer clone

        B = sequences.size(0)
        T = len(dec_out.scores)  # number of generated steps

        # streamed logits (avoid per-step CPU syncs)
        vals_gpu = torch.stack([logits.max(dim=-1)[0] for logits in dec_out.scores], dim=1)
        idxs_gpu = torch.stack([logits.max(dim=-1)[1] for logits in dec_out.scores], dim=1)
        vals_cpu = vals_gpu.detach().cpu()  # Single transfer
        idxs_cpu = idxs_gpu.detach().cpu()  # Single transfer

        values = vals_cpu.numpy()
        indices = idxs_cpu.numpy()
        pad_id = self.decoder.tokenizer.pad_token_id
        eos_id = self.decoder.tokenizer.eos_token_id

        # ANALYZE PERFORMANCE BOTTLECNECKS BELOW
        # repitition detection & handling heuristics
        output["repeats"] = []
        for b in range(B):
            mask = indices[b] != pad_id
            N = int(mask.sum())
            if N == 0:
                output["repeats"].append(None)
                continue

            v_seq = values[b, mask]
            # variance per small batch of scores
            var = np.array([np.var(chunk) / max(1, len(chunk)) for chunk in self._batch_chunks(v_seq, b=15)])
            if var.size < 10:
                output["repeats"].append(None)
                continue

            varvar = np.array([np.var(w) for w in self._subdiv_sliding(var[::-1], w=10)])[::-1]
            minlen = 120

            # if EOS exists and we didn't pad the rest, likely no repetitions
            if (indices[b] == eos_id).any() and (N + 1) < indices.shape[1]:
                output["repeats"].append(None)
                continue

            small_var = np.where(varvar < 0.045)[0]
            if early_stopping and (small_var.size > 1) and np.all(np.diff(small_var) < 2):
                idx = int(min(max(small_var[0], 1) * 1.08 + minlen, 4095))
                if idx / N > 0.9:  # at most last bit
                    output["repeats"].append(None)
                    continue
                elif small_var[0] < 30:
                    idx = 0
                logging.warning(f"Found repetitions in sample {b}")
                output["repeats"].append(idx)

                # conditional copy
                if repetitions is None:
                    repetitions = sequences.clone()

                # in-place
                sequences[b, idx:] = pad_id           # in-place truncate predictions
                repetitions[b, :idx] = pad_id         # in-place keep only repetitions

            else:
                output["repeats"].append(None)

        # finalize tensor refs
        if repetitions is None:
            repetitions = sequences

        # assign
        output["sequences"]   = sequences
        output["repetitions"] = repetitions

        # decode strings
        output["repetitions"] = self.decoder.tokenizer.batch_decode(
            output["repetitions"], skip_special_tokens=True
        )
        output["predictions"] = postprocess(
            self.decoder.tokenizer.batch_decode(
                output["sequences"], skip_special_tokens=True
            ),
            markdown_fix=False,
        )

        if return_attentions:
            output["attentions"] = {
                "self_attentions": getattr(dec_out, "decoder_attentions", None),
                "cross_attentions": getattr(dec_out, "cross_attentions", None),
            }

        return output

    @classmethod
    def from_pretrained(cls, model_path: str | bytes | Path, *args, **kwargs):
        # 1) ensure we don't mask true mismatches
        #kwargs.setdefault("ignore_mismatched_sizes", False)
        kwargs.setdefault("ignore_mismatched_sizes", True)
        # 2) we might get a user-supplied config; mark it to defer compile
        cfg = kwargs.get("config", None)
        if cfg is not None:
            setattr(cfg, "__defer_compile__", True)
        model: "NougatModelInference" = super(NougatModelInference, cls).from_pretrained(
            model_path, *args, **kwargs
        )

        # 3) handle decoder absolute position embeddings deterministically
        want = model.config.max_position_embeddings + 2
        emb = model.decoder.model.model.decoder.embed_positions.weight
        if emb.shape[0] != want:
            with torch.no_grad():
                model.decoder.model.model.decoder.embed_positions.weight = torch.nn.Parameter(
                    model.decoder.resize_bart_abs_pos_emb(emb.detach(), want)
                )

        # 4) compile the encoder AFTER all weights are loaded (optional)
        if getattr(model.config, "compile_encoder", True) and not model._compiled:
            try:
                model.encoder = torch.compile(model.encoder, fullgraph=True, mode="reduce-overhead")
                model._compiled = True
            except Exception:
                # fall back quietly; determinism intact
                pass

        return model
