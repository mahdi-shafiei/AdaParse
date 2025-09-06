import torch
if torch.xpu.is_available():
    import intel_extension_for_pytorch as ipex
import re
from contextlib import ExitStack, nullcontext

def resolve_device() -> str:
    """
    One canonical policy: CUDA → XPU → MPS → CPU
    """
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    mps = getattr(torch.backends, "mps", None)
    if mps and torch.backends.mps.is_built() and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def resolve_dtype(full_precision: bool, device_str: str) -> torch.dtype:
    """
    Decide the target dtype for inference given the device and a 'bf16' preference.
    - CUDA: use bf16 if supported, else fp16 fallback (or fp32 if you prefer stricter).
    - XPU: use bf16 (fast path on Intel XPUs).
    - MPS: prefer fp16 today
    - CPU: fp32 (bf16 is slower/spotty)
    """
    if full_precision:
        return torch.float32
    if device_str == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device_str == "xpu":
        return torch.bfloat16
    if device_str == "mps":
        return torch.float16
    return torch.float32

def move_to_device_accelerator(model):
    """
    Move stat. model to accelerator (AdaParse's Regression model)
    """
    device = resolve_device()
    return model.to(device), device

def move_to_custom_device(model, bf16: bool = True):
    """
    Extension of nougat.utils.move_to_custom_device()
    to accommodate Nougat for cuda etc.
    """
    device = resolve_device()
    model = model.to(device)
    # bf16 only on CUDA/XPU for stability
    if bf16 and device in {"cuda", "xpu"}:
        model = model.to(torch.bfloat16)
    return model

def build_doc_and_indices(pages: list[str], sentinel: str = "\uE000") -> tuple[str, list[int]]:
    """
    Join text via `sentinel` char, `uE000` by default
    """
    assert isinstance(sentinel, str) and len(sentinel) == 1, "sentinel must be a single character"
    joined = sentinel.join(pages)
    # apply text normalization
    normalized = re.sub(r"\n{3,}", "\n\n", joined).strip()
    parts = normalized.split(sentinel)
    # remove sentinels while recording start offsets
    starts, cur = [], 0
    for part in parts:
        starts.append(cur)
        cur += len(part)
    return "".join(parts), starts

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
