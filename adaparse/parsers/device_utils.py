import torch
import re

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
