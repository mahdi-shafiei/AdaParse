# tests/test_standalone_swin_encoder.py
from __future__ import annotations
import json
import os
import sys
import logging
from pathlib import Path
from typing import Optional

import pytest

log = logging.getLogger(__name__)

# ----- helpers: locate repo root and checkpoint dir -----

REPO_ROOT = Path(__file__).resolve().parents[1]
STATE_JSON = REPO_ROOT / ".adaparse_state.json"
ENV_FILE   = REPO_ROOT / ".adaparse.env"

def _read_checkpoint_from_state() -> Optional[Path]:
    if STATE_JSON.is_file():
        try:
            data = json.loads(STATE_JSON.read_text())
            p = Path(data.get("checkpoint_dir", ""))
            if p.is_dir():
                return p
        except Exception as e:
            log.info("Could not parse %s: %s", STATE_JSON, e)
    return None

def _read_checkpoint_from_envfile() -> Optional[Path]:
    if ENV_FILE.is_file():
        try:
            for line in ENV_FILE.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("ADAPARSE_CHECKPOINT="):
                    # allow quoted or unquoted
                    raw = line.split("=", 1)[1]
                    raw = raw.strip().strip('"').strip("'")
                    p = Path(raw)
                    if p.is_dir():
                        return p
        except Exception as e:
            log.info("Could not parse %s: %s", ENV_FILE, e)
    return None

def _discover_checkpoint_dir() -> Optional[Path]:
    # 1) state json
    p = _read_checkpoint_from_state()
    if p:
        return p
    # 2) .env as fallback
    p = _read_checkpoint_from_envfile()
    if p:
        return p
    return None

CKPT_DIR = _discover_checkpoint_dir()

pytestmark = pytest.mark.skipif(
    CKPT_DIR is None,
    reason="checkpoint dir not found; run initial_setup_adaparse_project.sh first "
           "so .adaparse_state.json/.adaparse.env exist"
)

# Register this mark in pyproject.toml (recommended) to silence warnings:
# [tool.pytest.ini_options]
# markers = ["fs: filesystem / checkpoint tests"]

@pytest.mark.fs
def test_swinencoder_forward_cpu():
    # Make repo importable (in case PYTHONPATH isn't set)
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    # import heavy deps lazily so skip is fast if needed
    torch = pytest.importorskip("torch")
    timm  = pytest.importorskip("timm")

    # Import after path setup
    from adaparse.parsers.nougat_parser.model import SwinEncoder

    # Locate the expected PTH in the checkpoint dir
    assert CKPT_DIR is not None  # for type-checkers
    pth = CKPT_DIR / "swin_base_patch4_window12_384_22kto1k.pth"
    if not pth.is_file():
        # try a fallback glob if filename changes
        cand = sorted(CKPT_DIR.glob("swin_base*384*.pth"))
        if cand:
            pth = cand[0]
    assert pth.is_file(), f"could not find Swin weights (.pth) in {CKPT_DIR}"

    log.info("Using checkpoint dir: %s", CKPT_DIR)
    log.info("Using Swin weights   : %s", pth)

    # Build encoder (standalone Swin implementation)
    # Note: name_or_path accepts a Path to the directory containing the .pth
    # (this matches how the code in model.py was adapted to prefer local PTH over remote).
    torch.manual_seed(34)
    encoder = SwinEncoder(
        input_size=[896, 672],
        align_long_axis=False,
        window_size=7,
        encoder_layer=[2, 2, 14, 2],
        name_or_path=str(CKPT_DIR),   # point to the folder that holds the .pth
        patch_size=4,
        embed_dim=128,
        num_heads=[4, 8, 16, 32],
    )

    # Simple CPU forward (keep it small enough for CI/interactive nodes)
    torch.manual_seed(34)
    x = torch.rand((1, 3, 896, 672), dtype=torch.float32)
    out = encoder(x)

    # Accept either a tensor or (tensor, ...) style returns
    if isinstance(out, (list, tuple)):
        y = out[0]
    else:
        y = out

    assert hasattr(y, "shape"), "encoder output is not a tensor-like"
    assert y.dtype == torch.float32, f"unexpected dtype: {y.dtype}"
    assert torch.isfinite(y).all(), "output contains non-finite values"
    assert y.shape[0] == 1, f"unexpected batch dim: {y.shape}"

    # Extremely gentle signal check: total magnitude > 0
    total = y.abs().sum().item()
    log.info("Output abs-sum: %.6f", total)
    assert total > 0.0
