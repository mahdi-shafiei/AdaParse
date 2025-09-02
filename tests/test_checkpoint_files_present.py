from __future__ import annotations
import json
import logging
import os
import socket
from pathlib import Path
from typing import Tuple, List, Optional

import pytest

# Toggle with ADAPARSE_RUN_FS_TESTS (defaults to 1 = run)
RUN_FS = os.getenv("ADAPARSE_RUN_FS_TESTS", "1") == "1"
pytestmark = pytest.mark.skipif(
    not RUN_FS, reason="set ADAPARSE_RUN_FS_TESTS=1 to run file-system checks"
)

# Files we expect inside the checkpoint dir
REQUIRED_FILES: List[str] = [
    "config.json",
    "pytorch_model.bin",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "swin_base_patch4_window12_384_22kto1k.pth",
]

# Only enforce size thresholds for the heavy blobs (avoid false positives on small JSONs)
MIN_SIZE: dict[str, int] = {
    "pytorch_model.bin": 10_000_000,            # ~10MB+
    "swin_base_patch4_window12_384_22kto1k.pth": 100_000_000,  # ~100MB+
}

def _repo_root() -> Path:
    # tests/ is under repo/, so parent() of this file's parent should be repo root
    return Path(__file__).resolve().parents[1]

def _load_state_json() -> Optional[dict]:
    state = _repo_root() / ".adaparse_state.json"
    if state.is_file():
        try:
            return json.loads(state.read_text())
        except Exception:
            return None
    return None

def _infer_machine() -> str:
    env = os.environ.get("ADAPARSE_MACHINE")
    if env:
        return env
    host = socket.gethostname()
    if host.startswith("aurora-uan-"): return "aurora"
    if host.startswith("polaris"):      return "polaris"
    if host.startswith("sophia"):       return "sophia"
    if host.startswith("lambda"):       return "lambda"
    return "local"

def _base_path_for(machine: str) -> Optional[str]:
    if os.environ.get("ADAPARSE_BASE_PATH"):
        return os.environ["ADAPARSE_BASE_PATH"]
    return {
        "aurora":  "/lus/flare/projects",
        "polaris": "/eagle/projects",
        "sophia":  "/eagle/projects",
        "lambda":  "/homes",
        "local":   None,
    }[machine]

def _project_root_via_env_or_state() -> Optional[Path]:
    # Highest priority explicit env
    v = os.environ.get("ADAPARSE_PROJECT_ROOT")
    if v:
        return Path(v)
    # Next: state json
    st = _load_state_json()
    if st and st.get("project_path"):
        return Path(st["project_path"])
    # Fallback inference
    machine = _infer_machine()
    base = _base_path_for(machine)
    if base is None:
        return None
    user = (
        os.environ.get("ADAPARSE_USER_NAME")
        or os.environ.get("USER")
        or os.environ.get("USERNAME")
    )
    proj = os.environ.get("ADAPARSE_PROJECT_NAME")
    if machine in {"polaris", "sophia", "aurora"} and not proj:
        return None
    if proj:
        return Path(base) / proj / user / "adaparse_data"
    else:
        return Path(base) / user / "adaparse_data"

def _checkpoint_dir() -> Optional[Path]:
    # 1) explicit env
    v = os.environ.get("ADAPARSE_CHECKPOINT")
    if v:
        return Path(v)
    # 2) state json
    st = _load_state_json()
    if st and st.get("checkpoint_dir"):
        return Path(st["checkpoint_dir"])
    # 3) project_root + default subpath
    root = _project_root_via_env_or_state()
    return (root / "meta" / "nougat" / "checkpoint") if root else None

# Parametrize per file (like the online test does per URL)
@pytest.mark.fs
@pytest.mark.parametrize("fname", REQUIRED_FILES, ids=REQUIRED_FILES)
def test_checkpoint_file_present_and_sane(fname: str):
    log = logging.getLogger(__name__)
    ckpt = _checkpoint_dir()
    if not ckpt:
        pytest.skip("checkpoint dir not resolved. Set ADAPARSE_CHECKPOINT or run setup to create .adaparse_state.json")
    if not ckpt.exists():
        pytest.skip(f"checkpoint dir not found on disk: {ckpt}")

    fpath = ckpt / fname
    log.info("Checking file: %s", fpath)
    assert fpath.is_file(), f"Missing {fname} in {ckpt}"

    size = fpath.stat().st_size
    log.info("%s size: %d bytes", fname, size)

    # Enforce min sizes where meaningful
    if fname in MIN_SIZE:
        assert size > MIN_SIZE[fname], f"{fname} too small (got {size} bytes) → likely a broken download"
