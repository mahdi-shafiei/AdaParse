#!/usr/bin/env python
import os, sys, json, yaml, hashlib
from pathlib import Path
import torch
from PIL import Image, ImageDraw

# --- seed + deterministic ---
os.environ.setdefault("PYTHONHASHSEED", "34")
torch.manual_seed(34)
torch.use_deterministic_algorithms(True)
torch.set_num_threads(1); torch.set_num_interop_threads(1)

CFG_YAML = Path("/home/siebenschuh/AdaParse/configs/nougat/aurora.yaml")
REPO_ROOT = CFG_YAML.parents[2] if CFG_YAML.exists() else Path("/home/siebenschuh/AdaParse")
STATE_JSON = REPO_ROOT / ".adaparse_state.json"
ENV_FILE   = REPO_ROOT / ".adaparse.env"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def find_checkpoint_dir():
    if STATE_JSON.exists():
        try:
            d = json.loads(STATE_JSON.read_text())
            p = d.get("checkpoint_dir")
            if p and Path(p).exists():
                return Path(p)
        except Exception:
            pass
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            if line.startswith("ADAPARSE_CHECKPOINT="):
                p = line.split("=",1)[1].strip()
                if p and Path(p).exists():
                    return Path(p)
    if CFG_YAML.exists():
        try:
            y = yaml.safe_load(CFG_YAML.read_text())
            def scan(o):
                if isinstance(o, dict):
                    for v in o.values():
                        r = scan(v);
                        if r: return r
                elif isinstance(o, list):
                    for v in o:
                        r = scan(v)
                        if r: return r
                elif isinstance(o, str):
                    if "checkpoint" in o and Path(o).exists():
                        return Path(o)
            r = scan(y)
            if r: return r
        except Exception:
            pass
    raise FileNotFoundError("checkpoint_dir not found")

ckpt_dir = find_checkpoint_dir()
print(f"[INFO] checkpoint_dir = {ckpt_dir}")

from adaparse.parsers.nougat_parser.model import NougatConfigInference, NougatModelInference

# Pull shapes (best-effort) or use defaults
def maybe_get(d, path, default=None):
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur

yaml_cfg = {}
if CFG_YAML.exists():
    try:
        yaml_cfg = yaml.safe_load(CFG_YAML.read_text())
    except Exception:
        pass

input_size       = maybe_get(yaml_cfg, "model.input_size", [896, 672])
align_long_axis  = bool(maybe_get(yaml_cfg, "model.align_long_axis", False))
window_size      = int(maybe_get(yaml_cfg, "model.window_size", 7))
encoder_layer    = maybe_get(yaml_cfg, "model.encoder_layer", [2,2,14,2])
patch_size       = int(maybe_get(yaml_cfg, "model.patch_size", 4))
embed_dim        = int(maybe_get(yaml_cfg, "model.embed_dim", 128))
num_heads        = maybe_get(yaml_cfg, "model.num_heads", [4,8,16,32])
hidden_dimension = int(maybe_get(yaml_cfg, "model.hidden_dimension", 1024))
decoder_layer    = int(maybe_get(yaml_cfg, "model.decoder_layer", 10))
MAX_LEN          = 64

# Build an inference config; crucially, set name_or_path → ckpt_dir
cfg = NougatConfigInference(
    input_size=input_size,
    align_long_axis=align_long_axis,
    window_size=window_size,
    encoder_layer=encoder_layer,
    decoder_layer=decoder_layer,
    max_position_embeddings=MAX_LEN,
    max_length=MAX_LEN,
    name_or_path=str(ckpt_dir),
    patch_size=patch_size,
    embed_dim=embed_dim,
    num_heads=num_heads,
    hidden_dimension=hidden_dimension,
    cond_gen=False,
    deterministic=True,
    full_precision=True,         # float32; avoids bf16 subtle diffs on CPU
    decoder_dropout=0.0,
    decoder_attention_dropout=0.0,
    decoder_activation_dropout=0.0,
)

# *** THIS is the key change: load ALL weights from the checkpoint dir ***
#print('BEFORE\n\n')
model = NougatModelInference.from_pretrained(
    str(ckpt_dir),
    config=cfg,                     # use our inference config
    local_files_only=True,
    ignore_mismatched_sizes=True,   # new
)
#print('AFTER\n\n')
model.eval()

#print(f"[INFO] device={next(model.parameters()).device}, dtype={next(model.parameters()).dtype}")

# Deterministic synthetic image
H, W = input_size
img = Image.new("RGB", (W, H), (255,255,255))
d = ImageDraw.Draw(img); d.rectangle([W//4, H//3, W//4+120, H//3+60], outline=(0,0,0), width=4)
d.text((W//4+10, H//3+15), "AdaParse", fill=(0,0,0))

with torch.inference_mode():
    x = model.encoder.prepare_input(img).unsqueeze(0)

tok = model.decoder.tokenizer
ids = tok("x", return_tensors="pt", add_special_tokens=True)["input_ids"]
if ids.shape[1] < 8:
    pad = torch.full((1, 8 - ids.shape[1]), tok.pad_token_id, dtype=ids.dtype)
    ids = torch.cat([ids, pad], dim=1)

def run_once():
    with torch.inference_mode():
        enc = model.encoder(x)
        out = model.decoder(input_ids=ids, encoder_hidden_states=enc, labels=ids)
        return out.loss.detach().cpu(), out.logits.detach().cpu()

loss1, logits1 = run_once()
loss2, logits2 = run_once()

m1 = hashlib.md5(logits1.numpy().tobytes()).hexdigest()
m2 = hashlib.md5(logits2.numpy().tobytes()).hexdigest()

print(f"[OK] run1 loss={float(loss1):.6f}, logits md5={m1}")
print(f"[OK] run2 loss={float(loss2):.6f}, logits md5={m2}")
print("[OK] deterministic =", m1 == m2 and float(loss1) == float(loss2))
