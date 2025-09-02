# tests/test_bart_decoder.py
from __future__ import annotations
import os, sys, json, logging, types
from pathlib import Path
import pytest
import torch

pytestmark = pytest.mark.fs  # mark like your other FS/integration tests

# --- locate repo root and import BARTDecoder ---------------------------------
THIS = Path(__file__).resolve()
ROOT = THIS.parents[1]
sys.path.insert(0, str(ROOT))

from adaparse.parsers.nougat_parser.model import BARTDecoder  # type: ignore


def _checkpoint_dir() -> Path | None:
    """Find checkpoint dir via env, .adaparse_state.json, or .adaparse.env."""
    # 1) explicit env
    env = os.getenv("ADAPARSE_CHECKPOINT")
    if env:
        p = Path(env)
        if p.exists():
            return p

    # 2) repo state json
    state_json = ROOT / ".adaparse_state.json"
    if state_json.exists():
        try:
            d = json.loads(state_json.read_text())
            p = Path(d.get("checkpoint_dir", ""))
            if p.exists():
                return p
        except Exception:
            pass

    # 3) repo env file
    env_file = ROOT / ".adaparse.env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("ADAPARSE_CHECKPOINT="):
                path = line.split("=", 1)[1].strip()
                p = Path(path)
                if p.exists():
                    return p
    return None


def _pad_or_trim(ids: torch.Tensor, pad_id: int, T: int) -> torch.Tensor:
    if ids.size(1) < T:
        need = T - ids.size(1)
        return torch.nn.functional.pad(ids, (0, need), value=pad_id)
    return ids[:, :T]


def test_bartdecoder_forward_cpu(caplog):
    caplog.set_level(logging.INFO)
    ckpt = _checkpoint_dir()
    if not ckpt:
        pytest.skip("No checkpoint dir found (.adaparse_state.json / .adaparse.env / env var)")

    logging.info("Using checkpoint dir: %s", ckpt)

    # Build a small decoder (random init, local tokenizer)
    torch.manual_seed(123)
    dec = BARTDecoder(
        decoder_layer=2,                 # keep it light
        max_position_embeddings=128,
        name_or_path=str(ckpt),          # tokenizer.json lives here
        hidden_dimension=1024,           # matches Nougat config
        cond_gen=False,                  # use CausalLM path
    )
    dec.eval()

    tok = dec.tokenizer
    assert tok.pad_token_id is not None and tok.eos_token_id is not None

    # Tiny batch of tokens; pad/trim to fixed length
    ids = tok.encode("hello world", add_special_tokens=True)
    ids = torch.tensor([ids], dtype=torch.long)
    ids = _pad_or_trim(ids, tok.pad_token_id, T=8)
    attn = ids.ne(tok.pad_token_id).long()

    # Random encoder states with the right hidden size
    B, S, D = ids.size(0), 64, dec.model.config.d_model
    torch.manual_seed(123)
    enc_hidden = torch.randn(B, S, D)

    # Forward with labels -> loss + logits
    out = dec(
        input_ids=ids,
        attention_mask=attn,
        encoder_hidden_states=enc_hidden,
        labels=ids,
    )

    assert hasattr(out, "logits")
    assert out.logits.shape[:2] == (B, ids.size(1))
    assert out.logits.shape[-1] == dec.model.config.vocab_size
    assert torch.isfinite(out.logits).all()
    assert out.loss is None or torch.isfinite(out.loss)

    # prepare_inputs_for_inference wiring
    enc_obj = types.SimpleNamespace(last_hidden_state=enc_hidden)
    pin = dec.prepare_inputs_for_inference(ids, encoder_outputs=enc_obj, attention_mask=attn)
    assert "encoder_hidden_states" in pin and pin["encoder_hidden_states"].shape == enc_hidden.shape
    assert "input_ids" in pin and pin["input_ids"].shape[0] == B

    # add_special_tokens should resize embeddings
    n0 = len(tok)
    dec.add_special_tokens(["<page>", "<cell>", "<page>"])  # dedup handled inside
    assert len(tok) >= n0 + 2
    assert dec.model.get_input_embeddings().weight.size(0) == len(tok)


def test_resize_bart_abs_pos_emb():
    D, old, new = 1024, 32, 64
    w = torch.randn(old, D)
    w2 = BARTDecoder.resize_bart_abs_pos_emb(w, new)
    assert w2.shape == (new, D)
    w3 = BARTDecoder.resize_bart_abs_pos_emb(w2, old)
    assert w3.shape == (old, D)
