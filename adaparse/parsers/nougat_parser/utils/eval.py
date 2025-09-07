# eval.py
from pathlib import Path
from rapidfuzz import fuzz
import re
import html
from typing import Literal

PAGE_SEP = '<><><><><><>NEWPAGE<><><><><><>'

def _normalize_os_newlines(s: str) -> str:
    return s.replace("\r\n", "\n").replace("\r", "\n")

def mmd_decode_page(s: str) -> str:
    # Only decode literal "\n" to real newlines; keep backslashes for LaTeX intact
    s = _normalize_os_newlines(s)
    return s.replace("\\n", "\n")

def canonicalize(s: str, *, mode: Literal["keep","flatten"]="keep") -> str:
    """
    Canonical form for comparisons on BOTH sides.

    - Normalize OS newlines
    - Decode HTML entities
    - Strip trailing spaces per line
    - Collapse ANY run of blank/whitespace-only lines to exactly ONE blank line (\n\n)
    - Trim leading/trailing blank space/newlines
    - (Optionally) flatten newlines to spaces
    """
    s = html.unescape(_normalize_os_newlines(s))

    # drop trailing spaces at line ends
    s = "\n".join(ln.rstrip() for ln in s.splitlines())

    # collapse whitespace-only lines to true empties, then collapse runs -> one blank line
    #   matches: newline + optional spaces/tabs + one-or-more newlines
    s = re.sub(r"\n[ \t]*\n+", "\n\n", s)

    # trim top/bottom whitespace (including leading or trailing blank lines)
    s = s.strip()

    if mode == "keep":
        return s
    else:
        # flatten to one paragraph
        s = re.sub(r"\s*\n\s*", " ", s)
        s = re.sub(r"\s{2,}", " ", s).strip()
        return s

def load_mmd(path: str | Path, page_sep: str = PAGE_SEP, decode: bool = True) -> list[str]:
    s = Path(path).read_text(encoding="utf-8")
    pages = s.split(page_sep)
    return [mmd_decode_page(p) for p in pages] if decode else pages

def page_similarity(text_A: str, text_B: str) -> float:
    a = canonicalize(text_A, mode="keep")
    b = canonicalize(text_B, mode="keep")
    return fuzz.partial_ratio(a, b)

# Only needed if you ever WRITE .mmd files:
def mmd_encode_page(page_text: str) -> str:
    return _normalize_os_newlines(page_text).replace("\n", "\\n")
