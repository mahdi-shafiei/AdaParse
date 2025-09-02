import os
import time
import logging
from typing import Tuple, List

import pytest
import requests

RUN = os.getenv("ADAPARSE_RUN_ONLINE_TESTS") == "1"
pytestmark = pytest.mark.skipif(
    not RUN, reason="set ADAPARSE_RUN_ONLINE_TESTS=1 to run online checks"
)

# What to check (name, url)
CHECKS: List[Tuple[str, str]] = [
    ("nougat repo", "https://github.com/facebookresearch/nougat"),
    ("nougat releases (landing)", "https://github.com/facebookresearch/nougat/releases"),
    ("nougat releases/download root", "https://github.com/facebookresearch/nougat/releases/"),
    ("nougat base-0.1.0 model", "https://github.com/facebookresearch/nougat/releases/tag/0.1.0-base"),
    ("Swin org", "https://github.com/SwinTransformer"),
    ("Swin storage releases", "https://github.com/SwinTransformer/storage/releases"),
    ("Swin 22k->1k weight",
     "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22kto1k.pth"),
]

def _fetch(url: str, timeout_head=8, timeout_get=20):
    """HEAD with redirect follow; fallback to GET(stream) if HEAD blocked (405/403/4xx)."""
    time.sleep(1.0)
    try:
        r = requests.head(url, allow_redirects=True, timeout=timeout_head)
        if r.status_code >= 400 or r.status_code in (403, 405):
            r = requests.get(url, allow_redirects=True, timeout=timeout_get, stream=True)
        return r
    except requests.RequestException as e:
        pytest.fail(f"request failed for {url}: {e}")

@pytest.mark.online
@pytest.mark.parametrize("name,url", CHECKS, ids=[c[0] for c in CHECKS])
def test_online_resource_available(name: str, url: str):
    log = logging.getLogger(__name__)
    log.info("Checking: %s → %s", name, url)
    r = _fetch(url)
    log.info("Result: %s → HTTP %s (final_url=%s)", name, r.status_code, r.url)

    # Expect reachable resource (200-399). GitHub often returns 200/301/302 to final asset.
    assert 200 <= r.status_code < 400, f"{name}: unexpected HTTP {r.status_code} for {url}"

    # If this is the .pth weight file, sanity-check size header when available.
    if url.endswith(".pth"):
        size_hdr = r.headers.get("Content-Length")
        if size_hdr and size_hdr.isdigit():
            size = int(size_hdr)
            log.info("%s Content-Length=%d bytes", name, size)
            assert size > 100_000, f"{name}: suspiciously small ({size} bytes)"
        else:
            # Some CDNs omit Content-Length on HEAD; best we can do is note it.
            log.info("%s Content-Length missing (CDN/streamed response)", name)
