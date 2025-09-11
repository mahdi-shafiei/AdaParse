from __future__ import annotations

from pathlib import Path
from typing import Optional, Union
import pymupdf

def _is_closed(doc) -> bool:
    """Robustly detect a closed PyMuPDF Document across versions."""
    if doc is None:
        return True
    # PyMuPDF exposes isClosed (camelCase). Some builds may also expose is_closed.
    return bool(getattr(doc, "isClosed", getattr(doc, "is_closed", False)))


def safe_doc_open(
    path: Union[str, Path],
    logger: Optional[object] = None,
    **open_kwargs,
):
    """
    Safely open a PDF with PyMuPDF.
    - Returns a Document on success, or None on failure.
    - Never raises if the file is missing/unreadable—logs instead.

    Example:
        doc = safe_doc_open(pdf_path, logger)
        if doc is None:
            ...  # fallback
    """
    p = Path(path) if not isinstance(path, Path) else path
    if not p or not p.exists() or not p.is_file():
        if logger:
            logger.warning(f"safe_doc_open: not a readable file: {p}")
        return None
    try:
        doc = pymupdf.open(p.as_posix(), **open_kwargs)
        return doc
    except Exception as e:
        if logger:
            logger.warning(f"safe_doc_open: failed to open {p}: {e}")
        return None


def safe_doc_close(
    doc,
    logger: Optional[object] = None,
) -> bool:
    """
    Close PyMuPDF document (idempotent).
    - Returns True if we closed it in this call.
    - Returns False if it was already closed or doc is None.
    - Never raises; logs instead.
    """
    if doc is None:
        return False
    if _is_closed(doc):
        return False
    try:
        doc.close()
        return True
    except ValueError:
        # Typically "document closed" — treat as already-closed.
        if logger:
            logger.debug("safe_doc_close(): document already closed; ignoring.")
        return False
    except Exception as e:
        if logger:
            logger.warning(f"safe_doc_close(): error while closing document: {e}")
        return False
