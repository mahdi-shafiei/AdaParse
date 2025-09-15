from __future__ import annotations
from pathlib import Path
import json
from typing import Iterable, Sequence, Mapping, Any

def write_jsonl_from_tuples(
    items: Iterable[Sequence[Any] | Mapping[str, Any]],
    out_path: str | Path,
    *,
    keys: Sequence[str] | None = None,
    ensure_ascii: bool = False,
) -> Path:
    """
    Write items to a JSONL file. Each line is a JSON object.
    - If an item is a dict, it's written as-is.
    - If an item is a tuple/list, it will be zipped with `keys` (if provided),
      otherwise auto-keyed as col1, col2, ... in order.

    Example:
        model_doc_outputs = [
            ("English", "M. C. Curthoys and H. S. Jones argue ..."),
            ("Spanish", "A lo largo del siglo XVIII, ..."),
        ]
        write_jsonl_from_tuples(model_doc_outputs, "model_doc_outputs.jsonl",
                                keys=["lang", "text"])
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for item in items:
            if isinstance(item, Mapping):
                obj = dict(item)  # already a dict
            else:
                # sequence/tuple → dict
                if keys is not None:
                    if len(keys) != len(item):
                        raise ValueError(
                            f"Tuple length {len(item)} != keys length {len(keys)}"
                        )
                    obj = dict(zip(keys, item))
                else:
                    obj = {f"col{i+1}": v for i, v in enumerate(item)}
            f.write(json.dumps(obj, ensure_ascii=ensure_ascii) + "\n")

    return out_path
