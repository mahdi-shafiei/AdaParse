# --------------------------------------------------------
# timm-0.5.4 
# timm/models/layers/trace_utils.py
# --------------------------------------------------------
try:
    from torch import _assert
except ImportError:
    def _assert(condition: bool, message: str):
        assert condition, message

