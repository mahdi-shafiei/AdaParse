
import re
import os
from pathlib import Path

# {REPO}/.adaparse.env
ENV_BASENAME = ".adaparse.env"
_ASSIGN = re.compile(r"^\s*(?:export\s+)?([A-Za-z_]\w*)\s*=\s*(.*)\s*$")

def load_env_file(path: Path) -> None:
    """Load environment variables from .env file."""
    path = Path(path)
    if not path or not path.exists():
        return

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _ASSIGN.match(line)
        if not m:
            continue
        k, v = m.group(1), m.group(2)
        # Handle quoted values
        if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
            v = v[1:-1]
        else:
            # Strip trailing inline comment on unquoted values
            if " #" in v:
                v = v.split(" #", 1)[0].rstrip()
        os.environ[k] = os.path.expanduser(os.path.expandvars(v))

def find_env_file(start: Path) -> Path | None:
    """Find .adaparse.env file by searching upwards from start directory."""
    start = Path(start)
    p = start.resolve()
    while True:
        cand = p / ENV_BASENAME
        if cand.exists():
            return cand
        if p.parent == p:
            return None
        p = p.parent

def get_nougat_checkpoint() -> Path | None:
    """Get Nougat checkpoint path from environment."""
    # Load env file from parent directory of tests
    test_dir = Path(__file__).parent
    env_path = find_env_file(test_dir)
    if env_path:
        load_env_file(env_path)

    checkpoint_path = os.environ.get('NOUGAT_CHECKPOINT')
    if checkpoint_path:
        return Path(checkpoint_path)
    return None

def get_adaparse_checkpoint() -> Path | None:
    """Get AdaParse (regressor) checkpoint path from environment."""
    # Load env file from parent directory of tests
    test_dir = Path(__file__).parent
    env_path = find_env_file(test_dir)
    if env_path:
        load_env_file(env_path)

    checkpoint_path = os.environ.get('ADAPARSE_CHECKPOINT')
    if checkpoint_path:
        return Path(checkpoint_path)
    return None
