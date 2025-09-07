from __future__ import annotations

import os, re, sys, argparse, subprocess
from pathlib import Path

HEAVY_SCRIPT = "nougat_model_minimal.py"
ENV_BASENAME = ".adaparse.env"
_ASSIGN = re.compile(r"^\s*(?:export\s+)?([A-Za-z_]\w*)\s*=\s*(.*)\s*$")

def load_env_file(path: Path) -> None:
    """Minimal, quiet .env loader (KEY=VALUE lines)."""
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
        if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
            v = v[1:-1]
        else:
            # strip trailing inline comment on unquoted values
            if " #" in v:
                v = v.split(" #", 1)[0].rstrip()
        os.environ[k] = os.path.expanduser(os.path.expandvars(v))

def find_env_file(start: Path) -> Path | None:
    p = start.resolve()
    while True:
        cand = p / ENV_BASENAME
        if cand.exists():
            return cand
        if p.parent == p:
            return None
        p = p.parent

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Run Nougat MVP using NOUGAT_CONFIG_YAML from .adaparse.env"
    )
    ap.add_argument("--env-file", type=Path, help=f"path to {ENV_BASENAME} (default: search upwards)")
    ap.add_argument("--config", "-c", type=Path, help="override NOUGAT_CONFIG_YAML")
    ap.add_argument("--dry-run", action="store_true", help="print the command and exit")
    ap.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    args, passthrough = ap.parse_known_args(argv)

    # Load env quietly
    env_path = args.env_file or find_env_file(Path.cwd())
    if env_path:
        load_env_file(env_path)

    # Resolve config (CLI override > env)
    cfg_path = Path(args.config) if args.config else Path(os.environ.get("NOUGAT_CONFIG_YAML", ""))
    if not cfg_path:
        print("[error] NOUGAT_CONFIG_YAML not set and no --config provided.", file=sys.stderr)
        return 2
    if not cfg_path.exists():
        print(f"[error] Config not found: {cfg_path}", file=sys.stderr)
        return 2

    # Find the heavy script next to this wrapper
    script_path = Path(__file__).with_name(HEAVY_SCRIPT)
    if not script_path.exists():
        print(f"[error] cannot find {HEAVY_SCRIPT} next to {Path(__file__).name}", file=sys.stderr)
        return 2

    cmd = [sys.executable, str(script_path), "--config_path", str(cfg_path), *passthrough]

    if args.verbose or args.dry_run:
        print("cmd:", " ".join(map(str, cmd)))
        if env_path and args.verbose:
            print(f"env-file: {env_path}")

    if args.dry_run:
        return 0

    rc = subprocess.run(cmd, check=False).returncode
    if rc != 0:
        print(f"[error] {HEAVY_SCRIPT} exited with code {rc}", file=sys.stderr)
    return rc

if __name__ == "__main__":
    raise SystemExit(main())
