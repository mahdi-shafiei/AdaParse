# tests/test_nougat.py
from __future__ import annotations

import sys
from pathlib import Path
import pytest

pytestmark = pytest.mark.fs  # run with:  -m fs

def test_wrapper_dry_run_invokes_heavy_script_path(tmp_path, capsys):
    # 1) import the wrapper module that lives in tests/ as provided
    #    (this is your CLI wrapper, not a pytest file)
    from tests import test_nougat_wrapper as wrapper

    # 2) create a tiny config file that exists
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("parser_settings:\n  name: nougat\n", encoding="utf-8")

    # 3) ensure the expected heavy script exists NEXT TO THE WRAPPER MODULE
    #    because the wrapper uses: Path(__file__).with_name(HEAVY_SCRIPT)
    wrapper_dir = Path(wrapper.__file__).resolve().parent
    heavy_path = wrapper_dir / wrapper.HEAVY_SCRIPT
    if not heavy_path.exists():
        heavy_path.write_text(
            "import argparse\n"
            "ap = argparse.ArgumentParser()\n"
            "ap.add_argument('--config_path', required=True)\n"
            "ap.parse_args()\n"
            "print('heavy stub ok')\n",
            encoding="utf-8",
        )

    # 4) call wrapper.main in dry-run mode with our config
    rc = wrapper.main(["--dry-run", "--verbose", "--config", str(cfg)])
    out = capsys.readouterr().out

    # 5) assertions: dry-run should succeed and print the command with our cfg
    assert rc == 0
    assert "cmd:" in out
    assert str(cfg) in out
    # bonus: it should have pointed at the heavy script we just wrote
    assert str(heavy_path) in out or heavy_path.name in out
