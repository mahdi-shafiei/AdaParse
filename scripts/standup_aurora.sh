# 1) framework + oneAPI (Aurora Python): good
module purge
module load frameworks/2025.0.0.lua
source /opt/aurora/24.347.0/oneapi/setvars.sh --force 2>/dev/null || true
# - sanitize
#unset PYTHONPATH
#export PYTHONNOUSERSITE=1
#export PYTHONDONTWRITEBYTECODE=1
#export PATH="$HOME/bin:$HOME/.local/bin:$PATH"

# 2) user site-packages append
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$HOME/.local/aurora/frameworks/2025.0.0/lib/python3.10/site-packages" # ERROR
#export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$HOME/.local/aurora/frameworks/aurora_frameworks-2025.2.0/lib/python3.10/site-packages" # seems new

# 3) shims are visible first on PATH
export PATH="$HOME/bin:$HOME/.local/bin:$PATH"
export PATH="$HOME/.local/aurora/frameworks/2025.0.0/bin:$PATH"

# 4) HuggingFace
export HF_HOME="${HF_HOME:-/lus/flare/projects/FoundEpidem/siebenschuh/HF}"

# 5) Offline: no update pings for transformers/albumentations
export NO_ALBUMENTATIONS_UPDATE="${NO_ALBUMENTATIONS_UPDATE-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE-1}"

# 6) conda environment (only daparse3 is built correctly)
conda activate adaparse10
# - sanitize
unset PYTHONPATH
export PYTHONNOUSERSITE=1

# 7) parsl interchange shim once
mkdir -p "$HOME/bin"
cat > "$HOME/bin/interchange.py" <<'PY'
#!/usr/bin/env python3
import runpy, sys
sys.exit(runpy.run_module("parsl.executors.high_throughput.interchange", run_name="__main__"))
PY
chmod +x "$HOME/bin/interchange.py"

# 8) validation
echo "=== PATH ==="
echo "$PATH"
echo "=== PYTHONPATH ==="
echo "${PYTHONPATH:-<empty>}"
which python && python --version
python -c "import torch; print('Torch:', torch.__version__, 'from', torch.__file__)"
python -c "import parsl, sys; print('Parsl version:', parsl.__version__); print('parsl file:', parsl.__file__)"
python -c "import parsl.executors.high_throughput.interchange as i; print('interchange module:', i.__file__)"
python -c "import shutil; print('interchange.py on PATH:', shutil.which('interchange.py'))"
