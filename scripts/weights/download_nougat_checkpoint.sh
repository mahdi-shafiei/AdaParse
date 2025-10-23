#!/usr/bin/env bash
# Prefetch the Hugging Face Nougat checkpoint into a local directory
#
# Usage:
#   download_nougat_checkpoint.sh <TARGET_DIR> [REVISION]
#
# Env overrides:
#   REPO_ID            (default: facebook/nougat-base)
#   QUIET=1            (less chatty; hides progress bars)
#   HUGGINGFACE_TOKEN  (optional; inherited by CLI if set)
#
# Notes:
# - Avoids ~/.local/bin/hf.
# - Prefers 'huggingface-cli'; falls back to 'python -m huggingface_hub.cli.hf'.
# - Forces ONLINE just for this command; resumes partial downloads.
# - Uses a per-target local cache to reduce lock chatter.

set -Eeuo pipefail

REPO_ID="${REPO_ID:-facebook/nougat-base}"
TARGET_DIR="${1:-}"
REVISION="${2:-main}"
QUIET="${QUIET:-0}"

if [[ -z "${TARGET_DIR}" ]]; then
  echo "ERROR: TARGET_DIR argument is required."
  echo "Usage: $0 <TARGET_DIR> [REVISION]"
  exit 2
fi

mkdir -p "${TARGET_DIR}"

# --- Check Availability ---
PYTHON_BIN="${PYTHON_BIN:-$(command -v python || command -v python3 || echo python)}"
if ! command -v huggingface-cli >/dev/null 2>&1 && \
   ! "${PYTHON_BIN}" -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('huggingface_hub') else 1)"; then
  echo "[ERR] Neither 'huggingface-cli' nor 'huggingface_hub' is available in ${PYTHON_BIN}."
  echo "      Load module/conda env & try: ${PYTHON_BIN} -m pip install -U 'huggingface_hub[cli]'"
  exit 3
fi

# --------------------------------------------------------------------
# Pick a downloader that does NOT use ~/.local/bin/hf
# 1) Prefer 'huggingface-cli' (worked for you)
# 2) Else use current Python's module CLI: python -m huggingface_hub.cli.hf
# --------------------------------------------------------------------
DOWNLOADER=()
if command -v huggingface-cli >/dev/null 2>&1; then
  DOWNLOADER=(huggingface-cli download)
else
  PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
  if "${PYTHON_BIN}" -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('huggingface_hub') else 1)"; then
    DOWNLOADER=("${PYTHON_BIN}" -m huggingface_hub.cli.hf download)
  else
    echo "ERROR: Neither 'huggingface-cli' nor 'huggingface_hub' (in current Python) is available."
    echo "Install one of:"
    echo "  ${PYTHON_BIN} -m pip install -U huggingface_hub        # module CLI"
    echo "  ${PYTHON_BIN} -m pip install -U 'huggingface_hub[cli]'  # also installs 'huggingface-cli'"
    exit 3
  fi
fi

# Optional niceties (local cache, quieter output, no telemetry) to reduce lock chatter
export HUGGINGFACE_HUB_CACHE="${TARGET_DIR}/.cache/huggingface"
mkdir -p "${HUGGINGFACE_HUB_CACHE}"
{ echo "# cache"; echo "*"; } > "${HUGGINGFACE_HUB_CACHE}/.gitignore" 2>/dev/null || true
export HF_HUB_DISABLE_TELEMETRY=1
if [[ "${QUIET}" == "1" ]]; then
  export HF_HUB_DISABLE_PROGRESS_BARS=1
fi

# Non-blocking lock to avoid races in multi-job runs
exec 9>"${TARGET_DIR}/.download.lock" || true
if ! flock -n 9; then
  [[ "${QUIET}" == "1" ]] || echo "[HF] Another download is in progress for ${TARGET_DIR}; skipping."
  exit 0
fi

need_download() {
  # require config + (safetensors or bin) + tokenizer.json + preprocessor_config.json
  [[ -f "${TARGET_DIR}/config.json" ]] && \
  { [[ -f "${TARGET_DIR}/model.safetensors" ]] || [[ -f "${TARGET_DIR}/pytorch_model.bin" ]]; } && \
  [[ -f "${TARGET_DIR}/tokenizer.json" ]] && \
  [[ -f "${TARGET_DIR}/preprocessor_config.json" ]]
}

if need_download; then
  [[ "${QUIET}" == "1" ]] || echo "[HF] Already present: ${TARGET_DIR} (skipping)"
  exit 0
fi

[[ "${QUIET}" == "1" ]] || echo "[HF] Prefetching ${REPO_ID}@${REVISION} → ${TARGET_DIR}"

# Force ONLINE just for this command; resume if partial; (no deprecated symlink flag)
TRANSFORMERS_OFFLINE=0 HF_HUB_OFFLINE=0 \
"${DOWNLOADER[@]}" "${REPO_ID}" \
  --revision "${REVISION}" \
  --local-dir "${TARGET_DIR}" \
  --resume-download \
  ${QUIET:+--quiet}

# Basic verification
if ! need_download; then
  echo "ERROR: Download appears incomplete in ${TARGET_DIR}."
  exit 4
fi

# Stamp metadata
{
  echo "repo=${REPO_ID}"
  echo "revision=${REVISION}"
  echo "fetched_at=$(date -Iseconds)"
} > "${TARGET_DIR}/.fetched"

[[ "${QUIET}" == "1" ]] || echo "[HF] Done. Checkpoint ready at: ${TARGET_DIR}"
