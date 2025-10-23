#!/usr/bin/env bash
# Prefetch multiple Hugging Face AdaParse checkpoints into a root directory.
#
# Usage:
#   scripts/weights/download_all_adaparse_prediction_models.sh [<TARGET_ROOT>] [REVISION]
#
# TARGET_ROOT resolution order:
#   1) If <TARGET_ROOT> arg provided → use it
#   2) Else, if $ADAPARSE_PROJECT_ROOT set (from .adaparse.env) → "$ADAPARSE_PROJECT_ROOT/prediction"
#   3) Else → error
#
# Env:
#   REPOS="7shoe/adaparse-scibert-uncased 7shoe/adaparse-specter-docwise 7shoe/adaparse-specter-pagewise"
#   QUIET=1
#   HUGGINGFACE_TOKEN=...
#   PYTHON_BIN=python3

set -Eeuo pipefail

# --- locate repo root and .adaparse.env (script is under ./scripts/weights/) ---
# Resolve script dir (portable)
get_script_dir() {
  local src="${BASH_SOURCE[0]}"
  while [[ -h "$src" ]]; do
    local dir
    dir="$(cd -P "$(dirname "$src")" && pwd)"
    src="$(readlink "$src")"
    [[ "$src" != /* ]] && src="$dir/$src"
  done
  cd -P "$(dirname "$src")" && pwd
}
SCRIPT_DIR="$(get_script_dir)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Optional: allow override of env file path via ADAPARSE_ENV_FILE
ENV_FILE="${ADAPARSE_ENV_FILE:-${REPO_ROOT}/.adaparse.env}"
if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck source=/dev/null
  . "${ENV_FILE}"
  set +a
fi

# --- resolve target root ---
_arg_root="${1:-}"
REVISION="${2:-main}"
QUIET="${QUIET:-0}"

if [[ -n "${_arg_root}" ]]; then
  TARGET_ROOT="${_arg_root}"
elif [[ -n "${ADAPARSE_PROJECT_ROOT:-}" ]]; then
  TARGET_ROOT="${ADAPARSE_PROJECT_ROOT%/}/prediction"
else
  echo "ERROR: Need <TARGET_ROOT> or ADAPARSE_PROJECT_ROOT in ${ENV_FILE}." >&2
  echo "Usage: $0 [<TARGET_ROOT>] [REVISION]" >&2
  exit 2
fi

# Default repos
REPOS="${REPOS:-7shoe/adaparse-scibert-uncased 7shoe/adaparse-specter-docwise 7shoe/adaparse-specter-pagewise}"

mkdir -p "${TARGET_ROOT}"
[[ "${QUIET}" == "1" ]] || { echo "=== PREDICTION ROOT ==="; echo "${TARGET_ROOT}"; echo; }

# -----------------------------------------------------------
# Pick a downloader (prefer 'huggingface-cli', else module CLI)
# -----------------------------------------------------------
DOWNLOADER=()
if command -v huggingface-cli >/dev/null 2>&1; then
  DOWNLOADER=(huggingface-cli download)
else
  PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
  if "${PYTHON_BIN}" -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('huggingface_hub') else 1)"; then
    DOWNLOADER=("${PYTHON_BIN}" -m huggingface_hub.cli.hf download)
  else
    echo "ERROR: Need 'huggingface-cli' or 'huggingface_hub' in current Python." >&2
    echo "Try: ${PYTHON_BIN} -m pip install -U 'huggingface_hub[cli]'" >&2
    exit 3
  fi
fi

# Helper: does a target dir already have a working checkpoint?
need_download() {
  local dir="$1"
  [[ -f "${dir}/config.json" ]] && \
  { [[ -f "${dir}/model.safetensors" ]] || [[ -f "${dir}/pytorch_model.bin" ]] || [[ -f "${dir}/onnx/model.onnx" ]]; } && \
  { [[ -f "${dir}/tokenizer.json" ]] || [[ -f "${dir}/vocab.txt" ]] || [[ -f "${dir}/sentencepiece.bpe.model" ]] || [[ -f "${dir}/spiece.model" ]]; }
}

# Quiet/telemetry
export HF_HUB_DISABLE_TELEMETRY=1
[[ "${QUIET}" == "1" ]] && export HF_HUB_DISABLE_PROGRESS_BARS=1
status() { [[ "${QUIET}" == "1" ]] || echo "$@"; }

rc=0
for REPO in ${REPOS}; do
  SUBDIR_NAME="${REPO//\//__}"
  TARGET_DIR="${TARGET_ROOT}/${SUBDIR_NAME}"
  mkdir -p "${TARGET_DIR}"

  # Per-repo cache
  export HUGGINGFACE_HUB_CACHE="${TARGET_DIR}/.cache/huggingface"
  mkdir -p "${HUGGINGFACE_HUB_CACHE}"
  { echo "# cache"; echo "*"; } > "${HUGGINGFACE_HUB_CACHE}/.gitignore" 2>/dev/null || true

  # Per-repo non-blocking lock
  exec 9>"${TARGET_DIR}/.download.lock" || true
  if ! flock -n 9; then
    status "[HF] Another download is in progress for ${TARGET_DIR}; skipping."
    continue
  fi

  if need_download "${TARGET_DIR}"; then
    status "[HF] Already present: ${TARGET_DIR} (skipping)"
    continue
  fi

  status "[HF] Prefetching ${REPO}@${REVISION} → ${TARGET_DIR}"

  # Force online; resume partial downloads
  if ! TRANSFORMERS_OFFLINE=0 HF_HUB_OFFLINE=0 \
    "${DOWNLOADER[@]}" "${REPO}" \
      --revision "${REVISION}" \
      --local-dir "${TARGET_DIR}" \
      --resume-download \
      ${QUIET:+--quiet}; then
    echo "ERROR: Downloader failed for ${REPO}" >&2
    rc=4
    continue
  fi

  if ! need_download "${TARGET_DIR}"; then
    echo "ERROR: Download appears incomplete in ${TARGET_DIR}." >&2
    rc=4
    continue
  fi

  {
    echo "repo=${REPO}"
    echo "revision=${REVISION}"
    echo "fetched_at=$(date -Iseconds)"
  } > "${TARGET_DIR}/.fetched"

  status "[HF] Done: ${TARGET_DIR}"
done

exit "${rc}"

