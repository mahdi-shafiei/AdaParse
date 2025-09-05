#!/usr/bin/env bash
set -euo pipefail

# -------------------------------------------------
#
#    LEGACY: Nougat Github Source Repo
#           (replaced w/ transformers' NougatModel)
#
# -------------------------------------------------
# download_nougat_checkpoint.sh
# Downloads a Nougat checkpoint from GitHub releases into a target directory.
# Makes the directory public/editable (a+rwX) at the end.
#
# Usage:
#   ./download_nougat_checkpoint.sh NOUGAT_CHECKPOINT [MODEL_TAG] [--force]
#
# Examples:
#   ./download_nougat_checkpoint.sh ./checkpoint
#   ./download_nougat_checkpoint.sh ./checkpoint 0.1.0-base
#   ./download_nougat_checkpoint.sh ./checkpoint --force
# -------------------------------------------------

# default is `base` model
# Source: https://github.com/facebookresearch/nougat
BASE_URL="https://github.com/facebookresearch/nougat/releases/download"
DEFAULT_MODEL_TAG="0.1.0-base"

# human-readable disk usage
human_size() {
  local bytes="$1"
  if command -v numfmt >/dev/null 2>&1; then
    numfmt --to=si --suffix=B "$bytes"    # e.g., 1.01GB
  else
    awk -v b="$bytes" 'BEGIN{
      split("B KB MB GB TB PB", u, " ");
      i=1; while (b>=1000 && i<length(u)) { b/=1000; i++ }
      printf("%.2f%s", b, u[i])
    }'
  fi
}


usage() {
  cat <<USAGE
Usage: $0 NOUGAT_CHECKPOINT [MODEL_TAG] [--force]

Arguments:
  NOUGAT_CHECKPOINT   Directory to save checkpoint files (required)
  MODEL_TAG           Release tag (default: ${DEFAULT_MODEL_TAG})
  --force             Re-download even if a file already exists

Files downloaded from: ${BASE_URL}/<MODEL_TAG>/
USAGE
}

# Download a single file with progress + sanity check
download_one() {
  local url="$1" name="$2" dest="$3" force="$4"

  if [[ -f "${dest}/${name}" && "$force" != "1" ]]; then
    echo "↪ Skipping ${name} (exists). Use --force to re-download."
    return 0
  fi

  echo "↓ Downloading ${name}"
  mkdir -p "$dest"
  curl -fL --progress-bar -o "${dest}/${name}.part" "$url"
  mv "${dest}/${name}.part" "${dest}/${name}"

  # Sanity check: file should be larger than 15 bytes
  local size
  size=$(wc -c < "${dest}/${name}")
  if [[ "$size" -le 15 ]]; then
    echo "ERROR: ${name} too small (${size} bytes) — download likely failed." >&2
    rm -f "${dest}/${name}"
    return 1
  fi
  local pretty; pretty=$(human_size "$size")
  echo "✓ ${name} (${pretty})"
}

main() {
  # Help
  if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage; exit 0
  fi

  # Args
  if [[ $# -lt 1 ]]; then
    usage; exit 1
  fi
  local dest="$1"
  shift || true

  local model_tag="${DEFAULT_MODEL_TAG}"
  local force="0"

  # Parse optional args: [MODEL_TAG] [--force]
  for arg in "$@"; do
    if [[ "$arg" == "--force" ]]; then
      force="1"
    else
      model_tag="$arg"
    fi
  done

  echo "Downloading Nougat checkpoint:"
  echo "  dest      : $dest"
  echo "  model_tag : $model_tag"
  echo "  force     : $([[ "$force" == "1" ]] && echo yes || echo no)"
  echo

  mkdir -p "$dest"

  # Release file list (typical HF-style assets)
  local files=(
    "config.json"
    "pytorch_model.bin"
    "special_tokens_map.json"
    "tokenizer.json"
    "tokenizer_config.json"
  )

  local failures=0
  for f in "${files[@]}"; do
    url="${BASE_URL}/${model_tag}/${f}"
    if ! download_one "$url" "$f" "$dest" "$force"; then
      ((failures+=1))
    fi
  done

  # Make directory public/editable
  echo
  echo "Setting public read/write permissions on: $dest"
  chmod -R a+rwX "$dest"

  if [[ "$failures" -gt 0 ]]; then
    echo "Completed with ${failures} failed file(s)." >&2
    exit 2
  fi
  echo " - - Download complete. Files saved to: $dest"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  main "$@"
fi
