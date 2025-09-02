#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------
# download_swin_weights.sh
# Downloads Swin-B weights (22k→1k, 384) into a target directory.
# Makes the directory public/editable (a+rwX) at the end.
#
# Meaning:
#   SwinTransformer is instantiated via `timm` pre-trained weights.
#   Then, the weights are modified for SwinEncoder from Nougat checkpoint.
#
# Usage:
#   ./download_swin_weights.sh DEST_DIR [--force] [--url <URL>] [--name <FILENAME>]
#
# Example:
#   ./download_swin_weights.sh ./checkpoint
#   ./download_swin_weights.sh ./checkpoint --force
# ---------------------------------------------

# equivalent weights to timm's 0.5.4 and hence Nougat's: timm.create_model('swin_base_patch4_window12_384`)
DEFAULT_URL="https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22kto1k.pth"
DEFAULT_NAME="swin_base_patch4_window12_384_22kto1k.pth"

# human-readable disk usage
human_size() {
  local bytes="$1"
  if command -v numfmt >/dev/null 2>&1; then
    numfmt --to=si --suffix=B "$bytes"
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
Usage: $0 DEST_DIR [--force] [--url <URL>] [--name <FILENAME>]

Arguments:
  DEST_DIR          Directory to save the .pth file (required)
  --force           Re-download even if the file already exists
  --url <URL>       Override weights URL (default: ${DEFAULT_URL})
  --name <FILENAME> Override output filename (default: ${DEFAULT_NAME})

Downloads from the Swin official storage (GitHub releases).
USAGE
}

download_one() {
  local url="$1" name="$2" dest="$3" force="$4"

  mkdir -p "$dest"

  if [[ -f "${dest}/${name}" && "$force" != "1" ]]; then
    echo "↪ Skipping ${name} (exists). Use --force to re-download."
    return 0
  fi

  echo "↓ Downloading ${name}"
  curl -fL --progress-bar -o "${dest}/${name}.part" "$url"
  mv "${dest}/${name}.part" "${dest}/${name}"

  local size
  size=$(wc -c < "${dest}/${name}")
  if [[ "$size" -le 1024 ]]; then
    echo "ERROR: ${name} too small (${size} bytes) — download likely failed." >&2
    rm -f "${dest}/${name}"
    return 1
  fi
  local pretty; pretty=$(human_size "$size")
  echo "✓ ${name} (${pretty})"
}

main() {
  if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage; exit 0
  fi
  if [[ $# -lt 1 ]]; then
    usage; exit 1
  fi

  local dest="$1"; shift || true
  local force="0"
  local url="${DEFAULT_URL}"
  local name="${DEFAULT_NAME}"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --force) force="1"; shift ;;
      --url)   url="${2:-}"; shift 2 ;;
      --name)  name="${2:-}"; shift 2 ;;
      *) echo "Unknown arg: $1"; usage; exit 1 ;;
    esac
  done

  echo "Downloading Swin weights:"
  echo "  dest  : $dest"
  echo "  url   : $url"
  echo "  name  : $name"
  echo "  force : $([[ "$force" == "1" ]] && echo yes || echo no)"
  echo

  local failures=0
  if ! download_one "$url" "$name" "$dest" "$force"; then
    ((failures+=1))
  fi

  echo
  echo "Setting public read/write permissions on: $dest"
  chmod -R a+rwX "$dest" || true

  if [[ "$failures" -gt 0 ]]; then
    echo "Completed with ${failures} failed download(s)." >&2
    exit 2
  fi
  echo " - - Download complete. File saved to: $dest/$name"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  main "$@"
fi
