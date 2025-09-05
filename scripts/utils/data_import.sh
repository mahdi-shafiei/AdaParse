#!/usr/bin/env bash
set -Eeuo pipefail
# data_import.sh
# Usage: data_import.sh --archive /path/to/archive --input-dir /path/to/input

ARCHIVE=""
INPUT_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --archive) ARCHIVE="${2:-}"; shift 2;;
    --input-dir) INPUT_DIR="${2:-}"; shift 2;;
    *) echo "Unknown arg: $1" >&2; exit 2;;
  esac
done

[[ -n "$INPUT_DIR" ]] || { echo "ERROR: --input-dir required" >&2; exit 2; }

echo "=== DATA IMPORT ==="
if [[ -z "$ARCHIVE" || ! -f "$ARCHIVE" ]]; then
  echo "[INFO] Archive not found or not provided: ${ARCHIVE:-<none>} (skipping)"
  exit 0
fi

mkdir -p "$INPUT_DIR"
TMP_EXTRACT="${INPUT_DIR}/.extract_tmp"
rm -rf "$TMP_EXTRACT"
mkdir -p "$TMP_EXTRACT"

echo "[INFO] Extracting: $ARCHIVE"
case "$ARCHIVE" in
  *.tar.gz|*.tgz) tar -xzf "$ARCHIVE" -C "$TMP_EXTRACT" ;;
  *.tar)          tar -xf  "$ARCHIVE" -C "$TMP_EXTRACT" ;;
  *.zip)
    if command -v unzip >/dev/null 2>&1; then unzip -q "$ARCHIVE" -d "$TMP_EXTRACT"
    else echo "ERROR: 'unzip' not found; cannot extract $ARCHIVE" >&2; exit 1; fi ;;
  *) echo "ERROR: Unknown archive type: $ARCHIVE" >&2; exit 1 ;;
esac

echo "[INFO] Moving PDFs into: $INPUT_DIR"
MOVED=0
while IFS= read -r -d '' pdf; do
  mv -n "$pdf" "$INPUT_DIR"/ && ((MOVED+=1)) || true
done < <(find "$TMP_EXTRACT" -type f \( -iname '*.pdf' -o -iname '*.PDF' \) -print0)

rm -rf "$TMP_EXTRACT"
find "$INPUT_DIR" -type f ! -iname '*.pdf' -delete || true

COUNT_PDF=$(find "$INPUT_DIR" -maxdepth 1 -type f -iname '*.pdf' | wc -l | tr -d ' ')
echo "[INFO] PDFs moved: $MOVED"
echo "[INFO] PDFs available in input: $COUNT_PDF"
echo
