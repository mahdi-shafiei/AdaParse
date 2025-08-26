#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------
# setup_adaparse_project.sh
# Creates project dirs, (Aurora) unpacks sample PDFs,
# and renders a config YAML from a template by replacing tokens.
#
# Supported machines: polaris, sophia, aurora, lambda, local
# Script location: /AdaParse/scripts/setup_adaparse_project.sh
# PDF tarball:     /AdaParse/data/twenty_test_pdfs.tar.gz
# Template input:  /AdaParse/templates/aurora/canvas.yaml
# ---------------------------------------------

# Defaults
MACHINE="local"
USER_NAME=""
PROJECT_NAME=""
BASE_PATH=""

# Github Repo
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"   # -> /AdaParse
# Parser config template
TEMPLATE_PATH="${REPO_ROOT}/templates/aurora/canvas.yaml"
# Sample pdfs
ARCHIVE_LOCAL="${REPO_ROOT}/data/twenty_test_pdfs.tar.gz"
# Machine configs
ENV_MAP_PATH="${REPO_ROOT}/configs/machines/envs.yaml"
ABS_STANDUP_AURORA="${REPO_ROOT}/scripts/standup_aurora.sh"

# Helpers
die()  { echo "ERROR: $*" >&2; exit 1; }
note() { echo "[INFO] $*"; }
esc_sed() { printf '%s' "$1" | sed -e 's/[\/&|]/\\&/g'; }
yaml_get_value() {
  # $1 = file, $2 = key
  awk -v k="$2" '
    $0 ~ "^[[:space:]]*"k":[[:space:]]*" {
      sub(/^[^:]*:[[:space:]]*/, "", $0); print; exit
    }' "$1"
}


usage() {
  cat <<'USAGE'
Usage:
  setup_adaparse_project.sh \
    --user_name <name> \
    [--machine polaris|sophia|aurora|lambda|local] \
    [--project_name <proj>] \
    [--base_path <abs path>]

Notes:
--user_name is required.
--project_name is required for: polaris, sophia, aurora (not required for lambda/local).
--base_path is auto-set for polaris/sophia/aurora/lambda.
  For local, you MUST provide --base_path.

Paths (fixed):
- Script   : /AdaParse/scripts/setup_adaparse_project.sh
- PDFs tar : /AdaParse/data/twenty_test_pdfs.tar.gz
- Template : /AdaParse/configs/templates/aurora/canvas.yaml
USAGE
}

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --machine) MACHINE="${2:-}"; shift 2 ;;
    --user_name) USER_NAME="${2:-}"; shift 2 ;;
    --project_name) PROJECT_NAME="${2:-}"; shift 2 ;;
    --base_path) BASE_PATH="${2:-}"; shift 2 ;;
    *) die "Unknown argument: $1 (use --help)";;
  esac
done

# ---------------------------------------------
# Auto-detect machine if not provided
# ---------------------------------------------
if [[ -z "${MACHINE:-}" || "$MACHINE" == "local" ]]; then
  echo "=== MACHINE ==="
  HOST=$(hostname)
  if [[ "$HOST" == aurora-uan-* ]]; then
    MACHINE="aurora"
    echo "[INFO] ALCF AURORA is inferred machine"
  elif [[ "$HOST" =~ ^lambda[0-9]+$ ]]; then
    MACHINE="lambda"
    echo "[INFO] CELS LAMBDA is inferred machine"
  elif [[ "$HOST" == polaris* ]]; then
    MACHINE="polaris"
    echo "[INFO] ALCF POLARIS is inferred machine"
  elif [[ "$HOST" == sophia* ]]; then
    MACHINE="sophia"
    echo "[INFO] ALCF SOPHIA is inferred machine"
  else
    MACHINE="local"
    echo "[INFO] No known pattern matched → assuming LOCAL machine"
  fi
fi
echo ""

[[ -z "$USER_NAME" ]] && { usage; die "--user_name is required."; }

case "$MACHINE" in
  polaris|sophia|aurora|lambda|local) ;;
  *) die "--machine must be one of: polaris, sophia, aurora, lambda, local";;
esac

# Auto base_path
if [[ -z "$BASE_PATH" ]]; then
  case "$MACHINE" in
    polaris|sophia) BASE_PATH="/eagle/projects" ;;
    aurora)         BASE_PATH="/lus/flare/projects" ;;
    lambda)         BASE_PATH="/homes" ;;
    local)          : ;;  # must be provided
  esac
fi
[[ "$MACHINE" == "local" && -z "$BASE_PATH" ]] && { usage; die "--base_path is required when --machine=local."; }

# Enforce project for the big clusters
if [[ "$MACHINE" =~ ^(polaris|sophia|aurora)$ ]] && [[ -z "$PROJECT_NAME" ]]; then
  usage; die "--project_name is required for machine=$MACHINE."
fi

# Compute project_path
if [[ -n "$PROJECT_NAME" ]]; then
  PROJECT_PATH="${BASE_PATH%/}/${PROJECT_NAME}/${USER_NAME}/adaparse_data"
else
  PROJECT_PATH="${BASE_PATH%/}/${USER_NAME}/adaparse_data"
fi

# ---------------------------------------------
# SECTION: SCHEDULER OPTIONS (machine-aware)
# ---------------------------------------------
echo "=== SCHEDULER OPTIONS ==="
case "$MACHINE" in
  aurora)         SCHEDULER_OPTIONS='#PBS -l filesystems=home:flare' ;;
  polaris|sophia) SCHEDULER_OPTIONS='#PBS -l filesystems=home:eagle' ;;
  lambda|local)   SCHEDULER_OPTIONS='' ;;
  *)              SCHEDULER_OPTIONS='' ;;
esac
echo "[INFO] scheduler_options resolved: ${SCHEDULER_OPTIONS:-<empty>}"
echo

# -------------------------------
# SECTION: ARGUMENT SUMMARY
# -------------------------------
echo "=== ARGUMENTS ==="
echo "machine      : $MACHINE"
echo "user_name    : $USER_NAME"
echo "project_name : ${PROJECT_NAME:-<none>}"
echo "base_path    : $BASE_PATH"
echo "project_path : $PROJECT_PATH"
echo

# -------------------------------
# SECTION: CREATE DIRECTORIES
# -------------------------------
INPUT_DIR="${PROJECT_PATH}/input/small-pdf-dataset"
OUT_PYMUPDF="${PROJECT_PATH}/output/pymupdf"
OUT_PYPDF="${PROJECT_PATH}/output/pypdf"
OUT_NOUGAT="${PROJECT_PATH}/output/nougat"
OUT_ADAPARSE="${PROJECT_PATH}/output/adaparse"
CONF_DIR="../configs"

mkdir -p "$INPUT_DIR" "$OUT_PYMUPDF" "$OUT_PYPDF" "$OUT_NOUGAT" "$OUT_ADAPARSE"
[[ -d "$PROJECT_PATH" ]] || die "Failed to create project path: $PROJECT_PATH"

echo "=== DIRECTORIES CREATED ==="
printf "%s\n" \
  "$INPUT_DIR" \
  "$OUT_ADAPARSE" \
  "$OUT_PYMUPDF" \
  "$OUT_PYPDF" \
  "$OUT_NOUGAT"
echo ""

# -------------------------------
# SECTION: NOUGAT META DIRS
# -------------------------------
NOUGAT_CHECKPOINT="${PROJECT_PATH}/meta/nougat/checkpoint"
NOUGAT_MMD_OUT="${PROJECT_PATH}/meta/nougat/mmd"
NOUGAT_LOGS="${PROJECT_PATH}/meta/nougat/logs"

mkdir -p "$NOUGAT_CHECKPOINT" "$NOUGAT_MMD_OUT" "$NOUGAT_LOGS"

echo "=== NOUGAT META DIRS ==="
printf "%s\n" \
  "$NOUGAT_CHECKPOINT" \
  "$NOUGAT_MMD_OUT" \
  "$NOUGAT_LOGS"
echo


# -------------------------------
# SECTION: DATA IMPORT (always)
# -------------------------------
echo "=== DATA IMPORT ==="
ARCHIVE_LOCAL="${REPO_ROOT}/data/twenty_test_pdfs.tar.gz"
[[ -f "$ARCHIVE_LOCAL" ]] || { echo "[INFO] Archive not found at: $ARCHIVE_LOCAL (skipping)"; echo; }

if [[ -f "$ARCHIVE_LOCAL" ]]; then
  TMP_EXTRACT="${INPUT_DIR}/.extract_tmp"
  rm -rf "$TMP_EXTRACT"
  mkdir -p "$TMP_EXTRACT"

  echo "[INFO] Extracting: $ARCHIVE_LOCAL"
  tar -xzf "$ARCHIVE_LOCAL" -C "$TMP_EXTRACT"

  echo "[INFO] Moving PDFs into: $INPUT_DIR"
  # Move all PDFs found anywhere under the temp extraction into INPUT_DIR
  MOVED=0
  while IFS= read -r -d '' pdf; do
    mv -n "$pdf" "$INPUT_DIR"/ && ((MOVED+=1)) || true
  done < <(find "$TMP_EXTRACT" -type f -iname '*.pdf' -print0)

  # Cleanup temp and any non-PDFs/noise
  rm -rf "$TMP_EXTRACT"
  find "$INPUT_DIR" -type f ! -iname '*.pdf' -delete || true

  COUNT_PDF=$(find "$INPUT_DIR" -maxdepth 1 -type f -iname '*.pdf' | wc -l | tr -d ' ')
  echo "[INFO] PDFs moved: $MOVED"
  echo "[INFO] PDFs available in input: $COUNT_PDF"
  echo
fi

# ---------------------------------------------
# SECTION: MACHINE ENVS
# ---------------------------------------------
echo "=== MACHINE ENVS ==="
[[ -f "$ENV_MAP_PATH" ]] || die "env map not found: $ENV_MAP_PATH"

RAW_CMD="$(yaml_get_value "$ENV_MAP_PATH" "$MACHINE" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
# strip quotes if present
if [[ "$RAW_CMD" =~ ^\".*\"$ || "$RAW_CMD" =~ ^\'.*\'$ ]]; then
  RAW_CMD="${RAW_CMD:1:${#RAW_CMD}-2}"
fi
# replace aurora placeholder if present
WORKER_INIT_CMD="${RAW_CMD//<ABS_PATH_TO_standup_aurora.sh>/$ABS_STANDUP_AURORA}"
[[ -n "$WORKER_INIT_CMD" ]] || die "No worker_init found for machine '$MACHINE' in $ENV_MAP_PATH"

echo "[INFO] worker_init resolved for $MACHINE: $WORKER_INIT_CMD"
echo

# ---------------------------------------------
# SECTION: NOUGAT CHECKPOINT DOWNLOAD
# ---------------------------------------------
echo "=== NOUGAT CHECKPOINT DOWNLOAD ==="
bash "${REPO_ROOT}/scripts/download_nougat_checkpoint.sh" "$NOUGAT_CHECKPOINT"
echo

# -------------------------------
# SECTION: CONFIG RENDER (batch over parsers)
# -------------------------------
echo "=== CONFIG RENDER ==="

PARSERS=(adaparse pymupdf nougat pypdf)
RENDERED_FILES=()  # track outputs
PRIMARY_CONFIG_YAML="" # for usage example

for PARSER in "${PARSERS[@]}"; do
  IN_TMPL="${REPO_ROOT}/configs/templates/${PARSER}/template.yaml"
  OUT_DIR="${REPO_ROOT}/configs/${PARSER}"
  OUT_TMPL="${OUT_DIR}/${MACHINE}_small_test.yaml"

  if [[ ! -f "$IN_TMPL" ]]; then
    echo "[WARN] Missing template for ${PARSER}: $IN_TMPL (skipping)"
    continue
  fi
  mkdir -p "$OUT_DIR"

  # Common substitutions
  COMMON_SED_ARGS=(
    -e "s|<PROJECT_PATH>|$(esc_sed "$PROJECT_PATH")|g"
    -e "s|<PROJECT_NAME>|$(esc_sed "$PROJECT_NAME")|g"
    -e "s|<COMPUTE_NAME>|$(esc_sed "$MACHINE")|g"
    -e "s|<WORKER_INIT>|$(esc_sed "$WORKER_INIT_CMD")|g"
    -e "s|<SCHEDULER_OPTIONS>|$(esc_sed "$SCHEDULER_OPTIONS")|g"
  )

  if [[ "$PARSER" == "nougat" || "$PARSER" == "adaparse" ]]; then
    sed "${COMMON_SED_ARGS[@]}" \
        -e "s|<NOUGAT_CHECKPOINT>|$(esc_sed "$NOUGAT_CHECKPOINT")|g" \
        -e "s|<NOUGAT_MMD_OUT>|$(esc_sed "$NOUGAT_MMD_OUT")|g" \
        -e "s|<NOUGAT_LOGS>|$(esc_sed "$NOUGAT_LOGS")|g" \
        "$IN_TMPL" > "$OUT_TMPL"
  else
    sed "${COMMON_SED_ARGS[@]}" "$IN_TMPL" > "$OUT_TMPL"
  fi

  echo "[OK] ${PARSER} -> ${OUT_TMPL}"
  RENDERED_FILES+=("$OUT_TMPL")

  # - always prefer pymupdf if it rendered
  if [[ -s "$OUT_TMPL" ]]; then
    [[ -z "$PRIMARY_CONFIG_YAML" ]] && PRIMARY_CONFIG_YAML="$OUT_TMPL"
    [[ "$PARSER" == "pymupdf" ]] && PRIMARY_CONFIG_YAML="$OUT_TMPL"
  fi
done

echo

# -------------------------------
# SECTION: USAGE
# -------------------------------
echo "=== USAGE EXAMPLE ==="
echo "python -m adaparse.convert --config ${OUT_TMPL}"
echo

# -------------------------------
# SECTION: DONE
# -------------------------------
echo "=== DONE ==="
