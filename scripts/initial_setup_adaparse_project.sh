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

# ---------------------------------------------
# SECTION: PROJECT LIST HELPER
# ---------------------------------------------
# Prefer repo helper, fallback to PATH helper, then raw sbank-list-projects
PARSE_SBANK="${REPO_ROOT}/pbs/parse_sbank_projects.sh"

_list_user_projects() {
  # Prints project names, one per line. Returns 0 on success, 1 on failure.
  if [[ -x "$PARSE_SBANK" ]]; then
    "$PARSE_SBANK"
    return $?
  fi
  if command -v parse_sbank_projects.sh >/dev/null 2>&1; then
    parse_sbank_projects.sh
    return $?
  fi
  if command -v sbank-list-projects >/dev/null 2>&1; then
    # Last-resort inline parser (same logic as helper)
    sbank-list-projects -f project_name 2>/dev/null | awk '
      BEGIN { IGNORECASE=0 }
      /^[[:space:]]*$/ { next }         # blank
      /^Totals:/ { exit }                # stop at totals
      /^-+$/ { next }                    # separator row
      $1 == "Project" { next }           # header
      index($0, ":")>0 { next }          # labels like "Aurora:"
      { gsub(/^[[:space:]]+|[[:space:]]+$/, "", $0) }
      /^[A-Za-z][A-Za-z0-9_-]*$/ { print $0 }
    ' | sort -u
    return 0
  fi
  return 1
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

# ---------------------------------------------
# AUTO MODE (run with no args to be guided)
# Place this block ABOVE the existing "Parse args" loop.
# ---------------------------------------------
ORIG_ARGS=("$@")
ORIG_ARGC=${#ORIG_ARGS[@]}

_cancel_tokens='^(|c|n|no|q|quit|exit)$'  # empty input cancels too

if [[ $ORIG_ARGC -eq 0 && -t 0 && -t 1 ]]; then
  echo "=== AUTO MODE ==="
  _actual_user="$(whoami 2>/dev/null || echo "${USER:-}")"
  _host_lc="$(hostname 2>/dev/null | tr '[:upper:]' '[:lower:]')"

  # Infer machine from hostname
  _machine="local"
  if   [[ "$_host_lc" == aurora-uan-* ]]; then _machine="aurora"
  elif [[ "$_host_lc" =~ ^lambda[0-9]+$ ]]; then _machine="lambda"
  elif [[ "$_host_lc" == polaris* ]]; then      _machine="polaris"
  elif [[ "$_host_lc" == sophia* ]]; then       _machine="sophia"
  fi

  # Default base path mapping
  case "$_machine" in
    polaris|sophia) _base="/eagle/projects" ;;
    aurora)         _base="/lus/flare/projects" ;;
    lambda)         _base="/homes" ;;
    local)          _base="" ;;  # must ask user
  esac

  echo "Detected:"
  echo "  user    : $_actual_user"
  echo "  host    : $_host_lc"
  echo "  machine : $_machine"
  [[ -n "$_base" ]] && echo "  base    : $_base"
  echo

  # For big clusters, pick a project (sbank best-effort)
  # For big clusters, pick a project (sbank best-effort)
_project=""
if [[ "$_machine" =~ ^(polaris|sophia|aurora)$ ]]; then
if mapfile -t _proj_list < <(_list_user_projects); then
    if [[ ${#_proj_list[@]} -gt 0 ]]; then
    echo "Projects for $_actual_user:"
    i=1
    for p in "${_proj_list[@]}"; do
        printf "  [%d] %s\n" "$i" "$p"
        ((i++))
    done
    echo
    read -r -p "Pick project # or type name (or C/No/N/Q to cancel): " _choice_raw
    _choice="$(printf '%s' "$_choice_raw" | tr '[:upper:]' '[:lower:]')"
    if [[ "$_choice" =~ $_cancel_tokens ]]; then
        echo "[INFO] Cancelled by user."
        exit 0
    fi
    if [[ "$_choice_raw" =~ ^[0-9]+$ ]] && (( _choice_raw>=1 && _choice_raw<=${#_proj_list[@]} )); then
        _project="${_proj_list[_choice_raw-1]}"
    else
        _project="$_choice_raw"
    fi
    else
    echo "[WARN] No projects found from helper."
    read -r -p "Enter project name (or C/No/N/Q to cancel): " _choice_raw
    _choice="$(printf '%s' "$_choice_raw" | tr '[:upper:]' '[:lower:]')"
    if [[ "$_choice" =~ $_cancel_tokens ]]; then
        echo "[INFO] Cancelled by user."
        exit 0
    fi
    _project="$_choice_raw"
    fi
else
    echo "[WARN] Could not enumerate projects (no helper / no sbank)."
    read -r -p "Enter project name (or C/No/N/Q to cancel): " _choice_raw
    _choice="$(printf '%s' "$_choice_raw" | tr '[:upper:]' '[:lower:]')"
    if [[ "$_choice" =~ $_cancel_tokens ]]; then
    echo "[INFO] Cancelled by user."
    exit 0
    fi
    _project="$_choice_raw"
fi

if [[ -z "$_project" ]]; then
    echo "ERROR: project is required on $_machine." >&2
    exit 1
fi
fi

# For local machine, we need a base path
if [[ "$_machine" == "local" ]]; then
while :; do
    read -r -e -p "Enter base path for local setup (or C/No/N/Q to cancel): " _bp_raw
    _bp="$(printf '%s' "$_bp_raw" | tr '[:upper:]' '[:lower:]')"
    if [[ "$_bp" =~ $_cancel_tokens ]]; then
    echo "[INFO] Cancelled by user."
    exit 0
    fi
    if [[ -d "$_bp_raw" ]]; then
    _base="$_bp_raw"
    break
    fi
    echo "[WARN] Directory not found: $_bp_raw"
done
fi


  # Assemble re-exec command with explicit flags
  _cmd=( "$0" --user_name "$_actual_user" --machine "$_machine" )
  [[ -n "$_project" ]] && _cmd+=( --project_name "$_project" )
  [[ -n "$_base" ]]    && _cmd+=( --base_path "$_base" )

  echo
  echo "[INFO] About to run:"
  printf '  %q' "${_cmd[@]}"; echo
  read -r -p "Proceed? [Y/n]: " _yn
  _yn_lc="$(printf '%s' "${_yn:-y}" | tr '[:upper:]' '[:lower:]')"
  if [[ "$_yn_lc" =~ ^(n|no)$ ]]; then
    echo "[INFO] Cancelled by user."
    exit 0
  fi

  exec "${_cmd[@]}"
fi

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
# SECTION: CHECK ARGUMENTS (sanity + environment)
# ---------------------------------------------
warn() { echo "[WARNING] $*" >&2; }

# Normalize for comparisons
_host_lc="$(hostname 2>/dev/null | tr '[:upper:]' '[:lower:]')"
_mach_lc="$(printf '%s' "$MACHINE" | tr '[:upper:]' '[:lower:]')"
_user_actual="$(whoami 2>/dev/null || echo "${USER:-}")"

# 1) MACHINE vs actual hostname pattern
#    If user picked a machine that doesn't match the current host naming, warn (do not exit).
_expected_patterns=()
case "$_mach_lc" in
  aurora)  _expected_patterns=("aurora-uan-" "aurora") ;;
  polaris) _expected_patterns=("polaris") ;;
  sophia)  _expected_patterns=("sophia") ;;
  lambda)  _expected_patterns=("lambda") ;;
  local)   _expected_patterns=() ;;  # no constraint
  *)       _expected_patterns=() ;;
esac

if [[ ${#_expected_patterns[@]} -gt 0 ]]; then
  _match=0
  for pat in "${_expected_patterns[@]}"; do
    if [[ "$_host_lc" == *"$pat"* ]]; then _match=1; break; fi
  done
  if [[ $_match -eq 0 ]]; then
    warn "Selected --machine='$MACHINE' but hostname='$_host_lc' doesn't match expected patterns: ${_expected_patterns[*]}"
  fi
fi

# 2) USER_NAME vs actual login user
if [[ -n "$USER_NAME" && "$USER_NAME" != "$_user_actual" ]]; then
  warn "Provided --user_name='$USER_NAME' differs from logged-in user='$_user_actual'."
fi

# 3) PROJECT_NAME membership via 'sbank' (PBS-style accounts) — best-effort
#    Only check on the big clusters where PROJECT_NAME is mandatory.
if [[ "$_mach_lc" =~ ^(polaris|sophia|aurora)$ ]]; then
  if [[ -n "$PROJECT_NAME" ]]; then
    if command -v sbank >/dev/null 2>&1; then
      # Try with explicit -u first; fall back to generic listing.
      if _proj_list="$(sbank projects -u "$USER_NAME" 2>/dev/null)"; then
        if ! grep -q "$PROJECT_NAME" <<<"$_proj_list"; then
          warn "Project '$PROJECT_NAME' not found for user '$USER_NAME' via 'sbank projects -u'."
        fi
      elif _proj_list="$(sbank projects 2>/dev/null)"; then
        if ! grep -q "$PROJECT_NAME" <<<"$_proj_list"; then
          warn "Project '$PROJECT_NAME' not found in 'sbank projects' output."
        fi
      else
        warn "'sbank' is present but could not query projects; skipping project validation."
      fi
    else
      warn "'sbank' command not found; skipping project validation."
    fi
  fi
fi

# 4) BASE_PATH existence check (warn only; some sites mount late)
if [[ -n "$BASE_PATH" && ! -d "$BASE_PATH" ]]; then
  warn "BASE_PATH '$BASE_PATH' does not exist or is not a directory."
fi

# 5) PROJECT_PATH parent existence check (warn if parent missing)
_proj_parent="$(dirname -- "$PROJECT_PATH")"
if [[ ! -d "$_proj_parent" ]]; then
  warn "Parent directory for PROJECT_PATH does not exist: $_proj_parent"
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

# ---------------------------------------------
# SECTION: PERSISTENT PATHS FOR TESTING
# ---------------------------------------------
STATE_JSON="${REPO_ROOT}/.adaparse_state.json"
ENV_FILE="${REPO_ROOT}/.adaparse.env"

cat > "$STATE_JSON" <<JSON
{
  "machine": "$MACHINE",
  "user_name": "$USER_NAME",
  "project_name": "${PROJECT_NAME:-}",
  "base_path": "$BASE_PATH",
  "project_path": "$PROJECT_PATH",
  "checkpoint_dir": "$NOUGAT_CHECKPOINT"
}
JSON
echo "[INFO] wrote $STATE_JSON"

cat > "$ENV_FILE" <<ENV
ADAPARSE_MACHINE=$MACHINE
ADAPARSE_USER_NAME=$USER_NAME
ADAPARSE_PROJECT_NAME=${PROJECT_NAME:-}
ADAPARSE_BASE_PATH=$BASE_PATH
ADAPARSE_PROJECT_ROOT=$PROJECT_PATH
ADAPARSE_CHECKPOINT=$NOUGAT_CHECKPOINT
ENV
echo "[INFO] wrote $ENV_FILE  (you can:  source $ENV_FILE)"

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
  case "$ARCHIVE_LOCAL" in
    *.tar.gz|*.tgz)
      tar -xzf "$ARCHIVE_LOCAL" -C "$TMP_EXTRACT"
      ;;
    *.tar)
      tar -xf "$ARCHIVE_LOCAL" -C "$TMP_EXTRACT"
      ;;
    *.zip)
      if command -v unzip >/dev/null 2>&1; then
        unzip -q "$ARCHIVE_LOCAL" -d "$TMP_EXTRACT"
      else
        echo "ERROR: 'unzip' not found; cannot extract $ARCHIVE_LOCAL" >&2
        exit 1
      fi
      ;;
    *)
      echo "ERROR: Unknown archive type: $ARCHIVE_LOCAL" >&2
      exit 1
      ;;
  esac

  echo "[INFO] Moving PDFs into: $INPUT_DIR"
  MOVED=0
  while IFS= read -r -d '' pdf; do
    mv -n "$pdf" "$INPUT_DIR"/ && ((MOVED+=1)) || true
  done < <(find "$TMP_EXTRACT" -type f \( -iname '*.pdf' -o -iname '*.PDF' \) -print0)

  # Cleanup ONLY the temp extraction directory; keep the original archive
  rm -rf "$TMP_EXTRACT"

  # Optional: purge any non-PDFs that may have slipped into INPUT_DIR
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

# ---------------------------------------------------------------
# SECTION: PRE-TRAINED SWIN TRANSFORMER WEIGHTS DOWNLOAD
# ---------------------------------------------------------------
echo "=== SWIN WEIGHTS DOWNLOAD ==="
bash "${REPO_ROOT}/scripts/weights/download_swin_weights.sh" "$NOUGAT_CHECKPOINT"
echo

# ---------------------------------------------
# SECTION: NOUGAT CHECKPOINT DOWNLOAD
# ---------------------------------------------
echo "=== NOUGAT CHECKPOINT DOWNLOAD ==="
bash "${REPO_ROOT}/scripts/weights/download_nougat_checkpoint.sh" "$NOUGAT_CHECKPOINT"
echo

# ----------------------------------------------
# SECTION: CONFIG RENDER (batch over parsers)
# ----------------------------------------------
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
