#!/usr/bin/env bash
set -euo pipefail

# initial_setup_adaparse_project.sh
# Creates project dirs, imports sample PDFs, downloads Nougat weights,
# renders YAML configs, and writes state/env files.
#
# Location:
#   this file:        ./scripts/initial_setup_adaparse_project.sh
#   helpers (utils):  ./scripts/utils/*.sh

# --- Repo & path layout ---
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"   # ./scripts
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
UTILS_DIR="${SCRIPT_DIR}/utils"

ENV_MAP_PATH="${REPO_ROOT}/configs/machines/envs.yaml"
ABS_STANDUP_AURORA="${REPO_ROOT}/scripts/standup_aurora.sh"
ARCHIVE_LOCAL="${REPO_ROOT}/data/twenty_test_pdfs.tar.gz"

# --- Helper scripts ---
DETECT="${UTILS_DIR}/detect_machine.sh"
SHOWPROJ="${UTILS_DIR}/show_available_projects.sh"
COMPILE="${UTILS_DIR}/compile_auto_init_cmd.sh"
SANITY="${UTILS_DIR}/check_sanity.sh"
IMPORT="${UTILS_DIR}/data_import.sh"
RENDER="${UTILS_DIR}/render_config.sh"
DL_NOUGAT="${REPO_ROOT}/scripts/weights/download_nougat_checkpoint.sh"

# --- Utilities ---
die()  { echo "ERROR: $*" >&2; exit 1; }
note() { echo "[INFO] $*"; }
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
  initial_setup_adaparse_project.sh \
    --user_name <name> \
    [--machine polaris|sophia|aurora|lambda|local] \
    [--project_name <proj>] \
    [--base_path <abs path>]

Notes:
--user_name is required.
--project_name is required for: polaris, sophia, aurora.
--base_path is auto-set for polaris/sophia/aurora/lambda; for local you must supply it.

This script supports an interactive "auto" mode if run with no args in a TTY.
USAGE
}

# --- Defaults ---
MACHINE="local"
USER_NAME=""
PROJECT_NAME=""
BASE_PATH=""

# --- AUTO MODE (no args, interactive) ---
if [[ $# -eq 0 && -t 0 && -t 1 ]]; then
  echo "=== AUTO MODE ==="
  # Detect machine/user/base (prints as env lines; we eval them)
  eval "$("$DETECT" --format env)"   # defines: ACTUAL_USER, FQDN, DOMAIN, MACHINE, DEFAULT_BASE

  echo "Detected:"
  echo "  user    : ${ACTUAL_USER}"
  echo "  fqdn    : ${FQDN}"
  [[ -n "${DOMAIN}" ]] && echo "  domain  : ${DOMAIN}"
  echo "  machine : ${MACHINE}"
  [[ -n "${DEFAULT_BASE}" ]] && echo "  base    : ${DEFAULT_BASE}"
  echo

  # Build the re-exec command interactively, but SHOW the project list/prompt.
  # We tee the helper's stdout to a temp file so we can capture the final line (the command).
  tmpfile="$(mktemp)"
  # Run the compiler helper so its prompts & lists are visible
  "${COMPILE}" \
    --script "$0" \
    --user   "${ACTUAL_USER}" \
    --machine "${MACHINE}" \
    --default-base "${DEFAULT_BASE}" \
    --projects-helper "${SHOWPROJ}" \
  | tee "$tmpfile"

  # The last line is the fully quoted re-exec command
  REEXEC_CMD="$(tail -n 1 "$tmpfile")"
  rm -f "$tmpfile"

  if [[ -z "${REEXEC_CMD}" ]]; then
    die "Failed to construct re-exec command in auto mode."
  fi

  echo
  echo "[INFO] About to run:"
  echo "  ${REEXEC_CMD}"
  read -r -p "Proceed? [Y/n]: " _yn
  _yn_lc="$(printf '%s' "${_yn:-y}" | tr '[:upper:]' '[:lower:]')"
  [[ "$_yn_lc" =~ ^(n|no)$ ]] && { echo "[INFO] Cancelled by user."; exit 0; }

  exec bash -lc "${REEXEC_CMD}"
fi

# --- Parse args ---
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

[[ -z "$USER_NAME" ]] && { usage; die "--user_name is required."; }
case "$MACHINE" in polaris|sophia|aurora|lambda|local) ;; *) die "--machine must be one of: polaris, sophia, aurora, lambda, local";; esac

# Auto base path if missing (except local)
if [[ -z "$BASE_PATH" ]]; then
  case "$MACHINE" in
    polaris|sophia) BASE_PATH="/eagle/projects" ;;
    aurora)         BASE_PATH="/lus/flare/projects" ;;
    lambda)         BASE_PATH="/homes" ;;
    local)          : ;;
  esac
fi
[[ "$MACHINE" == "local" && -z "$BASE_PATH" ]] && { usage; die "--base_path is required when --machine=local."; }

# Enforce project for big clusters
if [[ "$MACHINE" =~ ^(polaris|sophia|aurora)$ ]] && [[ -z "$PROJECT_NAME" ]]; then
  usage; die "--project_name is required for machine=$MACHINE."
fi

# Compute PROJECT_PATH
if [[ -n "$PROJECT_NAME" ]]; then
  PROJECT_PATH="${BASE_PATH%/}/${PROJECT_NAME}/${USER_NAME}/adaparse_data"
else
  PROJECT_PATH="${BASE_PATH%/}/${USER_NAME}/adaparse_data"
fi

# --- Scheduler options (machine-aware) ---
echo "=== SCHEDULER OPTIONS ==="
case "$MACHINE" in
  aurora)         SCHEDULER_OPTIONS='#PBS -l filesystems=home:flare' ;;
  polaris|sophia) SCHEDULER_OPTIONS='#PBS -l filesystems=home:eagle' ;;
  lambda|local)   SCHEDULER_OPTIONS='' ;;
esac
echo "[INFO] scheduler_options resolved: ${SCHEDULER_OPTIONS:-<empty>}"
echo

# --- Argument summary ---
echo "=== ARGUMENTS ==="
echo "machine      : $MACHINE"
echo "user_name    : $USER_NAME"
echo "project_name : ${PROJECT_NAME:-<none>}"
echo "base_path    : $BASE_PATH"
echo "project_path : $PROJECT_PATH"
echo

# --- Sanity checks (warnings only) ---
"$SANITY" \
  --machine "$MACHINE" \
  --user_name "$USER_NAME" \
  --project_name "${PROJECT_NAME:-}" \
  --base_path "$BASE_PATH" \
  --project_path "$PROJECT_PATH"

# --- Create directories ---
INPUT_DIR="${PROJECT_PATH}/input/small-pdf-dataset"
OUT_PYMUPDF="${PROJECT_PATH}/output/pymupdf"
OUT_PYPDF="${PROJECT_PATH}/output/pypdf"
OUT_NOUGAT="${PROJECT_PATH}/output/nougat"
OUT_ADAPARSE="${PROJECT_PATH}/output/adaparse"
mkdir -p "$INPUT_DIR" "$OUT_PYMUPDF" "$OUT_PYPDF" "$OUT_NOUGAT" "$OUT_ADAPARSE"
[[ -d "$PROJECT_PATH" ]] || die "Failed to create project path: $PROJECT_PATH"

echo "=== DIRECTORIES CREATED ==="
printf "%s\n" "$INPUT_DIR" "$OUT_ADAPARSE" "$OUT_PYMUPDF" "$OUT_PYPDF" "$OUT_NOUGAT"
echo

# --- Nougat meta dirs ---
NOUGAT_CHECKPOINT="${PROJECT_PATH}/meta/nougat/checkpoint"
NOUGAT_MMD_OUT="${PROJECT_PATH}/meta/nougat/mmd"
NOUGAT_LOGS="${PROJECT_PATH}/meta/nougat/logs"
mkdir -p "$NOUGAT_CHECKPOINT" "$NOUGAT_MMD_OUT" "$NOUGAT_LOGS"

echo "=== NOUGAT META DIRS ==="
printf "%s\n" "$NOUGAT_CHECKPOINT" "$NOUGAT_MMD_OUT" "$NOUGAT_LOGS"
echo

# --- Persistent state/env ---
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
echo

# --- Machine envs (worker_init from YAML) ---
echo "=== MACHINE ENVS ==="
[[ -f "$ENV_MAP_PATH" ]] || die "env map not found: $ENV_MAP_PATH"
RAW_CMD="$(yaml_get_value "$ENV_MAP_PATH" "$MACHINE" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
if [[ "$RAW_CMD" =~ ^\".*\"$ || "$RAW_CMD" =~ ^\'.*\'$ ]]; then
  RAW_CMD="${RAW_CMD:1:${#RAW_CMD}-2}"
fi
WORKER_INIT_CMD="${RAW_CMD//<ABS_PATH_TO_standup_aurora.sh>/$ABS_STANDUP_AURORA}"
[[ -n "$WORKER_INIT_CMD" ]] || die "No worker_init for machine '$MACHINE' in $ENV_MAP_PATH"
echo "[INFO] worker_init resolved for $MACHINE: $WORKER_INIT_CMD"
echo

# --- Download Nougat checkpoint ---
echo "=== NOUGAT CHECKPOINT DOWNLOAD ==="
bash "$DL_NOUGAT" "$NOUGAT_CHECKPOINT"
echo

# --- Import sample data ---
bash "$IMPORT" --archive "$ARCHIVE_LOCAL" --input-dir "$INPUT_DIR"

# --- Render parser configs ---
PRIMARY_CONFIG_YAML_LINE="$(
  bash "$RENDER" \
    --repo-root "$REPO_ROOT" \
    --machine "$MACHINE" \
    --project-path "$PROJECT_PATH" \
    --project-name "${PROJECT_NAME:-}" \
    --worker-init "$WORKER_INIT_CMD" \
    --scheduler-options "$SCHEDULER_OPTIONS" \
    --nougat-checkpoint "$NOUGAT_CHECKPOINT" \
    --nougat-mmd-out "$NOUGAT_MMD_OUT" \
    --nougat-logs "$NOUGAT_LOGS" \
  | tail -n 1
)"
# Expect "PRIMARY_CONFIG_YAML=/abs/path"
eval "$PRIMARY_CONFIG_YAML_LINE"
echo

# --- Usage example ---
echo "=== USAGE EXAMPLE ==="
echo "python -m adaparse.convert --config ${PRIMARY_CONFIG_YAML}"
echo

echo "=== DONE ==="
