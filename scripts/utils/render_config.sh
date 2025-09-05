#!/usr/bin/env bash
set -Eeuo pipefail
# render_config.sh
# Usage:
#   render_config.sh \
#     --repo-root R --machine M --project-path P --project-name N \
#     --worker-init W --scheduler-options S \
#     --nougat-checkpoint C --nougat-mmd-out MM --nougat-logs L

REPO_ROOT=""; MACHINE=""; PROJECT_PATH=""; PROJECT_NAME=""
WORKER_INIT_CMD=""; SCHEDULER_OPTIONS=""
NOUGAT_CHECKPOINT=""; NOUGAT_MMD_OUT=""; NOUGAT_LOGS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-root)         REPO_ROOT="${2:-}"; shift 2;;
    --machine)           MACHINE="${2:-}"; shift 2;;
    --project-path)      PROJECT_PATH="${2:-}"; shift 2;;
    --project-name)      PROJECT_NAME="${2:-}"; shift 2;;
    --worker-init)       WORKER_INIT_CMD="${2:-}"; shift 2;;
    --scheduler-options) SCHEDULER_OPTIONS="${2:-}"; shift 2;;
    --nougat-checkpoint) NOUGAT_CHECKPOINT="${2:-}"; shift 2;;
    --nougat-mmd-out)    NOUGAT_MMD_OUT="${2:-}"; shift 2;;
    --nougat-logs)       NOUGAT_LOGS="${2:-}"; shift 2;;
    *) echo "Unknown arg: $1" >&2; exit 2;;
  esac
done

esc_sed() { printf '%s' "$1" | sed -e 's/[\/&|]/\\&/g'; }

echo "=== CONFIG RENDER ==="
PARSERS=(adaparse pymupdf nougat pypdf)
PRIMARY_CONFIG_YAML=""

for PARSER in "${PARSERS[@]}"; do
  IN_TMPL="${REPO_ROOT}/configs/templates/${PARSER}/template.yaml"
  OUT_DIR="${REPO_ROOT}/configs/${PARSER}"
  OUT_TMPL="${OUT_DIR}/${MACHINE}_small_test.yaml"

  if [[ ! -f "$IN_TMPL" ]]; then
    echo "[WARN] Missing template for ${PARSER}: $IN_TMPL (skipping)"
    continue
  fi
  mkdir -p "$OUT_DIR"

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

  if [[ -s "$OUT_TMPL" ]]; then
    [[ -z "$PRIMARY_CONFIG_YAML" ]] && PRIMARY_CONFIG_YAML="$OUT_TMPL"
    [[ "$PARSER" == "pymupdf" ]] && PRIMARY_CONFIG_YAML="$OUT_TMPL"
  fi
done

echo
echo "PRIMARY_CONFIG_YAML=${PRIMARY_CONFIG_YAML}"
