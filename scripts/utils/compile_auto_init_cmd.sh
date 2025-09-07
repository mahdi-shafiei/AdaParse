#!/usr/bin/env bash
set -Eeuo pipefail
# compile_auto_init_cmd.sh
# Builds the re-exec command for initial_setup_adaparse_projects.sh
# Usage:
#   compile_auto_init_cmd.sh \
#     --script /abs/path/to/initial_setup_adaparse_projects.sh \
#     --user <user> \
#     --machine <aurora|polaris|sophia|lambda|local> \
#     --default-base <path-or-empty> \
#     [--projects-helper /path/to/show_available_projects.sh] \
#     [--exec]
#
# Prints the command on stdout (quoted). If --exec is passed, execs it.

SCRIPT=""
USER_NAME=""
MACHINE=""
DEFAULT_BASE=""
PROJECTS_HELPER=""
DO_EXEC=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --script)        SCRIPT="${2:-}"; shift 2;;
    --user)          USER_NAME="${2:-}"; shift 2;;
    --machine)       MACHINE="${2:-}"; shift 2;;
    --default-base)  DEFAULT_BASE="${2:-}"; shift 2;;
    --projects-helper) PROJECTS_HELPER="${2:-}"; shift 2;;
    --exec)          DO_EXEC=1; shift;;
    *) echo "Unknown arg: $1" >&2; exit 2;;
  esac
done

[[ -n "$SCRIPT" && -n "$USER_NAME" && -n "$MACHINE" ]] || { echo "Missing required args" >&2; exit 2; }

cancel='^(|c|n|no|q|quit|exit)$'
project=""
base="$DEFAULT_BASE"

# Big clusters → pick project
if [[ "$MACHINE" =~ ^(polaris|sophia|aurora)$ ]]; then
  mapfile -t projs < <(
    if [[ -n "$PROJECTS_HELPER" && -x "$PROJECTS_HELPER" ]]; then
      "$PROJECTS_HELPER" || true
    else
      # Try sibling helper by default
      THIS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
      if [[ -x "${THIS_DIR}/show_available_projects.sh" ]]; then
        "${THIS_DIR}/show_available_projects.sh" || true
      fi
    fi
  )

  if [[ ${#projs[@]} -gt 0 ]]; then
    echo "Projects for ${USER_NAME}:"
    i=1; for p in "${projs[@]}"; do printf "  [%d] %s\n" "$i" "$p"; ((i++)); done
    echo
    read -r -p "Pick project # or type name (or C/No/N/Q to cancel): " choice_raw
    choice="$(printf '%s' "$choice_raw" | tr '[:upper:]' '[:lower:]')"
    if [[ "$choice" =~ $cancel ]]; then
      echo "[INFO] Cancelled." >&2; exit 130
    fi
    if [[ "$choice_raw" =~ ^[0-9]+$ ]] && (( choice_raw>=1 && choice_raw<=${#projs[@]} )); then
      project="${projs[choice_raw-1]}"
    else
      project="$choice_raw"
    fi
  else
    { echo; echo "[WARN] No projects found."; echo; } >&2
    read -r -p "Enter project name (or C/N/No to cancel): " choice_raw
    choice="$(printf '%s' "$choice_raw" | tr '[:upper:]' '[:lower:]')"
    if [[ "$choice" =~ $cancel ]]; then
      echo "[INFO] Initial setup of AdaParse: cancelled" >&2; exit 130
    fi
    project="$choice_raw"
  fi

  if [[ -z "$project" ]]; then
    echo "ERROR: project is required on $MACHINE." >&2
    exit 1
  fi
fi

# Local → ask for base path
if [[ "$MACHINE" == "local" ]]; then
  while :; do
    read -r -e -p "Enter base path for local setup (or C/No/N/Q to cancel): " bp_raw
    bp="$(printf '%s' "$bp_raw" | tr '[:upper:]' '[:lower:]')"
    if [[ "$bp" =~ $cancel ]]; then
      echo "[INFO] Cancelled." >&2; exit 0
    fi
    if [[ -d "$bp_raw" ]]; then
      base="$bp_raw"; break
    fi
    echo "[WARN] Directory not found: $bp_raw"
  done
fi

# Build command (fully quoted)
# shellcheck disable=SC2059
printf -v CMD "%q " "$SCRIPT" --user_name "$USER_NAME" --machine "$MACHINE"
[[ -n "$project" ]] && printf -v CMD "%s%q " "$CMD" --project_name "$project"
[[ -n "$base"    ]] && printf -v CMD "%s%q " "$CMD" --base_path "$base"

echo "$CMD"

if [[ $DO_EXEC -eq 1 ]]; then
  echo
  echo "[INFO] About to exec:"
  echo "  $CMD"
  exec bash -lc "$CMD"
fi
