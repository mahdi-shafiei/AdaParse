#!/usr/bin/env bash
set -Eeuo pipefail
# check_sanity.sh
# Usage:
#   check_sanity.sh --machine M --user_name U --project_name P? --base_path B --project_path PP

MACHINE=""; USER_NAME=""; PROJECT_NAME=""; BASE_PATH=""; PROJECT_PATH=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --machine) MACHINE="${2:-}"; shift 2;;
    --user_name) USER_NAME="${2:-}"; shift 2;;
    --project_name) PROJECT_NAME="${2:-}"; shift 2;;
    --base_path) BASE_PATH="${2:-}"; shift 2;;
    --project_path) PROJECT_PATH="${2:-}"; shift 2;;
    *) echo "Unknown arg: $1" >&2; exit 2;;
  esac
done

warn() { echo "[WARNING] $*" >&2; }

_host_lc="$(hostname 2>/dev/null | tr '[:upper:]' '[:lower:]')"
_mach_lc="$(printf '%s' "$MACHINE" | tr '[:upper:]' '[:lower:]')"
_user_actual="$(whoami 2>/dev/null || echo "${USER:-}")"

expected=()
case "$_mach_lc" in
  aurora)  expected=("aurora-uan-" "aurora") ;;
  polaris) expected=("polaris") ;;
  sophia)  expected=("sophia") ;;
  lambda)  expected=("lambda") ;;
  local)   expected=() ;;
esac

if [[ ${#expected[@]} -gt 0 ]]; then
  match=0
  for pat in "${expected[@]}"; do
    [[ "$_host_lc" == *"$pat"* ]] && { match=1; break; }
  done
  [[ $match -eq 0 ]] && warn "Selected --machine='$MACHINE' but hostname='$_host_lc' not in expected: ${expected[*]}"
fi

if [[ -n "$USER_NAME" && "$USER_NAME" != "$_user_actual" ]]; then
  warn "Provided --user_name='$USER_NAME' differs from logged-in user='$_user_actual'."
fi

if [[ "$_mach_lc" =~ ^(polaris|sophia|aurora)$ ]] && [[ -n "$PROJECT_NAME" ]]; then
  if command -v sbank >/dev/null 2>&1; then
    if proj_list="$(sbank projects -u "$USER_NAME" 2>/dev/null)"; then
      grep -q "$PROJECT_NAME" <<<"$proj_list" || warn "Project '$PROJECT_NAME' not found for user '$USER_NAME' via 'sbank projects -u'."
    elif proj_list="$(sbank projects 2>/dev/null)"; then
      grep -q "$PROJECT_NAME" <<<"$proj_list" || warn "Project '$PROJECT_NAME' not found in 'sbank projects' output."
    else
      warn "'sbank' present but cannot query; skipping project validation."
    fi
  else
    warn "'sbank' command not found; skipping project validation."
  fi
fi

if [[ -n "$BASE_PATH" && ! -d "$BASE_PATH" ]]; then
  warn "BASE_PATH '$BASE_PATH' does not exist or is not a directory."
fi

proj_parent="$(dirname -- "$PROJECT_PATH" 2>/dev/null || echo "")"
if [[ -n "$proj_parent" && ! -d "$proj_parent" ]]; then
  warn "Parent directory for PROJECT_PATH does not exist: $proj_parent"
fi
