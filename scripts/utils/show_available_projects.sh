#!/usr/bin/env bash
set -Eeuo pipefail
# show_available_projects.sh
# Prints available project names, one per line. Exits 0 if any found.

HELPER_PATH="${HELPER_PATH:-}"   # optional: path to pbs/parse_sbank_projects.sh

# Prefer explicit HELPER_PATH if given
if [[ -n "${HELPER_PATH}" && -x "${HELPER_PATH}" ]]; then
  "${HELPER_PATH}"
  exit $?
fi

# Repo helper in a common relative location (if called from repo scripts/)
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd 2>/dev/null || true)"
if [[ -x "${REPO_ROOT}/pbs/parse_sbank_projects.sh" ]]; then
  "${REPO_ROOT}/pbs/parse_sbank_projects.sh"
  exit $?
fi

# PATH helper
if command -v parse_sbank_projects.sh >/dev/null 2>&1; then
  parse_sbank_projects.sh
  exit $?
fi

# Fallback: sbank-list-projects
if command -v sbank-list-projects >/dev/null 2>&1; then
  sbank-list-projects -f project_name 2>/dev/null | awk '
    BEGIN { IGNORECASE=0 }
    /^[[:space:]]*$/ { next }
    /^Totals:/ { exit }
    /^-+$/ { next }
    $1 == "Project" { next }
    index($0, ":")>0 { next }
    { gsub(/^[[:space:]]+|[[:space:]]+$/, "", $0) }
    /^[A-Za-z][A-Za-z0-9_-]*$/ { print $0 }
  ' | sort -u
  exit 0
fi

# Last-ditch: sbank projects
if command -v sbank >/dev/null 2>&1; then
  sbank projects 2>/dev/null | awk '
    /^[[:space:]]*$/ { next }
    /^[A-Za-z][A-Za-z0-9_-]*$/ { print $0 }
  ' | sort -u
  exit 0
fi

# Nothing worked
exit 1
