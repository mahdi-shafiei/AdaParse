#!/usr/bin/env bash
set -euo pipefail

# Parse project names from: `sbank-list-projects -f project_name`
# Output: one project name per line (suitable for: mapfile -t arr < <(this_script))
#
# Works with table output like:
#   Project
#   ----------------
#   candle_aesp_CNDA
#   EpiCalib
#   FoundEpidem
#   SuperBERT
#
#   Totals:
#     Rows: 4
#     Aurora:
#
# Usage:
#   ./parse_sbank_projects.sh
#
# Exit codes:
#   0 = success (printed ≥1 names)
#   2 = no names found
#   3 = sbank-list-projects not found
#
# Tip to integrate:
#   mapfile -t _proj_list < <(/path/to/parse_sbank_projects.sh)

cmd="sbank-list-projects"
if ! command -v "$cmd" >/dev/null 2>&1; then
  echo "ERROR: '$cmd' not found in PATH." >&2
  exit 3
fi

# Call once; avoid multiple spawns and odd pager behavior
out="$("$cmd" -f project_name 2>/dev/null || true)"

# Nothing came back?
if [[ -z "${out//$'\n'/}" ]]; then
  exit 2
fi

# Parse robustly:
# - drop blank lines
# - drop header "Project" and separator "-----"
# - stop when "Totals:" section begins
# - drop any line with a colon (section labels like "Aurora:")
# - trim whitespace, then keep names like [A-Za-z][A-Za-z0-9_-]*
projects=$(
  printf '%s\n' "$out" | awk '
    BEGIN { IGNORECASE=0 }
    /^[[:space:]]*$/ { next }           # blank
    /^Totals:/ { exit }                  # stop at totals
    /^-+$/ { next }                      # separator row
    $1 == "Project" { next }             # header
    index($0, ":")>0 { next }            # section labels like "Aurora:"
    {
      # trim
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", $0)
      if ($0 ~ /^[A-Za-z][A-Za-z0-9_-]*$/) print $0
    }
  '
)

if [[ -z "${projects//$'\n'/}" ]]; then
  exit 2
fi

printf '%s\n' "$projects"
