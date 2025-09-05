#!/usr/bin/env bash
set -Eeuo pipefail
# detect_machine.sh
# Usage: detect_machine.sh [--format env|json]
FMT="${1:-env}"

actual_user="$(whoami 2>/dev/null || echo "${USER:-}")"
fqdn="$(hostname -f 2>/dev/null || hostname 2>/dev/null || echo "")"
domain="$(hostname -d 2>/dev/null || echo "")"
fqdn_lc="${fqdn,,}"

machine="local"
if [[ "$fqdn_lc" == *"alcf.anl.gov"* ]]; then
  if   [[ "$fqdn_lc" == *"aurora"*  ]]; then machine="aurora"
  elif [[ "$fqdn_lc" == *"sophia"*  ]]; then machine="sophia"
  elif [[ "$fqdn_lc" == *"polaris"* ]]; then machine="polaris"
  else                                       machine="local"
  fi
fi

default_base=""
case "$machine" in
  polaris|sophia) default_base="/eagle/projects" ;;
  aurora)         default_base="/lus/flare/projects" ;;
  lambda)         default_base="/homes" ;;
  local)          default_base="" ;;
esac

if [[ "$FMT" == "json" ]]; then
  cat <<JSON
{"ACTUAL_USER":"$actual_user","FQDN":"$fqdn","DOMAIN":"$domain","MACHINE":"$machine","DEFAULT_BASE":"$default_base"}
JSON
else
  echo "ACTUAL_USER=$actual_user"
  echo "FQDN=$fqdn"
  echo "DOMAIN=$domain"
  echo "MACHINE=$machine"
  echo "DEFAULT_BASE=$default_base"
fi
