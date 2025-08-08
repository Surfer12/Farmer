#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# SPDX-FileCopyrightText: 2025 Jumping Quail Solutions

set -euo pipefail

show_help() {
  cat <<'EOF'
Usage: status_summary.sh [-n N] [-f FILE] [--format short|table|json]

Summarize the last N entries from a StatusUpdate JSONL file.

Options:
  -n N           Number of entries to show (default: 5)
  -f FILE        Path to status.jsonl (default: internal/StatusUpdate/status.jsonl)
  --format FMT   Output format: short | table | json (default: short)
  -h, --help     Show this help and exit

Notes:
  - Requires `jq` for parsing. If not available, falls back to raw tail output.
  - In table format, uses '|' as a delimiter; pipe through `column -t -s '|'` for alignment.
EOF
}

N=5
FILE="internal/StatusUpdate/status.jsonl"
FORMAT="short"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -n)
      N="$2"; shift 2 ;;
    -f)
      FILE="$2"; shift 2 ;;
    --format)
      FORMAT="$2"; shift 2 ;;
    -h|--help)
      show_help; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2; show_help; exit 2 ;;
  esac
done

if [[ ! -f "$FILE" ]]; then
  echo "Status file not found: $FILE" >&2
  exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "jq not found; falling back to raw tail output" >&2
  tail -n "$N" "$FILE"
  exit 0
fi

case "$FORMAT" in
  short)
    tail -n "$N" "$FILE" | jq -rc '.ts+" | "+.component+" | "+.status+" | "+.summary'
    ;;
  table)
    # Emits pipe-delimited lines; caller can pretty print with: ... | column -t -s '|'
    echo "ts|component|status|summary"
    tail -n "$N" "$FILE" | jq -rc '[.ts,.component,.status,.summary] | @tsv' | awk 'BEGIN{FS="\t"; OFS="|"} {print $1,$2,$3,$4}'
    ;;
  json)
    tail -n "$N" "$FILE" | jq -s '.'
    ;;
  *)
    echo "Unknown format: $FORMAT" >&2; exit 2 ;;
esac


