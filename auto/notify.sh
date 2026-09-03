#!/usr/bin/env bash
set -uo pipefail

# Usage:
#   NTFY_TOPIC='your-unguessable-topic' ./auto/notify.sh 'message'
# Optional:
#   NTFY_TOKEN='tk_...' NTFY_TITLE='TornadoVM Study' ./auto/notify.sh 'message'

TOPIC="${NTFY_TOPIC:-}"
TOKEN="${NTFY_TOKEN:-}"
TITLE="${NTFY_TITLE:-TornadoVM CUDA Study}"
MESSAGE="${1:-}"

[ -n "$TOPIC" ] || { echo 'NTFY_TOPIC is required' >&2; exit 2; }
[ -n "$MESSAGE" ] || { echo 'message is required' >&2; exit 2; }

args=(-fsS -m 10 -H "Title: $TITLE" -H 'Tags: computer,gpu')
[ -n "$TOKEN" ] && args+=(-H "Authorization: Bearer $TOKEN")

curl "${args[@]}" -d "$MESSAGE" "https://ntfy.sh/$TOPIC" >/dev/null
