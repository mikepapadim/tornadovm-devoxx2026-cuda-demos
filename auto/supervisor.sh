#!/usr/bin/env bash
# Unattended Claude driver: one task per invocation, durable state on disk.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
export LC_ALL=C

CLAUDE_BIN="${CLAUDE_BIN:-claude}"
MODEL="${MODEL:-}"
MAX_ITER="${MAX_ITER:-2000}"
ITER_TIMEOUT="${ITER_TIMEOUT:-5400}"
IDLE_SLEEP="${IDLE_SLEEP:-30}"
MAX_BACKOFF="${MAX_BACKOFF:-1800}"
MAX_LIMIT_WAIT="${MAX_LIMIT_WAIT:-43200}"
STALL_LIMIT="${STALL_LIMIT:-3}"
PUSH="${PUSH:-1}"
BRANCH="${BRANCH:-main}"
NOTIFY_URL="${NOTIFY_URL:-}"

TASK_DIR="$ROOT/auto/tasks"
STATE_DIR="$ROOT/auto/state"
LOG_DIR="$ROOT/auto/logs"
PROMPT_FILE="$ROOT/auto/prompt.md"
STOP_FILE="$ROOT/auto/STOP"
PAUSE_FILE="$ROOT/auto/PAUSE"
mkdir -p "$TASK_DIR" "$STATE_DIR" "$LOG_DIR"

ts(){ date -u +%Y-%m-%dT%H:%M:%SZ; }
log(){ printf '[%s] %s\n' "$(ts)" "$*" | tee -a "$LOG_DIR/supervisor.log"; }
notify(){ [ -n "$NOTIFY_URL" ] || return 0; curl -fsS -m 10 -d "$1" "$NOTIFY_URL" >/dev/null 2>&1 || true; }

next_task(){
  local f id
  for f in "$TASK_DIR"/*.md; do
    [ -e "$f" ] || continue
    id="$(basename "$f" .md)"
    [ -f "$STATE_DIR/$id.done" ] && continue
    [ -f "$STATE_DIR/$id.blocked" ] && continue
    printf '%s\n' "$id"
    return 0
  done
  return 1
}

limit_epoch(){
  { grep -hoE 'usage limit reached\|[0-9]{9,}' "$1" 2>/dev/null | tail -1 | cut -d'|' -f2 || true
    grep -hoE '"resetsAt" *: *[0-9]{9,}' "$1" 2>/dev/null | tail -1 | grep -oE '[0-9]{9,}' || true
  } | grep -E '^[0-9]{9,}$' | tail -1
}
rate_limited(){ grep -qiE 'usage limit reached|rate_limit_error|rate_limit_event|429|Too Many Requests|out_of_credits|session limit' "$1" 2>/dev/null; }
auth_error(){ grep -qiE 'authentication_error|invalid api key|claude login|OAuth token has expired' "$1" 2>/dev/null; }

sleep_for(){
  local remain="$1" reason="$2"
  (( remain < 60 )) && remain=900
  (( remain > MAX_LIMIT_WAIT )) && remain="$MAX_LIMIT_WAIT"
  log "$reason; sleeping ${remain}s"
  notify "paused ${remain}s: $reason"
  while (( remain > 0 )); do
    [ -f "$STOP_FILE" ] && return 1
    local chunk=$(( remain > 300 ? 300 : remain ))
    sleep "$chunk"
    remain=$(( remain - chunk ))
  done
}

assert_remote(){
  local url
  url="$(git remote get-url origin 2>/dev/null || true)"
  [[ "$url" =~ ^(git@github\.com:|https://([^@/]+@)?github\.com/)mikepapadim/tornadovm-devoxx2026-cuda-demos(\.git)?/?$ ]] || {
    log "REFUSING: origin is '$url'"
    return 1
  }
  [ -z "$(git remote | grep -vx origin || true)" ] || log "warning: extra remotes exist; they will never be pushed"
}

command -v "$CLAUDE_BIN" >/dev/null || { log 'claude is not on PATH'; exit 1; }
[ -f "$PROMPT_FILE" ] || { log 'missing auto/prompt.md'; exit 1; }
assert_remote || exit 1

git config core.hooksPath .githooks

MODEL_ARG=(); [ -n "$MODEL" ] && MODEL_ARG=(--model "$MODEL")
iter=0; backoff=60; stall=0
log "=== supervisor start === next=$(next_task || echo NONE)"
notify "Devoxx CUDA study supervisor started"

while (( iter < MAX_ITER )); do
  [ -f "$STOP_FILE" ] && break
  while [ -f "$PAUSE_FILE" ]; do sleep 120; [ -f "$STOP_FILE" ] && break 2; done
  task="$(next_task)" || { log 'queue empty'; notify 'queue empty'; break; }
  iter=$((iter+1))
  logfile="$LOG_DIR/iter-$(printf '%04d' "$iter")-${task}.log"
  before="$(git rev-parse --short HEAD 2>/dev/null || echo none)"

  git pull --rebase -q origin "$BRANCH" 2>>"$LOG_DIR/supervisor.log" || true
  timeout --signal=TERM --kill-after=60 "$ITER_TIMEOUT" \
    "$CLAUDE_BIN" -p "$(cat "$PROMPT_FILE")" \
      "${MODEL_ARG[@]}" --output-format stream-json --verbose \
      --max-budget-usd "${MAX_BUDGET_USD:-6}" \
      --dangerously-skip-permissions >"$logfile" 2>&1
  rc=$?
  after="$(git rev-parse --short HEAD 2>/dev/null || echo none)"

  if rate_limited "$logfile"; then
    epoch="$(limit_epoch "$logfile")"
    if [ -n "$epoch" ]; then
      now=$(date +%s); sleep_for $(( epoch - now )) 'Claude usage limit' || break
    else
      sleep_for "$backoff" 'rate limit without reset time' || break
      backoff=$(( backoff * 2 )); (( backoff > MAX_BACKOFF )) && backoff="$MAX_BACKOFF"
    fi
    continue
  fi

  if auth_error "$logfile"; then
    log "AUTH ERROR; stopping: $logfile"; notify 'Claude authentication needs attention'; break
  fi

  if [ "$before" = "$after" ]; then
    stall=$((stall+1))
    log "task=$task rc=$rc no commit; stall=$stall/$STALL_LIMIT"
    if (( stall >= STALL_LIMIT )); then
      printf 'blocked by supervisor at %s after %d commit-free iterations; see %s\n' "$(ts)" "$stall" "$logfile" > "$STATE_DIR/$task.blocked"
      stall=0
    fi
  else
    stall=0; backoff=60
    log "task=$task rc=$rc commit $before -> $after"
  fi

  if [ "$PUSH" = 1 ]; then
    assert_remote && git push -q origin "HEAD:$BRANCH" 2>>"$LOG_DIR/supervisor.log" || log 'push failed; continuing'
  fi
  sleep "$IDLE_SLEEP"
done

log "=== supervisor exit === iterations=$iter"
notify "Devoxx CUDA study supervisor exited after $iter iterations"
