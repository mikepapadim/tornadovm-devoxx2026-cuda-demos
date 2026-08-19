#!/usr/bin/env bash
# auto/preflight.sh — run this before leaving the CUDA server unattended.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
pass=0; fail=0; warnings=0
ok(){ printf 'PASS %s\n' "$*"; pass=$((pass+1)); }
bad(){ printf 'FAIL %s\n' "$*"; fail=$((fail+1)); }
warn(){ printf 'WARN %s\n' "$*"; warnings=$((warnings+1)); }
sec(){ printf '\n== %s ==\n' "$*"; }

sec 'Basic tools'
for t in git bash curl timeout python3 jq make cmake mvn java; do
  command -v "$t" >/dev/null && ok "$t present" || bad "$t missing"
done
command -v jbang >/dev/null && ok 'jbang present' || warn 'jbang missing — JBang paths cannot be tested'

sec 'CUDA and NVIDIA profilers'
for t in nvidia-smi nvcc ptxas cuobjdump nvdisasm nsys ncu; do
  command -v "$t" >/dev/null && ok "$t present" || bad "$t missing"
done
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=name,driver_version,memory.total,persistence_mode --format=csv,noheader | sed 's/^/GPU: /'
  busy="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^$/d' | wc -l)"
  [ "$busy" -eq 0 ] && ok 'GPU is idle' || warn "$busy CUDA process(es) already running"
fi
command -v nsys >/dev/null && nsys --version 2>&1 | head -1 | sed 's/^/NSYS: /'
command -v ncu >/dev/null && ncu --version 2>&1 | head -1 | sed 's/^/NCU: /'
if command -v nsys >/dev/null; then
  warn 'Review nsys status below; GPU performance-counter permissions may need administrator changes'
  nsys status --environment 2>&1 | head -15 | sed 's/^/  /' || true
fi

sec 'Claude Code'
if command -v claude >/dev/null; then
  ok "claude present: $(claude --version 2>/dev/null | head -1)"
  if out="$(timeout 180 claude -p 'reply with exactly: OK' --output-format json 2>&1)"; then
    echo "$out" | grep -qi 'OK' && ok 'claude authenticated and responding' || bad "claude returned unexpected output: $(echo "$out" | head -c 200)"
  else
    bad 'claude -p failed — authenticate before unattended execution'
  fi
else bad 'claude missing'; fi

sec 'Git and publication boundary'
git rev-parse --is-inside-work-tree >/dev/null 2>&1 && ok 'inside git repository' || bad 'not a git repository'
origin="$(git remote get-url origin 2>/dev/null || true)"
[[ "$origin" =~ ^(git@github\.com:|https://([^@/]+@)?github\.com/)mikepapadim/tornadovm-devoxx2026-cuda-demos(\.git)?/?$ ]] && ok 'origin matches publication boundary' || bad "origin mismatch: $origin"
extra="$(git remote | grep -vx origin || true)"
[ -z "$extra" ] && ok 'no extra remotes' || bad "extra remotes: $extra"
[ -n "$(git config user.email 2>/dev/null || true)" ] && ok 'git identity configured' || bad 'git user.email missing'
[ -x .githooks/pre-push ] && ok 'pre-push guard executable' || bad 'pre-push guard missing/not executable'
[ "$(git config core.hooksPath 2>/dev/null || true)" = '.githooks' ] && ok 'core.hooksPath=.githooks' || bad 'core.hooksPath is not .githooks'
if git push --dry-run origin HEAD >/dev/null 2>&1; then ok 'push dry-run succeeds'; else bad 'push dry-run failed'; fi

sec 'Study wiring'
for f in CLAUDE.md PLAN.md STATE.md auto/prompt.md auto/supervisor.sh auto/gen_tasks.sh docs/run-conventions.md; do
  [ -f "$f" ] && ok "$f present" || bad "$f missing"
done
n="$(find auto/tasks -maxdepth 1 -type f -name '*.md' 2>/dev/null | wc -l)"
[ "$n" -gt 0 ] && ok "$n queued tasks" || bad 'no tasks queued'
grep -q 'java @' docs/run-conventions.md 2>/dev/null && ok 'Java @arg-file convention documented' || bad 'Java @arg-file convention missing'
grep -qi 'jbang' docs/run-conventions.md 2>/dev/null && ok 'JBang convention documented' || bad 'JBang convention missing'

sec 'Machine endurance'
avail="$(df -BG --output=avail . 2>/dev/null | tail -1 | tr -dc '0-9')"
[ "${avail:-0}" -ge 50 ] && ok "${avail}G free disk" || bad "only ${avail:-0}G free disk"
command -v tmux >/dev/null && ok 'tmux present' || bad 'tmux missing'

sec 'Working tree'
if git diff --quiet && git diff --cached --quiet; then ok 'working tree clean'; else bad 'working tree has uncommitted changes'; fi

printf '\n%d passed, %d warnings, %d failed\n' "$pass" "$warnings" "$fail"
if [ "$fail" -ne 0 ]; then
  echo 'Fix the failures before starting auto/supervisor.sh.'
  exit 1
fi
echo 'Preflight passed. Review warnings before walking away.'
