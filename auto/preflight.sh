#!/usr/bin/env bash
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
pass=0; fail=0
ok(){ echo "PASS $*"; pass=$((pass+1)); }
bad(){ echo "FAIL $*"; fail=$((fail+1)); }
warn(){ echo "WARN $*"; }

command -v git >/dev/null && ok 'git present' || bad 'git missing'
command -v bash >/dev/null && ok 'bash present' || bad 'bash missing'
command -v curl >/dev/null && ok 'curl present' || bad 'curl missing'
command -v timeout >/dev/null && ok 'timeout present' || bad 'timeout missing'
command -v claude >/dev/null && ok 'claude present' || bad 'claude missing'
command -v nvidia-smi >/dev/null && ok 'nvidia-smi present' || bad 'nvidia-smi missing'
command -v nvcc >/dev/null && ok 'nvcc present' || warn 'nvcc missing — toolkit build/probes may be blocked'
command -v nsys >/dev/null && ok 'nsys present' || warn 'nsys missing — profiling tasks will be blocked'

if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | sed 's/^/GPU: /'
  busy="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed '/^$/d' | wc -l)"
  [ "$busy" -eq 0 ] && ok 'GPU is idle' || warn "$busy CUDA process(es) already running"
fi

origin="$(git remote get-url origin 2>/dev/null || true)"
[[ "$origin" =~ ^(git@github\.com:|https://([^@/]+@)?github\.com/)mikepapadim/tornadovm-devoxx2026-cuda-demos(\.git)?/?$ ]] && ok 'origin matches publication boundary' || bad "origin mismatch: $origin"

for f in CLAUDE.md PLAN.md auto/prompt.md auto/supervisor.sh auto/tasks/00.md; do
  [ -f "$f" ] && ok "$f present" || bad "$f missing"
done

printf '\n%d passed, %d failed\n' "$pass" "$fail"
[ "$fail" -eq 0 ] || exit 1
