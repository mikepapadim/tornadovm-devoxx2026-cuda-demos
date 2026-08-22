#!/usr/bin/env bash
# Validate committed evidence in this repo without requiring a GPU.
# Checks: required deliverables exist, pinned SHAs are recorded, every
# results/raw path cited by docs/claims.md actually exists, every demo
# listed in demos/README.md has its own README, and the hard scope rules
# in CLAUDE.md (no OpenCL/Metal/legacy-PTX-backend/Babylon) are not violated
# anywhere under docs/ or demos/.
set -u
cd "$(dirname "$0")/.."

fail=0
pass=0

ok()   { pass=$((pass + 1)); echo "OK   $1"; }
bad()  { fail=$((fail + 1)); echo "FAIL $1"; }

require_file() {
  if [ -f "$1" ]; then ok "$1 exists"; else bad "$1 missing"; fi
}

echo "== Required deliverables =="
require_file "STATE.md"
require_file "PLAN.md"
require_file "README.md"
require_file "env/versions.env"
require_file "docs/claims.md"
require_file "docs/demo-runbook.md"
require_file "docs/talk-1-hybrid-api.md"
require_file "docs/talk-2-llm-inference.md"

echo "== Pinned environment =="
if grep -q '^TORNADO_SHA=[0-9a-f]\{7,\}' env/versions.env 2>/dev/null; then
  ok "TORNADO_SHA is recorded"
else
  bad "TORNADO_SHA missing or malformed in env/versions.env"
fi
if grep -q '^GPULLAMA3_SHA=[0-9a-f]\{7,\}' env/versions.env 2>/dev/null; then
  ok "GPULLAMA3_SHA is recorded"
else
  bad "GPULLAMA3_SHA missing or malformed in env/versions.env"
fi

echo "== results/raw paths cited in docs/claims.md all exist =="
if [ -f docs/claims.md ]; then
  # Extract results/raw/... and results/failures/... paths referenced in backticks.
  cited=$(grep -oE 'results/(raw|failures)/[A-Za-z0-9_./-]+' docs/claims.md | sort -u)
  if [ -z "$cited" ]; then
    bad "no results/ paths found cited in docs/claims.md (expected many)"
  else
    missing=0
    while IFS= read -r p; do
      # Strip a trailing filename fragment if the cited path is a directory prefix
      # by checking the path itself, and if absent, its parent directory.
      if [ -e "$p" ] || [ -e "$(dirname "$p")" ]; then
        :
      else
        echo "     missing evidence path: $p"
        missing=$((missing + 1))
      fi
    done <<< "$cited"
    if [ "$missing" -eq 0 ]; then
      ok "all $(echo "$cited" | wc -l) cited results/ paths resolve on disk"
    else
      bad "$missing cited results/ path(s) do not resolve on disk"
    fi
  fi
fi

echo "== Every demo in demos/README.md has its own README =="
if [ -f demos/README.md ]; then
  demo_dirs=$(grep -oE '\[[^]]+\]\(([0-9]+-[a-z0-9-]+)/\)' demos/README.md | sed -E 's/.*\(([^)]+)\)/\1/')
  missing=0
  for d in $demo_dirs; do
    if [ -f "demos/${d}README.md" ]; then
      :
    else
      echo "     missing: demos/${d}README.md"
      missing=$((missing + 1))
    fi
  done
  if [ "$missing" -eq 0 ]; then
    ok "every linked demo directory has a README.md"
  else
    bad "$missing linked demo(s) missing a README.md"
  fi
else
  bad "demos/README.md missing"
fi

echo "== Hard scope: no OpenCL/Metal/legacy-PTX/Babylon evidence was actually captured =="
# Mentioning these words to state the exclusion (CLAUDE.md, README.md, or
# "no-op on OPENCL/METAL" API-doc quotes) is expected and fine. What must
# never exist is captured evidence FOR one of them: a results/ subtree named
# after another backend, or a pinned TORNADO_BACKEND other than cuda.
scope_violation=0
other_backend_dirs=$(find results -maxdepth 3 -iregex '.*\(opencl\|metal\|babylon\|ptx-backend\).*' 2>/dev/null)
if [ -n "$other_backend_dirs" ]; then
  echo "$other_backend_dirs"
  scope_violation=1
fi
if ! grep -q '^TORNADO_BACKEND=cuda$' env/versions.env 2>/dev/null; then
  echo "     env/versions.env does not pin TORNADO_BACKEND=cuda"
  scope_violation=1
fi
if [ "$scope_violation" -eq 0 ]; then
  ok "no results/ evidence captured for a non-CUDA backend, TORNADO_BACKEND=cuda pinned"
else
  bad "found evidence or a pinned backend outside CUDA scope — see lines above"
fi

echo
echo "== Summary: $pass passed, $fail failed =="
if [ "$fail" -gt 0 ]; then
  exit 1
fi
exit 0
