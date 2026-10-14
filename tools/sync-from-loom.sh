#!/bin/bash
# tools/sync-from-loom.sh — one-way sync: Loom's crates/loom-swarm/src -> this repo's src/.
#
# The crate is developed inside the Loom workspace (private) and published from here. Loom's main.rs declares the
# modules (`mod x;`); here they live in the library (src/lib.rs, hand-maintained) and main.rs imports the ones it
# uses (`use loom_swarm::x;`). Everything else in src/ is copied verbatim. Cargo.toml is NOT synced: this repo owns
# the package metadata and feature list; the script diffs dependency and feature names and warns.
#
# Usage: tools/sync-from-loom.sh [--loom DIR] [--ref REF] [--no-commit]
#   --loom DIR     Loom checkout (default $LOOM_DIR or ~/Code/Loom); any worktree of it works
#   --ref REF      branch/sha to sync from (default feat/loom-swarm)
#   --no-commit    stage only; print the commit message that would have been used
# Records the Loom sha in .loom-sync so the next run can list what changed upstream in the commit body.
set -euo pipefail
LOOM=${LOOM_DIR:-$HOME/Code/Loom}; REF=feat/loom-swarm; COMMIT=1; CRATE=crates/loom-swarm
while [ $# -gt 0 ]; do case $1 in
  --loom) LOOM=$2; shift 2;; --ref) REF=$2; shift 2;; --no-commit) COMMIT=0; shift;;
  *) echo "bad arg $1" >&2; exit 2;; esac; done
CARGO=${CARGO:-cargo}
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT"
[ -z "$(git status --porcelain -- src)" ] || { echo "src/ has uncommitted changes; commit or stash first" >&2; exit 1; }

FULL=$(git -C "$LOOM" rev-parse "$REF^{commit}"); SHA=${FULL:0:7}
PREV=$(cat .loom-sync 2>/dev/null || true)
echo "sync $CRATE from $LOOM @ $REF ($SHA)${PREV:+, previously ${PREV:0:7}}"

# 1. modules Loom's main.rs declares, and which of them main.rs itself references
mapfile -t MODS < <(git -C "$LOOM" show "$FULL:$CRATE/src/main.rs" | sed -nE 's/^mod ([a-z_]+);$/\1/p')
[ ${#MODS[@]} -gt 0 ] || { echo "no 'mod x;' lines in Loom main.rs — layout changed, update this script" >&2; exit 1; }

# 2. copy every src file except lib.rs (ours) and main.rs (rewritten)
for f in $(git -C "$LOOM" ls-tree --name-only "$FULL" -- "$CRATE/src/" | xargs -n1 basename); do
  case $f in lib.rs|main.rs) continue;; esac
  git -C "$LOOM" show "$FULL:$CRATE/src/$f" > "src/$f"
done
# files we have that Loom no longer has (other than lib.rs) are removed
for f in src/*.rs; do b=$(basename "$f"); case $b in lib.rs|main.rs) continue;; esac
  git -C "$LOOM" cat-file -e "$FULL:$CRATE/src/$b" 2>/dev/null || { echo "  removing $f (gone from Loom)"; git rm -q "$f"; }; done

# 3. main.rs: `mod x;` (plus a #[cfg] line directly above it) becomes `use loom_swarm::x;` when main.rs uses x::,
#    or disappears when it does not (the library gates and owns the module).
git -C "$LOOM" show "$FULL:$CRATE/src/main.rs" | python3 -I -c '
import re, sys
src = sys.stdin.read()
mods = re.findall(r"^mod ([a-z_]+);$", src, re.M)
body = re.sub(r"^(#\[cfg\([^\n]*\)\]\n)?mod [a-z_]+;\n", "", src, flags=re.M)
uses = "".join(f"use loom_swarm::{m};\n" for m in mods if re.search(rf"\b{m}::", body))
# place the use lines where the first mod line was (after the crate doc comment)
i = re.search(r"^(#\[cfg\([^\n]*\)\]\n)?mod [a-z_]+;", src, re.M).start()
print(src[:i] + uses + body[i:], end="")
' > src/main.rs

# 4. the library must declare exactly the module set Loom declares
LIB=$(sed -nE 's/^pub mod ([a-z_]+);$/\1/p' src/lib.rs | sort | tr '\n' ' ')
WANT=$(printf '%s\n' "${MODS[@]}" | sort | tr '\n' ' ')
[ "$LIB" = "$WANT" ] || { echo "src/lib.rs declares [$LIB] but Loom main.rs declares [$WANT]; edit lib.rs (and its #[cfg] gates) by hand, then rerun" >&2; exit 1; }

# 5. dependency / feature drift (names only; versions are ours)
names() { sed -nE "/^\[$1\]/,/^\[/{s/^([A-Za-z0-9_-]+) *=.*/\1/p}" | sort; }
for sec in dependencies features; do
  d=$(diff <(git -C "$LOOM" show "$FULL:$CRATE/Cargo.toml" | names $sec) <(names $sec < Cargo.toml) || true)
  [ -z "$d" ] || printf 'WARNING: [%s] differ (< Loom, > here):\n%s\n' "$sec" "$d"
done

# 6. build and test as published
$CARGO build --release 2>&1 | tail -2
$CARGO test 2>&1 | grep -E "^test result|FAILED|panicked" || true
$CARGO test 2>&1 | grep -q "^test result: ok" || { echo "tests failed; nothing committed" >&2; exit 1; }

echo "$FULL" > .loom-sync
git add src .loom-sync
if git diff --cached --quiet; then echo "already in step with $SHA"; exit 0; fi
MSG="sync from Loom $CRATE @ $SHA"
BODY=$( [ -n "$PREV" ] && git -C "$LOOM" log --no-merges --format='- %h %s' "$PREV..$FULL" -- "$CRATE" || true )
if [ $COMMIT = 1 ]; then git commit -q -m "$MSG" ${BODY:+-m "$BODY"}; git log -1 --stat | head -20
else printf 'staged. commit message would be:\n%s\n%s\n' "$MSG" "$BODY"; fi
