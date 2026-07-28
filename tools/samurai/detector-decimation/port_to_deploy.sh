#!/usr/bin/env bash
# Port infer-interval + infer-gate-frames into a deploy checkout of nvmminfer.
#
#   DEPLOY_SRC=/path/to/checkout port_to_deploy.sh [--revert|--check]
#
# Replaces two hand-written Python patchers that matched nine literal source anchors
# and exited on any drift. Worse, the second depended on text the first had inserted
# -- including re-opening a comment block the first one closed -- so they had a hidden
# ordering coupling and could only ever run in one sequence.
#
# A real unified diff has none of that: `patch` locates hunks by context with fuzz,
# reports precisely which hunk failed, and reverses cleanly. The deploy checkout is
# not a git repo, which is why the porters existed at all -- but `patch` never needed
# one, only `git apply` would have.
#
# Idempotent: a dry-run reverse check detects an already-patched tree and exits 0.
set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
PATCH=$HERE/infer-props.patch
TARGET=${DEPLOY_SRC:?DEPLOY_SRC must point at the deploy checkout}/gst/nvmminfer/gstnvmminfer.cpp
MODE=${1:-apply}

[ -f "$PATCH" ]  || { echo "ERROR: missing $PATCH" >&2; exit 2; }
[ -f "$TARGET" ] || { echo "ERROR: no nvmminfer at $TARGET" >&2; exit 2; }

already_applied() { patch -p0 -R --dry-run -f "$TARGET" < "$PATCH" >/dev/null 2>&1; }

case "$MODE" in
  --check)
    if already_applied; then echo "already patched: $TARGET"; else echo "not patched: $TARGET"; fi
    ;;
  --revert)
    if already_applied; then
      patch -p0 -R "$TARGET" < "$PATCH" && echo "reverted: $TARGET"
    else
      echo "not patched, nothing to revert"
    fi
    ;;
  apply)
    if already_applied; then echo "already patched, nothing to do"; exit 0; fi
    # Backup only on the first apply, so re-running never clobbers the pristine copy.
    [ -f "$TARGET.orig-props" ] || cp "$TARGET" "$TARGET.orig-props"
    if ! patch -p0 --dry-run -f "$TARGET" < "$PATCH" >/dev/null 2>&1; then
      echo "ERROR: patch does not apply cleanly. Failing hunks:" >&2
      patch -p0 --dry-run -f "$TARGET" < "$PATCH" 2>&1 | grep -iE 'hunk|fail' >&2
      exit 1
    fi
    patch -p0 "$TARGET" < "$PATCH" && echo "patched: $TARGET (pristine copy at $TARGET.orig-props)"
    ;;
  *) echo "usage: DEPLOY_SRC=... $0 [apply|--check|--revert]" >&2; exit 2 ;;
esac
