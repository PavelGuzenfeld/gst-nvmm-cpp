#!/usr/bin/env bash
# Idempotent port of infer-props.patch into a deploy checkout that is not a git repo:
# patch(1) matches hunks by context with fuzz and reverses cleanly, where literal anchors drifted.
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
