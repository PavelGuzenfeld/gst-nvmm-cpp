#!/usr/bin/env bash
# Public-repo leak guard. $FORBIDDEN only knows leaks already found; SHAPES fail closed on
# the classes that leak (home paths, user@host, ticket ids, bare IPv4), catching new ones too.
set -euo pipefail

: "${FORBIDDEN:=rocx|thebandofficial|fire_arrow|mission-control|BZM-[0-9]|bzm-[0-9]|/home/nvidia|antiuav|anti-uav|wg2022|3700000000002|10\.0\.0\.41|drone}"

# Ticket ids need >= 2 letters so "NV12-1" passes. Keep every shape narrow enough for prose.
SHAPES='(^|[^A-Za-z0-9_])/home/[a-z][a-z0-9_-]+/|[a-z][a-z0-9_.-]*@([a-z][a-z0-9.-]*\.[a-z]{2,}|([0-9]{1,3}\.){3}[0-9]{1,3})|\b[A-Z]{2,}-[0-9]{2,}\b|\b(([0-9]{1,3})\.){3}[0-9]{1,3}\b'

# meson.build has no extension yet names subdirs: an extension-only filter misses a leaked dir name.
list_files() {
  git ls-files \
    | grep -E '\.(cpp|hpp|h|cu|md|sh|py|yml|yaml|txt|build|json|cmake)$|(^|/)meson\.build$' \
    | grep -v '^\.github/' \
    | grep -v '^tools/check_forbidden\.sh$'   # this file names the terms it forbids
}

# Product names before four-part versions, loopback/netmask literals, and dataset or standard
# ids (COCO-80) that the shapes cannot tell apart. Widening this to silence a real hit defeats it.
drop_benign() {
  grep -viE 'version|tensorrt|cuda|jetpack|l4t|cudnn|driver|0\.0\.0\.0|127\.0\.0\.1|255\.255|\b(COCO|IMAGENET|VOC|MNIST|CIFAR|KITTI|NUSCENES|UTF|ISO|RFC|SHA|AES|ITU|IEC)-[0-9]+' || true
}

fail=0

scan_files() {
  local hits
  hits=$(list_files | xargs -r grep -inE "$FORBIDDEN" 2>/dev/null || true)
  if [ -n "$hits" ]; then
    echo "ERROR: forbidden reference (known private name) in tracked files:"
    echo "$hits" | head -20 | sed 's/^/  /'
    fail=1
  fi

  # Case-sensitive: case-folded, the uppercase ticket-id shape fires on rotate-90 and batch-30.
  hits=$(list_files | xargs -r grep -nE "$SHAPES" 2>/dev/null | drop_benign)
  if [ -n "$hits" ]; then
    echo "ERROR: forbidden SHAPE (home path, user@host, ticket id, or IP) in tracked files:"
    echo "$hits" | head -20 | sed 's/^/  /'
    echo "  If one of these is legitimate, narrow the pattern -- do not delete the check."
    fail=1
  fi

  [ "$fail" -eq 0 ] && echo "OK: no forbidden references or shapes in tracked files"
  return "$fail"
}

# On push base_ref is empty and origin/main..HEAD is too, which once passed without reading a commit.
commit_range() {
  if [ -n "${BASE_REF:-}" ]; then
    git fetch --no-tags --quiet origin "$BASE_REF" 2>/dev/null || true
    echo "origin/${BASE_REF}..HEAD"
  elif [ -n "${BEFORE_SHA:-}" ] && [ "$BEFORE_SHA" != "0000000000000000000000000000000000000000" ] \
       && git cat-file -e "$BEFORE_SHA" 2>/dev/null; then
    echo "${BEFORE_SHA}..HEAD"
  else
    echo "HEAD~20..HEAD"
  fi
}

scan_messages() {
  local range hits
  range=$(commit_range)
  if ! git rev-parse "${range%%..*}" >/dev/null 2>&1; then
    echo "OK: base of '$range' unavailable, nothing to compare"; return 0
  fi
  echo "scanning commit messages over $range"
  hits=$(git log --format='%B' "$range" | grep -inE "$FORBIDDEN" || true)
  if [ -n "$hits" ]; then
    echo "ERROR: forbidden reference in a commit message:"; echo "$hits" | head -20 | sed 's/^/  /'
    echo "  Messages are far harder to scrub once pushed -- amend before merging."
    return 1
  fi
  echo "OK: no forbidden references in commit messages"
}

selftest() {
  local tmp rc=0
  tmp=$(mktemp -d); trap 'rm -rf "$tmp"' RETURN
  printf 'const char *s = "rocx";\n'            > "$tmp/deny.cpp"
  printf 'path = "/home/someone/assets"\n'      > "$tmp/shape_home.py"
  printf '# see ABCD-1234 for context\n'        > "$tmp/shape_ticket.md"
  printf 'ssh user@192.168.1.10\n'              > "$tmp/shape_host.sh"
  printf 'v = "1.2.3.4-rc"  # a version\n'      > "$tmp/benign.txt"

  printf 'nvmmconvert flip-method=rotate-90 batch-30 radius-15\n' > "$tmp/benign2.txt"

  for f in deny.cpp shape_home.py shape_ticket.md shape_host.sh; do
    if grep -qiE "$FORBIDDEN" "$tmp/$f" || grep -qE "$SHAPES" "$tmp/$f"; then
      echo "  ok   rejects $f"
    else
      echo "  FAIL misses $f"; rc=1
    fi
  done
  for f in benign.txt benign2.txt; do
    if grep -nE "$SHAPES" "$tmp/$f" | drop_benign | grep -q .; then
      echo "  FAIL false-positive on $f"; rc=1
    else
      echo "  ok   passes $f"
    fi
  done
  [ "$rc" -eq 0 ] && echo "selftest OK" || echo "selftest FAILED"
  return "$rc"
}

case "${1:-files}" in
  files)    scan_files ;;
  messages) scan_messages ;;
  selftest) selftest ;;
  *) echo "usage: $0 {files|messages|selftest}" >&2; exit 2 ;;
esac
