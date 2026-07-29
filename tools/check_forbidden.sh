#!/usr/bin/env bash
# Leak guard for a public repo: no private names, site paths or ticket ids in tracked
# files or commit messages.
#
#   check_forbidden.sh files      scan tracked file contents
#   check_forbidden.sh messages   scan commit messages for the current range
#   check_forbidden.sh selftest   prove both halves actually reject
#
# Two halves, deliberately:
#
#   DENYLIST ($FORBIDDEN)  specific names no pattern could infer -- internal org and
#                          project names, the mission-domain term, dataset id prefixes.
#                          Always lagging; it only knows leaks we already found.
#
#   SHAPES                 fail-closed patterns for the CLASSES of thing that leak --
#                          absolute home paths, user@host, ticket ids, bare IPv4.
#                          Catches identifiers nobody has thought of yet, which is the
#                          case the denylist structurally cannot cover.
#
# Run it locally exactly as CI does before pushing.
set -euo pipefail

: "${FORBIDDEN:=rocx|thebandofficial|fire_arrow|mission-control|BZM-[0-9]|bzm-[0-9]|/home/nvidia|antiuav|anti-uav|wg2022|3700000000002|10\.0\.0\.41|drone}"

# Shape patterns. Keep each one narrow enough not to fire on legitimate prose.
#   /home/<user>/         a real developer's tree, never valid in committed code
#   user@host / user@ip   ssh targets
#   ABC-1234              ticket ids (>=2 letters so it misses e.g. "NV12-1")
#   bare IPv4             excluding 0.0.0.0 and 127.0.0.1, which are legitimate
SHAPES='(^|[^A-Za-z0-9_])/home/[a-z][a-z0-9_-]+/|[a-z][a-z0-9_.-]*@([a-z][a-z0-9.-]*\.[a-z]{2,}|([0-9]{1,3}\.){3}[0-9]{1,3})|\b[A-Z]{2,}-[0-9]{2,}\b|\b(([0-9]{1,3})\.){3}[0-9]{1,3}\b'

# Files worth scanning. meson.build has no extension and carries subdir names, which
# is how a directory named after a private term slips past an extension-only filter.
list_files() {
  git ls-files \
    | grep -E '\.(cpp|hpp|h|cu|md|sh|py|yml|yaml|txt|build|json|cmake)$|(^|/)meson\.build$' \
    | grep -v '^\.github/' \
    | grep -v '^tools/check_forbidden\.sh$'   # this file names the terms it forbids
}

# Known-good matches that the shapes cannot distinguish structurally:
#   - four-part VERSION numbers are indistinguishable from IPv4 (TensorRT 10.3.0.30),
#     so exclude them by the product name that precedes them;
#   - loopback / unspecified / netmask literals are fine to commit;
#   - dataset and standard names collide with the ticket-id shape (COCO-80, VOC-2012).
# Keep this list specific. Widening it to silence a real hit defeats the guard.
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

  # Case-SENSITIVE, unlike the denylist: the ticket-id shape is uppercase by
  # convention, and matching case-insensitively turns it into "any word-dash-number",
  # which fires on rotate-90, batch-30, radius-15 and every similar identifier.
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

# Range selection is the part that was silently broken: on push, base_ref is empty and
# origin/main..HEAD is empty too, so the check passed without reading anything.
commit_range() {
  if [ -n "${BASE_REF:-}" ]; then
    git fetch --no-tags --quiet origin "$BASE_REF" 2>/dev/null || true
    echo "origin/${BASE_REF}..HEAD"
  elif [ -n "${BEFORE_SHA:-}" ] && [ "$BEFORE_SHA" != "0000000000000000000000000000000000000000" ] \
       && git cat-file -e "$BEFORE_SHA" 2>/dev/null; then
    echo "${BEFORE_SHA}..HEAD"
  else
    echo "HEAD~20..HEAD"   # first push / unknown base: scan a bounded recent window
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

# The guard is only worth anything if it actually rejects. Prove both halves on
# synthetic input rather than trusting that they would.
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
