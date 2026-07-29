#!/usr/bin/env bash
# Behavioural-parity gate on the default clip, at the deployed max-kf=2.
#
# This was a copy of quality_ab_clip.sh's body with the clip and seed hardcoded, which
# is how the two drifted apart. It now delegates, so the logic lives in one place.
#
# Diffs nvmmfusekf's EMITTED per-frame box -- what downstream actually receives, and
# where decimation's cost lands -- for interval 2 and 3 against interval 1.
# Bar: per-frame IoU >= 0.99 median, no frame < 0.9, no valid-flag flips.
exec "$(dirname "$0")/quality_ab_clip.sh" \
  "${CLIP:-clip1}" "${SEED_ROI:-910,490,120,120}" "${SEED_DELAY:-0}"
