#!/usr/bin/env bash
# quality_ab_clip.sh on the default clip and seed, overridable via CLIP, SEED_ROI, SEED_DELAY.
exec "$(dirname "$0")/quality_ab_clip.sh" \
  "${CLIP:-clip1}" "${SEED_ROI:-910,490,120,120}" "${SEED_DELAY:-0}"
