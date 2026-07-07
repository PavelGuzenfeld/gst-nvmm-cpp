# Reusable, OpenCV-only motion-analytics components (header-only):
#   dual_homography.hpp      — plane+parallax independent-motion residual (panning camera)
#   low_texture_motion.hpp   — frame-diff restricted to low-texture regions
#   active_region.hpp        — content rect, trimming uniform letterbox/pillarbox bars
#   persistence_gate.hpp     — causal track-before-detect confirmation (pure C++)
#   detection_motion_gate.hpp— detector ∩ independent-motion seed gate (composes the above)
#   motion_magnify.hpp       — Eulerian video magnification (reference impl)
#
# Header-only: nothing to compile here, just expose the include dir. Unit tests live
# in ../tests and are gated on OpenCV being available (have_opencv).
analytics_inc = include_directories('.')
