# analytics — reusable motion-analytics components

Header-only, OpenCV-only building blocks for finding and confirming an
**independently-moving object** under a **panning/translating camera**, where a
plain detector also fires on static structure. No GStreamer/CUDA dependency — each
header stands alone and unit-tests on the host (see `../tests/test_analytics_*.cpp`).

| header | what it does |
|---|---|
| `dual_homography.hpp` | Plane+parallax independent-motion residual: fit the dominant background plane (H1, RANSAC) and a second parallax plane (H2 on the outliers), keep the residual unexplained by **both** → suppresses flat background *and* near-field parallax that defeat single-homography subtraction. |
| `low_texture_motion.hpp` | Frame-difference restricted to **low-texture** regions (open sky/water/smooth wall) — the complementary case the homography residual can't cover (no features there). |
| `active_region.hpp` | Content rectangle of a frame, trimming uniform **letterbox/pillarbox** bars. Range-based (max−min per row/col), so it stays correct on a dark-but-textured (e.g. thermal) frame where a brightness threshold would fail. |
| `persistence_gate.hpp` | Causal **track-before-detect** confirmation: commit only to a detection that persists *and* carries caller-supplied "support" for K consecutive frames, then latch onto it. Pure C++ (no OpenCV). |
| `detection_motion_gate.hpp` | The composition: detector boxes **∩ independent motion** (dual-homography or low-texture) → persistence gate → the index of the confirmed moving detection. Self-latching (the heavy motion step runs only while searching). |
| `motion_magnify.hpp` | Eulerian video magnification (Wu et al. 2012, linear/IIR variant), streaming. Reference tool. **Caveat:** magnifies small periodic motion against a static/slow background; it does *not* survive large camera motion/parallax — for that use `dual_homography.hpp`. |

These are distinct from `gst/common/nvmm_motion.hpp`, which scores detector boxes
from a **precomputed optical-flow field**; the components here work directly on the
raw grayscale frames.

## Build

Header-only — just add `analytics_inc` to a target's `include_directories`. The unit
tests build automatically when OpenCV is present (`have_opencv` in the top
`meson.build`); a host/CI build without OpenCV configures and runs the rest fine
(the pure-C++ `persistence_gate` test still builds).
