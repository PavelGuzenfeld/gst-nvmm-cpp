# analytics — reusable motion-analytics components

Header-only, dependency-free (pure C++14) building blocks for finding and
confirming an **independently-moving object** under a **panning/translating
camera**, where a plain detector also fires on static structure. No
OpenCV/GStreamer/CUDA — each header stands alone and unit-tests on the host
(see `../tests/test_analytics_*.cpp`).

The implementations are **fused**: instead of mirroring the classic OpenCV op
chains 1:1, each component collapses its pipeline into a few single-pass sweeps
(e.g. Sobel×2+magnitude in one pass; blur→threshold→integral in one pass;
warp+absdiff+valid-mask+erode as one inverse-warp pass). OpenCV survives only
as a **test-time reference oracle** in the opt-in golden-comparison lane
(`-Danalytics_golden=enabled`), which validates every fused stage and component
against it.

| header | what it does |
|---|---|
| `image.hpp` | The buffer currency: owning `Image<T>` + non-owning strided `View<T>` — wrap any strided plane (e.g. a CPU-mapped NVMM luma plane) without a copy. |
| `image_ops.hpp` | Shared fused building blocks: OpenCV-compatible separable Gaussian blur with a fusable emit hook, REFLECT_101 borders, border-zeroing, window max. |
| `dual_homography.hpp` | Plane+parallax independent-motion residual: fit the dominant background plane (H1, RANSAC) and a second parallax plane (H2 on the outliers), keep the residual unexplained by **both** → suppresses flat background *and* near-field parallax that defeat single-homography subtraction. Two selectable feature pipelines: `small_motion` (FAST + bounded ZNCC search — sized to near-consecutive frames) and `orb` (from-scratch pyramid ORB + Hamming KNN). |
| `low_texture_motion.hpp` | Frame-difference restricted to **low-texture** regions (open sky/water/smooth wall) — the complementary case the homography residual can't cover (no features there). |
| `active_region.hpp` | Content rectangle of a frame, trimming uniform **letterbox/pillarbox** bars. Range-based (max−min per row/col), so it stays correct on a dark-but-textured (e.g. thermal) frame where a brightness threshold would fail. One sweep. |
| `persistence_gate.hpp` | Causal **track-before-detect** confirmation: commit only to a detection that persists *and* carries caller-supplied "support" for K consecutive frames, then latch onto it. |
| `detection_motion_gate.hpp` | The composition: detector boxes **∩ independent motion** (dual-homography or low-texture) → persistence gate → the index of the confirmed moving detection. Self-latching (the heavy motion step runs only while searching). |
| `motion_magnify.hpp` | Eulerian video magnification (Wu et al. 2012, linear/IIR variant), streaming, one fused pass per frame. Reference tool. **Caveat:** magnifies small periodic motion against a static/slow background; it does *not* survive large camera motion/parallax — for that use `dual_homography.hpp`. |

These are distinct from `gst/common/nvmm_motion.hpp`, which scores detector boxes
from a **precomputed optical-flow field**; the components here work directly on the
raw grayscale frames.

## Enabling & building

The components are **header-only and always available** — just include them; they
need nothing beyond the C++14 standard library. The meson features only gate
tests:

```sh
meson setup build -Danalytics=enabled          # analytics unit tests (no OpenCV)
meson test  -C build --suite '' analytics_dual_homography analytics_detection_motion_gate \
            analytics_persistence_gate analytics_low_texture_motion \
            analytics_active_region analytics_motion_magnify

meson setup build -Danalytics_golden=enabled   # golden comparisons vs OpenCV (requires OpenCV)
meson test  -C build golden_image_ops golden_low_texture_motion golden_active_region \
            golden_motion_magnify golden_dual_homography
```

With the defaults (both disabled) nothing analytics-related is built and OpenCV is
never searched; the headers stay on the include path for anyone who pulls them in.

## Integrating

Header-only, no deps. In a consumer target just add the include dir:

```cpp
#include "dual_homography.hpp"   // add analytics/ (analytics_inc) to include_directories
```

## Usage

The flagship is the composed gate: feed the detector's boxes plus the current and
two past grayscale frames, get back the index of the one detection that is moving
**independently of the camera** (or −1):

```cpp
nvmm::motion::MovingObjectGate gate;                 // sensible defaults
// each frame:
std::vector<nvmm::track::Detection> boxes = /* {cx, cy, conf, /*supported=*/false} */;
int i = gate.update(boxes, cur_gray, prev_gray, prev2_gray);
if (i >= 0) seed_tracker(boxes[i]);                  // confirmed independent mover
```

Frames are `nvmm::img::View<const uint8_t>` — wrap any 8-bit grayscale plane:

```cpp
nvmm::img::View<const uint8_t> cur_gray(plane_ptr, width, height, stride_in_bytes);
```

The pieces are usable on their own too: `independent_motion_residual()` for a raw
motion map, `active_region()` to trim letterbox/pillarbox before processing,
`PersistenceGate` as a standalone track-before-detect confirmer, `MotionMagnifier`
to reveal subtle periodic motion.

## What it's useful for

Seeding a tracker (SAM2/SAMURAI, …) onto the genuine moving target in a scene where
the detector also fires on **static structure** under a **panning / handheld**
camera — surveillance, inspection, search-and-track, any "find the thing that's
actually moving, not the background" task. The independent-motion cue plus temporal
persistence is what separates a real mover from detector false-positives on clutter.
