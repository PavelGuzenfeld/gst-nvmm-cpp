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

## Enabling & building

The components are **header-only and always available** — just include them. Their
**unit tests and the OpenCV dependency are opt-in** via a meson feature that is
**OFF by default**, so the normal (zero-copy NVMM) build pulls no OpenCV at all:

```sh
meson setup build -Danalytics=enabled     # build the analytics tests (requires OpenCV)
meson test  -C build analytics_dual_homography analytics_detection_motion_gate \
            analytics_persistence_gate analytics_low_texture_motion \
            analytics_active_region analytics_motion_magnify
```

With the default `-Danalytics=disabled`, OpenCV is never searched and no analytics
test is built; the headers stay on the include path for anyone who pulls them in.

## Integrating

Header-only, OpenCV-only (no GStreamer/CUDA). In a consumer target add the include
dir and link OpenCV:

```cpp
#include "dual_homography.hpp"   // add analytics/ (analytics_inc) to include_directories
```
```meson
dependencies : [dependency('opencv4')]
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
