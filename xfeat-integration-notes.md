# XFeat integration — why it is not a `gmc-backend` value

Design note for the remaining half of the XFeat work. The foundation
(`gst/common/xfeat_*`, `gst/nvmmdetgate`) landed separately because it is additive;
what is left is the part that touches `nvmmsamurai`, and it has been mis-specified
more than once.

## The mis-specification

It was proposed repeatedly — including in this repo's own PR descriptions — that
XFeat should become a fifth value on the existing `gmc-backend` property, alongside
`ncc` / `fft-cpu` / `fft-cuda` / `pva`, "rather than deleting the working backends".
That is wrong, and the interface says so plainly.

`gmc_backend.hpp` models **patch-based translation estimators**. The dispatch in
`samurai_tracker.cpp` is:

```
GmcShift gmc_estimate(const uint8_t *prev, const uint8_t *curr)
```

— two 128×128 grayscale patches (256 for `pva`), VIC-downscaled from the frame's
centre square, returning `{dx, dy, conf}`. Every existing backend fits because every
existing backend is that shape.

XFeat is not:

- it consumes the **full `NvBufSurface`**, resized to the matcher's own input, not a
  128 px centre patch — on a patch that small it finds almost no keypoints;
- inside `nvmmsamurai` the extraction is **shared**: one `extract()` per frame fills
  `cur_feat`, and *both* GMC and track-validity consume it. Expressing GMC alone as a
  backend either duplicates the CNN pass or requires the backend to reach into state
  the enum does not model;
- it returns `MatchPair` lists, from which translation is one derived product
  (`global_translation_median`) among several — the same matches also feed the RANSAC
  background affine and the two-reference residual that validity needs.

So the enum cannot express it without either tripling the per-frame CNN cost or
smuggling shared state through a seam designed to be stateless.

## The shape that is actually right

Two orthogonal axes, not one enum:

1. **Keep `gmc-backend`** exactly as it is, for the patch estimators. It stays the
   default path and keeps its fuzz and sanitizer coverage.
2. **Add a per-frame XFeat feature context** owned by the tracker, enabled by its own
   property (e.g. `xfeat-engine-dir`, off when unset). When active it performs one
   extraction per frame and serves GMC *and* validity from it, bypassing the patch
   path for both. `nvmmdetgate` is a separate element and already owns its own
   matcher, so it is unaffected either way.

This is close to what the original branch built. Its faults were not the design: it
deleted `samurai_gmc.hpp` and the four working backends, and it was cut from a base
predating them.

## Why this is not done yet

It is a real integration, not a rename: two implementations of GMC and of validity
have to coexist behind one switch, and the branch holding the XFeat versions is based
on a tree that no longer exists. It also cannot be compiled anywhere but the target.

The honest prerequisite is a decision about whether the XFeat path is meant to
*replace* the patch backends once proven, or coexist permanently. Coexisting
permanently means maintaining two GMC implementations and two validity
implementations indefinitely, which is a cost worth naming before starting rather
than discovering afterwards.
