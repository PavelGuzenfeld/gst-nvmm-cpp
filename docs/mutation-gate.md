# Mutation gate

A green suite and a blind suite look identical from the outside. Nothing in this
repo measured the difference until now.

The gate rewrites one operator or one numeric literal on a line the diff
touched, rebuilds, and runs the suite. A mutant the suite still passes is a hole
in the tests, and it blocks the commit until a test kills it or a committed
waiver in `.mutation-gate-waivers.toml` says why none can. That file does not
exist yet; the gate prints a ready-to-paste stanza when it blocks. Add
`line = <n>` to a stanza waiving a numeric literal — the gate refuses a bare
literal without one, because `old = "0"` alone would waive every `0 => 1` in the
file. It runs from a
pre-commit hook and from a Claude Code Stop hook; whichever runs first satisfies
the other. Spec and the decisions behind it: `PavelGuzenfeld/dotfiles#9`.

## Running it

```bash
pip install pre-commit && pre-commit install   # then it rides `git commit`
mutation-gate --worktree                       # on demand, unstaged changes
mutation-gate --file gst/common/kalman_box.cpp # every line of one file
mutation-gate --file analytics/image_ops.hpp --dry-run   # count, run nothing
```

It needs `ast-grep` on `PATH` and a built `gst-nvmm-cpp:dev` image:

```bash
docker build -f docker/Dockerfile.dev -t gst-nvmm-cpp:dev .
```

The gate configures and reuses `builddir-gate/` inside that image
(`-Danalytics=enabled`, C++14, debug). Nothing else in the tree writes there.

## What it costs

Measured in `gst-nvmm-cpp:dev` on a 20-core x86 host, mock `NvBufSurface`:

| Step | Wall |
|------|------|
| Clean configure + build (107 edges) | 2.1 s |
| Rebuild after touching `analytics/image_ops.hpp` | 0.78 s |
| Rebuild after touching `gst/common/nvmm_buffer.hpp` | 0.53 s |
| Rebuild after touching `gst/common/kalman_box.hpp` | 0.25 s |
| Suite, all 32 tests | 14.3 s |
| Suite, `--no-suite fuzz` (30 tests) | 2.4 s |
| Full gate command end to end (docker + build + 30 tests) | 1.9 s |

The two fuzz tests are 12.4 s of that 14.3 s and re-seed every run, so the gate
excludes the `fuzz` suite. CI still runs them — `ci.yml` calls plain `meson
test`, which takes every suite.

This is why there is no per-line test selection here. `coverage.py` contexts are
the gate's only narrowing mechanism and there is no C++ equivalent wired, so
every mutant runs the whole suite. At ~3 s a mutant that is the cheaper answer,
and a selection heuristic that missed the killing test would report a false
SURVIVED.

## Measured kill rate

`gst/common/kalman_box.cpp`, every line, 165 mutants: **142 killed, 23 survived
— 86%**. Baseline suite 2.3 s, whole sweep ~28 min.

Seven of the 23 enlarge a fixed-size array declaration — `const double std[8]`
to `[9]`, `std::array<double, 4> z{}` to `<double, 5>`. Every loop over them is
bounded by its own literal, so the extra slot is written by nobody and read by
nobody. Those are equivalent mutants and no test can kill them.

The other 16 point at `tests/test_kalman_box.cpp`. Two properties of that file
stand out:

- `check_cov_diag` reads `c[i][i]` and nothing else, so the off-diagonal half of
  the covariance has no oracle at all. The mutants confined to it survive — the
  inner `k < j` bound in `chol4`, the `L[j][j] != 0.0` guard, the `k = i + 1`
  start of the back-substitution.
- Every case is a well-conditioned filter driven through `initiate`, `predict`
  and `update`. Nothing constructs a singular or non-positive-definite
  covariance, so `s > 0.0` is never evaluated at its boundary and `s >= 0.0`
  returns the same value everywhere the test looks.

Neither group is a defect in the gate. The second is the hole it exists to find.

## What it cannot see

The gate is only as wide as the image it builds in. `gst-nvmm-cpp:dev` has no
TensorRT, VPI or CUDA toolkit, so `meson.build` never descends into
`nvmmofa`, `nvmminfer`, `nvmmsecondaryinfer`, `nvmmdrawdet`, `nvmmsamurai` or
`nvmmdetgate`, and `probes/` has no meson target at all. A mutant in any of them
survives for want of a build, not for want of a test — measured: all 70 mutants
on `gst/nvmmsamurai/samurai_seed_math.hpp` survived for exactly that reason.
Those paths are listed in `exclude_paths` in `.mutation-gate.toml` and the gate
announces each one it skips. Drop an entry when the gate can run in a JetPack
image on-device.

`analytics/` headers are reached only through the component tests
(`-Danalytics=enabled`). The OpenCV golden oracle (`-Danalytics_golden`) does not
build in that image either, so a line only the golden tests exercise is gated
against the component tests alone.

A mutant that fails to compile makes the command non-zero and is counted KILLED.
That is deliberate — it keeps the run moving — but it inflates the kill rate: an
invalid mutant is not evidence a test is watching.
