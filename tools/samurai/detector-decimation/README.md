# Detector-decimation harnesses

Throughput and quality A/B harnesses for `nvmminfer`'s `infer-interval` and
`infer-gate-frames` properties. Findings are written up in
[`detector-decimation-verdict.md`](../../../detector-decimation-verdict.md).

These scripts are **parameterised templates** — asset paths, engine names, clip names
and evaluation-set sequence IDs are supplied through the environment rather than
hardcoded, so the harnesses carry no site-specific data. Every script runs under
`set -u`, so an unset variable fails immediately rather than silently reading the
wrong path.

## Required environment

| variable | meaning |
|---|---|
| `ASSET_DIR` | engines, clips, results and scorecards live here |
| `DEPLOY_SRC` | checkout whose `builddir-fix` holds the deployed plugin build |
| `REPO_SRC` | checkout of this repo used for the reference build |
| `JETSON_HOST` | `user@host` of the target board |
| `EVALSET_ZIP` | archive containing the labelled evaluation sequences |
| `SEQ_LIST` | text file, one evaluation sequence name per line |
| `GT_LABEL` | per-sequence ground-truth label filename inside the archive |
| `SCORER` | scoring script that turns prediction CSVs into a scorecard |
| `LABELS_FILE` | class-label file for the detector engine |
| `ONNX_DIR`, `ONNX_DIR384`, `ONNX_DIR512` | ONNX graph sets, per crop size |
| `SCRATCH_DIR` | scratch for probe logs |
| `REMOTE_HOME`, `LOCAL_BIN` | home and user-bin on the target board |

Engines are referenced as `$ASSET_DIR/trt/detector.engine`,
`detector_ir.engine` and `samurai_consts.bin`; clips as
`$ASSET_DIR/clip1.mp4` … `clip3.mp4`. Rename your own assets to match, or symlink.

Evaluation sequences are referred to as `seq-A` … `seq-M`. Map them to your own
sequence names in `$SEQ_LIST`; the writeup uses the same neutral labels.

## Scripts

| script | what it measures |
|---|---|
| `ab_detector_cost.sh` | detector's share of the frame budget (removal A/B) |
| `interval_sweep.sh` | fps vs `infer-interval` |
| `fps_deployed.sh` | fps at the deployed element list |
| `quality_ab.sh`, `quality_ab_clip.sh` | per-frame emitted-box parity vs an undecimated baseline |
| `explain_nontoggle.py` | attributes parity failures to fusion-flag changes vs real divergence |
| `find_targets.sh` | locates frame-associated high-confidence detections, to pick a verified seed |
| `gt_score_ab.sh`, `gt_score_ab_deployed.sh`, `gt_score_gated.sh` | ground-truth scoring per arm |
| `flush_carry_ab.sh` | `nvmmfusekf` flush-carry experiment (element change reverted; kept as record) |
| `render_overlays.sh` | overlay videos for visual inspection |
| `port_to_deploy.sh` + `infer-props.patch` | port the two `nvmminfer` properties into a deploy checkout: `apply` / `--check` / `--revert`, idempotent |

Parity comparison itself lives in [`../../trajectory_compare.py`](../../trajectory_compare.py).
