# Scripts

Download checkpoints and all task artifacts for a dataset/budget:

```bash
python scripts/download_artifacts.py --dataset cifar100 --buffer 200
```

The downloader checks SHA-256 before extraction. Downloads and evaluation
do not require a GitHub account.

## Evaluate saved artifacts

`evaluate_artifacts.py` recomputes classifier-level metrics from saved
`artifact_task*.pt` files. It does not load the training framework or a model
checkpoint; it only needs the saved test features, labels, ETF scores,
transported prototypes, and replay-estimated radii.

RCR is implemented in [rcr.py](../rcr/rcr.py).

Example:

```bash
python scripts/evaluate_artifacts.py artifacts/paper_runs/* \
  --heads etf rcr \
  --out-dir artifacts/evaluations/paper_runs
```

If your default Python does not include PyTorch, pass an environment explicitly:

```bash
PYTHON=/path/to/python bash scripts/recompute_paper_metrics.sh
```

The output files are:

- `artifact_task_metrics.csv`: Class-IL, Task-IL, route accuracy, wrong-task
  error, and same-task error after each task.
- `artifact_run_summary.csv`: final FAA/FF summary for each evaluated head.

Available heads:

- `etf`: nearest-ETF ProNC classifier.
- `rcr`: RCR scores recomputed from features, prototypes, and radii.
- `ce`, `ldc`, `radial`: saved linear, nearest-centroid, and radial scores,
  when present.
- `saved_mix`: the score mixture stored in the artifact. Use `rcr`, not
  `saved_mix`, to reproduce the reported method.

## Regenerate paper artifacts

`train_paper_artifact.sh` trains a run defined in `configs/paper_runs.csv`.
See [Reproducing the checkpoints](../configs/README.md) for the data layout
and all training settings.

Example:

```bash
PYTHON=/path/to/python bash scripts/train_paper_artifact.sh seq_cifar100_buffer200_seed0 \
  --data-root ./data \
  --output-root outputs/paper_runs
```

Add `--dry-run` to print the fully resolved command without starting training.
For a quick smoke test, add `--epochs 1` and optionally `--stop-after N`.

The script saves:

- `command.txt`: resolved training parameters.
- `log.txt`: training and evaluation output.
- `artifacts/artifact_task*.pt`: tensors used for offline `s_ETF` and `s_RCR`
  evaluation.
- task checkpoints through the framework's `--savecheck task` option.

After a run finishes, recompute the table metrics with:

```bash
PYTHON=/path/to/python bash scripts/recompute_paper_metrics.sh outputs/paper_runs artifacts/evaluations/new_run
```

## Packaging a release

`prepare_release_assets.py` packages checkpoints and feature artifacts;
`upload_release.py` uploads the archives to GitHub Releases. These scripts
are not needed for training or evaluation. The packager reads training
checkpoints containing Python objects: use it only with trusted files.
