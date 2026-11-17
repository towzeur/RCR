# Reproducing the Checkpoints

Run the commands below from the repository root. Use Python 3.10 and install
the training dependencies:

```bash
python -m pip install -r rcr/requirements.txt
```

## Data

Pass the parent data directory with `--data-root`. The loaders expect:

```text
data/
  CIFAR10/cifar-10-batches-py/
  CIFAR100/cifar-100-python/
  TINYIMG/processed/
    x_train_01.npy ... x_train_20.npy
    y_train_01.npy ... y_train_20.npy
    x_val_01.npy   ... x_val_20.npy
    y_val_01.npy   ... y_val_20.npy
```

CIFAR-10 and CIFAR-100 are downloaded automatically if absent. TinyImageNet
uses the processed NumPy format from the Mammoth data loader, not the raw
image-folder layout. Its loader includes a download option for that format.
See [seq_tinyimagenet.py](../rcr/datasets/seq_tinyimagenet.py).

## Training

Each row of [paper_runs.csv](paper_runs.csv) defines one dataset, replay
budget, and seed. For example:

```bash
bash scripts/train_paper_artifact.sh seq_cifar10_buffer200_seed0 \
  --data-root /path/to/data --output-root outputs/paper_runs

bash scripts/train_paper_artifact.sh seq_cifar100_buffer200_seed0 \
  --data-root /path/to/data --output-root outputs/paper_runs

bash scripts/train_paper_artifact.sh seq_tinyimagenet_buffer200_seed0 \
  --data-root /path/to/data --output-root outputs/paper_runs
```

Replace `buffer200` with `buffer500` for the larger replay budget, and
`seed0` with `seed1` or `seed2` for the other seeds. The GPU is selected
with `--device 0`; use `--python /path/to/python` to choose an environment.
Add `--dry-run` to inspect a command before starting it.

All runs use ResNet-18, batch size 32, and SGD with zero momentum and weight
decay. Classes are introduced in dataset label order, without permutation.
The learning rate is multiplied by 0.1 at each milestone.

| Dataset | Tasks | Classes/task | Epochs/task | Learning rate | Milestones | Alignment / distillation weights |
|---|---:|---:|---:|---:|---|---|
| Seq-CIFAR-10, buffer 200 | 5 | 2 | 50 | 0.01 | 35, 45 | 13 / 90 |
| Seq-CIFAR-10, buffer 500 | 5 | 2 | 50 | 0.01 | 35, 45 | 12 / 80 |
| Seq-CIFAR-100 | 10 | 10 | 50 | 0.03 | 35, 45 | 18 / 170 |
| Seq-TinyImageNet | 10 | 20 | 100 | 0.03 | 45, 75, 95 | 20 / 165 |

To check installation before a full run, add `--epochs 1 --stop-after 1`.
This uses a shorter training budget and is not a paper experiment.

## Outputs

For a run named `<run>`, the script creates:

```text
outputs/paper_runs/<run>/
  command.txt
  log.txt
  artifacts/artifact_task1.pt
  ...
  results/
```

The framework also saves a training checkpoint after each task under
`rcr/checkpoints/`. Filenames contain the run name, timestamp, and task
index (zero-based). Feature artifacts use one-based task indices and contain
the ETF matrix, transported prototypes, radii, and evaluation features.

The downloadable `checkpoints/<run>/final.pt` files are inference exports:
they omit optimizer state and replay images. Use
`rcr.rcr.load_inference_checkpoint` to load them.

## Evaluation

After training, compute the ETF and RCR results with:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python scripts/evaluate_artifacts.py \
  outputs/paper_runs/seq_cifar100_buffer200_seed0 \
  --heads etf rcr --out-dir outputs/cifar100_buffer200_seed0_metrics
```

To evaluate all three seeds, use `seq_cifar100_buffer200_seed*` instead.
The output CSVs report Class-IL and Task-IL FAA/FF, final route accuracy,
and per-task metrics.

`rcr` scores are recomputed from features and class statistics using the
fixed parameters in [rcr.py](../rcr/rcr.py). Additional score-export options
in the CSV do not override this classifier. The `source_run` column links
the run to its checkpoint identifier and is not a training parameter.
