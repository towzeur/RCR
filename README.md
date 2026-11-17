# When Neural Collapse Fails to Route

Official repository for the BMVC 2026 paper:

**When Neural Collapse Fails to Route: Routing Drift in Continual Learning**

[Project page](https://towzeur.github.io/RCR/) |
[Paper](https://towzeur.github.io/RCR/assets/paper.pdf) |
[Supplement](https://towzeur.github.io/RCR/assets/supplement.pdf)

## Overview

Progressive neural-collapse classifiers can retain strong task-local
discrimination while failing the global Class-IL decision. The paper identifies
this failure mode as routing drift and introduces Reliability-Calibrated Routing
(RCR), a classifier-level correction that keeps the trained backbone and ETF
frame fixed while combining global ETF evidence with local radial reliability.

## Code and Checkpoints

The [v1.0.0 release](https://github.com/towzeur/RCR/releases/tag/v1.0.0)
provides final inference checkpoints and per-task feature artifacts for
CIFAR-10, CIFAR-100, and TinyImageNet, with replay budgets 200/500 and seeds
0, 1, and 2. Downloads are split into six archives, one per dataset/budget.

The checkpoints contain the trained ResNet-18, ETF vertices, transported
prototypes, and class radii. They exclude training images and optimizer state;
they are for inference, not resuming training. The per-task artifacts allow
FAA, FF, Task-IL, and route accuracy to be recomputed without a GPU or dataset
download. SHA-256 checksums and file provenance accompany the release.

### Reproduce Metrics Without Training

Use Python 3.10 or newer. From the repository root:

```bash
python -m pip install -r requirements-eval.txt
python scripts/download_artifacts.py --dataset cifar100 --buffer 200
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python scripts/evaluate_artifacts.py \
  artifacts/paper_runs/seq_cifar100_buffer200_seed* \
  --heads etf rcr --out-dir outputs/cifar100_buffer200
```

For this setting, the three-seed mean Class-IL FAA / FF is
**41.71 / 38.55** for ETF and **49.74 / 16.27** for RCR. The output CSVs
also report Task-IL FAA/FF, route accuracy, and task-by-task error statistics.
Checkpoint and artifact loading uses `weights_only=True`.

### Use the Checkpoints

```python
import torch
from rcr.rcr import load_inference_checkpoint, rcr_scores

net, state = load_inference_checkpoint(
    "checkpoints/seq_cifar100_buffer200_seed0/final.pt"
)
# images: a batch with the dataset's evaluation preprocessing applied.
with torch.no_grad():
    features = net(images, returnt="features")
    scores = rcr_scores(
        features, state["etf"].T, state["ldc_proto"],
        state["ldcrad_class_radius"], state["classes_per_task"]
    )
    predictions = scores.argmax(dim=1)
```

Evaluation transforms are defined in `rcr/datasets/seq_cifar10.py`,
`seq_cifar100.py`, and `seq_tinyimagenet.py`. ETF vertices use the
`[feature_dim, classes]` convention in checkpoints.

### Train and Export New Artifacts

```bash
python -m pip install -r rcr/requirements.txt
bash scripts/train_paper_artifact.sh seq_cifar100_buffer200_seed0 \
  --data-root /path/to/data --output-root outputs/paper_runs
```

Training uses [ProNC](https://github.com/Continue-Edge-AI-Lab/ProNC), built on
[Mammoth](https://github.com/aimagelab/mammoth). The script saves checkpoints
and features after each task; `rcr/rcr.py` computes the RCR scores.
See [Reproducing the checkpoints](configs/README.md) for the data layout,
training settings, and output files.

Run the numerical tests with `python -m unittest discover -s tests -v`.

## Repository Layout

```text
docs/        Project page and paper assets.
rcr/         RCR code and the continual-learning framework used for experiments.
scripts/     Evaluation and artifact-generation entry points.
configs/     Experiment configurations and run metadata.
artifacts/   Reproducibility artifacts and cached evaluation summaries.
```

The paper and supplementary PDFs are mirrored in this repository.

## License

The code is distributed under the MIT license; third-party notices and
exceptions are documented in [rcr/NOTICE.md](rcr/NOTICE.md) and the source files.

## Citation

```bibtex
@inproceedings{larue2026rcr,
  title     = {When Neural Collapse Fails to Route: Routing Drift in Continual Learning},
  author    = {Larue, Nicolas},
  booktitle = {British Machine Vision Conference (BMVC)},
  year      = {2026},
  url       = {https://towzeur.github.io/RCR/}
}
```
