# RCR Training Code

This directory contains the RCR implementation and the training framework
used for the paper's experiments, based on
[ProNC](https://github.com/Continue-Edge-AI-Lab/ProNC) and
[Mammoth](https://github.com/aimagelab/mammoth).

## Code

- `rcr.py`: RCR scoring and statistics: feature-frame transport,
  class radii, NC shrinkage, and final scoring.
- `models/exorth_ce_svd.py`: ProNC baseline with task-route reporting support.
- `models/exorth_ce_svd_ldcncm.py`: Mammoth integration for the RCR transport
  component.
- `models/exorth_ce_svd_ldcrad.py`: Mammoth integration for RCR radii,
  inference, and feature export.
- `backbone/`, `datasets/`, `utils/`: framework components from
  the public ProNC implementation.

## Reproducing Results

Checkpoints and per-task feature artifacts are available in the
[release](https://github.com/towzeur/RCR/releases/tag/v1.0.0).

Use [evaluate_artifacts.py](../scripts/evaluate_artifacts.py) to recompute
FAA, FF, Task-IL, and route accuracy from the saved artifacts without
retraining. See the [repository README](../README.md) for installation and
downloads, and [Reproducing the checkpoints](../configs/README.md) for
training commands.
