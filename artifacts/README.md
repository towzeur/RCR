# Artifacts

This directory contains saved end-of-task tensors for reproducing classifier
metrics without running a backbone. Download them from the
[GitHub release](https://github.com/towzeur/RCR/releases/tag/v1.0.0), for example:

```bash
python scripts/download_artifacts.py --dataset cifar100 --buffer 200
```

Each archive also extracts three final inference checkpoints into
`checkpoints/<run>/final.pt`, one per seed. These are distinct from the
feature artifacts below: they contain backbone weights, ETF vertices,
transported prototypes, and class radii, but no replay images or optimizer.
They can be loaded with `rcr.rcr.load_inference_checkpoint`.

## Layout

```text
paper_runs/
  seq_cifar10_buffer200_seed0/
    artifact_task1.pt
    ...
  seq_cifar100_buffer500_seed2/
    artifact_task1.pt
    ...
  seq_tinyimagenet_buffer500_seed2/
    artifact_task1.pt
    ...
```

Each `artifact_task*.pt` file contains the held-out test features, labels,
ETF scores, and the RCR state needed to recompute `s_ETF` and `s_RCR`.
If the ETF matrix is absent, the evaluator recovers it from the saved
features and ETF logits without using labels.
The `ldc_proto` tensor stores RCR's transported class prototypes, and
`ldcrad_class_radius` stores the replay-estimated raw class radii.

The release includes all tasks for each dataset/buffer/seed run so that final
average accuracy (FAA) and final forgetting (FF) can be recomputed exactly from
the saved artifacts.

## Training new checkpoints

See [Reproducing the checkpoints](../configs/README.md) for data preparation,
training commands, and settings. The training script exports features and
class statistics after every task.
