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

## Release Status

- Project page: available.
- Camera-ready paper and supplementary PDF: available.
- Minimal RCR scoring code: coming soon.
- Evaluation scripts for frozen checkpoints: coming soon.
- Configs for Seq-CIFAR-10, Seq-CIFAR-100, and Seq-TinyImageNet: coming soon.
- Lightweight result artifacts and table-generation metadata: coming soon.

## Repository Layout

```text
docs/        Project page served by GitHub Pages.
rcr/         RCR scoring code. Coming soon.
scripts/     Evaluation and figure/table generation scripts. Coming soon.
configs/     Dataset and experiment configuration files. Coming soon.
artifacts/   Lightweight generated summaries and metadata. Coming soon.
```

The camera-ready paper and supplementary material are mirrored under
`docs/assets/` so the project page remains self-contained.

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
