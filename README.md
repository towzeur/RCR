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

## Coming Soon

Code, scripts, and reproducibility artifacts will be released here.

## Repository Layout

```text
docs/        Project page and paper assets.
rcr/         RCR code and the continual-learning framework used for experiments.
scripts/     Evaluation and artifact-generation entry points.
configs/     Experiment configurations and run metadata.
artifacts/   Reproducibility artifacts and cached evaluation summaries.
```

The paper and supplementary PDFs are mirrored in this repository.

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
