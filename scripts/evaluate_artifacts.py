#!/usr/bin/env python
"""Compute FAA, FF, and route accuracy from artifact_task*.pt feature exports.

Evaluation uses saved features and class statistics; no backbone is loaded.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rcr.rcr import rcr_scores_from_artifact


HEAD_KEYS = {
    "ce": "ce_scores",
    "etf": "etf_scores",
    "ldc": "ldc_scores",
    "radial": "ldcrad_scores",
    "saved_mix": "mixed_scores",
}


def artifact_task(path: Path) -> int:
    match = re.search(r"artifact_task(\d+)\.pt$", path.name)
    if not match:
        raise ValueError(f"Cannot infer task id from {path}")
    return int(match.group(1))


def score_metrics(scores: torch.Tensor, labels: torch.Tensor, cpt: int) -> dict[str, float]:
    pred = scores.argmax(dim=1)
    true_task = labels // cpt
    pred_task = pred // cpt
    routed = pred_task.eq(true_task)
    correct = pred.eq(labels)

    task_correct = torch.zeros_like(correct)
    for task in true_task.unique().tolist():
        task = int(task)
        mask = true_task.eq(task)
        start = task * cpt
        end = start + cpt
        local_pred = scores[mask, start:end].argmax(dim=1) + start
        task_correct[mask] = local_pred.eq(labels[mask])

    return {
        "n": float(labels.numel()),
        "class_il": correct.float().mean().item() * 100.0,
        "task_il": task_correct.float().mean().item() * 100.0,
        "route": routed.float().mean().item() * 100.0,
        "wrong_task": (~routed).float().mean().item() * 100.0,
        "same_task_error": (routed & ~correct).float().mean().item() * 100.0,
    }


def head_scores(artifact: dict, split: dict, head: str) -> torch.Tensor | None:
    if head == "rcr":
        return rcr_scores_from_artifact(artifact, split)
    key = HEAD_KEYS[head]
    scores = split.get(key)
    return scores.float() if scores is not None else None


def per_seen_task_metrics(scores: torch.Tensor, labels: torch.Tensor, cpt: int) -> dict[str, list[float]]:
    pred = scores.argmax(dim=1)
    true_task = labels // cpt
    pred_task = pred // cpt
    tasks = int(true_task.max().item()) + 1
    out = {"class": [], "task": [], "route": []}
    for task in range(tasks):
        mask = true_task.eq(task)
        if not torch.any(mask):
            continue
        start = task * cpt
        end = start + cpt
        local_pred = scores[mask, start:end].argmax(dim=1) + start
        out["class"].append(pred[mask].eq(labels[mask]).float().mean().item() * 100.0)
        out["task"].append(local_pred.eq(labels[mask]).float().mean().item() * 100.0)
        out["route"].append(pred_task[mask].eq(task).float().mean().item() * 100.0)
    return out


def finite_mean(values: list[float]) -> float:
    values = [v for v in values if torch.isfinite(torch.tensor(v))]
    return sum(values) / len(values) if values else float("nan")


def forgetting(history: list[list[float]]) -> float:
    if len(history) <= 1:
        return 0.0
    final = history[-1]
    vals = []
    for task in range(len(final) - 1):
        best = max(row[task] for row in history[:-1] if len(row) > task)
        vals.append(best - final[task])
    return finite_mean(vals)


def load_artifact(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)


def evaluate_artifact(path: Path, heads: list[str], split_name: str) -> list[dict[str, str]]:
    artifact = load_artifact(path)
    split = artifact.get(split_name, {})
    labels = split.get("labels")
    if labels is None:
        return []
    cpt = int(artifact["classes_per_task"])
    rows = []
    for head in heads:
        scores = head_scores(artifact, split, head)
        if scores is None:
            continue
        metrics = score_metrics(scores.float(), labels.long(), cpt)
        run_name = path.parent.parent.name if path.parent.name == "artifacts" else path.parent.name
        row = {
            "run": run_name,
            "artifact": str(path),
            "task": str(int(artifact["task"])),
            "split": split_name,
            "head": head,
        }
        row.update({k: f"{v:.6f}" for k, v in metrics.items()})
        rows.append(row)
    return rows


def run_artifacts(run_dir: Path) -> list[Path]:
    nested = run_dir / "artifacts"
    search_dir = nested if nested.is_dir() else run_dir
    return sorted(search_dir.glob("artifact_task*.pt"), key=artifact_task)


def summarize_run(run_dir: Path, heads: list[str], split_name: str) -> list[dict[str, str]]:
    artifacts = run_artifacts(run_dir)
    histories = {head: {"class": [], "task": [], "route": []} for head in heads}

    for path in artifacts:
        artifact = load_artifact(path)
        split = artifact.get(split_name, {})
        labels = split.get("labels")
        if labels is None:
            continue
        cpt = int(artifact["classes_per_task"])
        for head in heads:
            scores = head_scores(artifact, split, head)
            if scores is None:
                continue
            metrics = per_seen_task_metrics(scores.float(), labels.long(), cpt)
            for key in histories[head]:
                histories[head][key].append(metrics[key])

    rows = []
    for head in heads:
        if not histories[head]["class"]:
            continue
        final_class = histories[head]["class"][-1]
        final_task = histories[head]["task"][-1]
        final_route = histories[head]["route"][-1]
        rows.append(
            {
                "run": run_dir.name,
                "split": split_name,
                "head": head,
                "tasks": str(len(histories[head]["class"])),
                "class_faa": f"{finite_mean(final_class):.6f}",
                "class_ff": f"{forgetting(histories[head]['class']):.6f}",
                "task_faa": f"{finite_mean(final_task):.6f}",
                "task_ff": f"{forgetting(histories[head]['task']):.6f}",
                "route_final": f"{finite_mean(final_route):.6f}",
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", help="Run directories or artifact_task*.pt files.")
    parser.add_argument("--split", default="test", choices=["test", "buffer"])
    parser.add_argument("--heads", nargs="+", default=["etf", "rcr"], choices=sorted([*HEAD_KEYS, "rcr"]))
    parser.add_argument("--out-dir", default="artifacts/eval")
    args = parser.parse_args()

    paths = [Path(p) for p in args.runs]
    artifact_paths = [p for p in paths if p.is_file()]
    run_dirs = [p for p in paths if p.is_dir()]

    task_rows = []
    for path in artifact_paths:
        task_rows.extend(evaluate_artifact(path, args.heads, args.split))
    for run_dir in run_dirs:
        for path in run_artifacts(run_dir):
            task_rows.extend(evaluate_artifact(path, args.heads, args.split))

    summary_rows = []
    for run_dir in run_dirs:
        summary_rows.extend(summarize_run(run_dir, args.heads, args.split))

    out_dir = Path(args.out_dir)
    task_fields = [
        "run", "artifact", "task", "split", "head", "n",
        "class_il", "task_il", "route", "wrong_task", "same_task_error",
    ]
    summary_fields = [
        "run", "split", "head", "tasks",
        "class_faa", "class_ff", "task_faa", "task_ff", "route_final",
    ]
    write_csv(out_dir / "artifact_task_metrics.csv", task_rows, task_fields)
    write_csv(out_dir / "artifact_run_summary.csv", summary_rows, summary_fields)
    print(out_dir / "artifact_task_metrics.csv")
    print(out_dir / "artifact_run_summary.csv")


if __name__ == "__main__":
    main()
