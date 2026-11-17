#!/usr/bin/env python3
"""Package feature artifacts and final inference checkpoints for release.

Source files are read only. Training checkpoints contain pickled Python
objects; only process files you trust.
"""

import argparse
import csv
import hashlib
import json
import re
import sys
import tarfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT / "rcr"))
    torch.set_num_threads(2)
    groups = {}
    with (ROOT / "configs/paper_runs.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        run = row["run"]
        group = run.rsplit("_seed", 1)[0]
        task_count = 5 if row["dataset"] == "seq-cifar10" else 10
        log = (args.historical_runs / row["source_run"] / "log.txt").read_text()
        prefixes = re.findall(r"Saving checkpoint into ([^\s]+)", log)
        if len(set(prefixes)) != 1:
            raise ValueError(f"Ambiguous checkpoint provenance: {run}")
        source = args.checkpoints / f"{prefixes[0]}_{task_count - 1}.pt"
        legacy = torch.load(source, map_location="cpu", weights_only=False)
        artifact_paths = sorted(
            (ROOT / "artifacts/paper_runs" / run).glob("artifact_task*.pt"),
            key=lambda p: int(re.search(r"task(\d+)", p.name)[1]),
        )
        if [p.name for p in artifact_paths] != [f"artifact_task{i}.pt" for i in range(1, task_count + 1)]:
            raise ValueError(f"Incomplete artifact trajectory: {run}")
        final = torch.load(artifact_paths[-1], map_location="cpu", weights_only=True)
        state = legacy["model"]
        for key in ("ldc_proto", "ldcrad_class_radius"):
            if not torch.equal(state[key], final[key]):
                raise ValueError(f"Checkpoint/artifact mismatch: {run}/{key}")
        etf = legacy["model_extras"]["etf"].cpu()
        saved_scores = final["test"]["etf_scores"]
        reconstructed = final["test"]["features"] @ etf[:, :final["class_num"]]
        if not torch.allclose(reconstructed, saved_scores, atol=2e-5, rtol=2e-5):
            raise ValueError(f"Checkpoint ETF does not reproduce saved scores: {run}")
        network = {key.removeprefix("net."): value.cpu() for key, value in state.items() if key.startswith("net.")}
        if not network:
            raise ValueError(f"Missing backbone: {run}")
        payload = {
            "format_version": 1,
            "run": run,
            "backbone": "resnet18",
            "num_filters": 64,
            "num_classes": final["class_num"],
            "classes_per_task": final["classes_per_task"],
            "task": task_count,
            "backbone_state_dict": network,
            "etf": etf,
            "ldc_proto": final["ldc_proto"],
            "ldcrad_class_radius": final["ldcrad_class_radius"],
            "training_config": {key: row[key] for key in ("dataset", "buffer", "seed", "epochs", "lr", "main_weight", "distill_weight", "lr_scheduler", "lr_milestones")},
        }
        target = out / "staging/checkpoints" / run / "final.pt"
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, target)
        torch.load(target, weights_only=True, map_location="cpu")
        files = [(target, f"checkpoints/{run}/final.pt")]
        files.extend((p, f"artifacts/paper_runs/{run}/{p.name}") for p in artifact_paths)
        groups.setdefault(group, []).append({"run": run, "source_checkpoint": source.name,
            "source_checkpoint_sha256": sha256(source), "files": files})
        print(f"Validated {run}: backbone + ETF + RCR state", flush=True)
    manifest = {"format_version": 1, "tag": args.tag, "assets": []}
    for group, runs in sorted(groups.items()):
        name = f"rcr_{group}.tar.gz"
        path = out / name
        entries = []
        with tarfile.open(path, "w:gz", compresslevel=1) as archive:
            for run in runs:
                for src, dest in run["files"]:
                    archive.add(src, arcname=dest, recursive=False)
                    entries.append({"path": dest, "bytes": src.stat().st_size, "sha256": sha256(src)})
        if path.stat().st_size >= 2 * 1024 ** 3:
            raise ValueError(f"GitHub asset exceeds 2 GiB: {name}")
        manifest["assets"].append({"name": name, "bytes": path.stat().st_size,
            "sha256": sha256(path), "runs": [{k: v for k, v in run.items() if k != "files"} for run in runs],
            "contents": entries})
        print(f"Packed {name}: {path.stat().st_size / 1024**2:.1f} MiB", flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    files = [out / entry["name"] for entry in manifest["assets"]] + [out / "manifest.json"]
    (out / "SHA256SUMS").write_text("".join(f"{sha256(p)}  {p.name}\n" for p in files))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-runs", type=lambda x: Path(x).resolve(), required=True)
    parser.add_argument("--checkpoints", type=lambda x: Path(x).resolve(), required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/release")
    parser.add_argument("--tag", default="v1.0.0")
    prepare(parser.parse_args())
