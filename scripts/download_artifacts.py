#!/usr/bin/env python3
"""Download selected RCR release archives, verify checksums, and extract safely."""

import argparse
import hashlib
import json
import tarfile
from pathlib import Path
from urllib.request import urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["cifar10", "cifar100", "tinyimagenet"], required=True)
    parser.add_argument("--buffer", type=int, choices=[200, 500], required=True)
    parser.add_argument("--tag", default="v1.0.0")
    parser.add_argument("--destination", type=Path, default=Path("."))
    parser.add_argument("--repo", default="towzeur/RCR")
    args = parser.parse_args()
    base = f"https://github.com/{args.repo}/releases/download/{args.tag}/"
    with urlopen(base + "manifest.json", timeout=60) as response:
        manifest = json.load(response)
    name = f"rcr_seq_{args.dataset}_buffer{args.buffer}.tar.gz"
    entry = next(a for a in manifest["assets"] if a["name"] == name)
    args.destination.mkdir(parents=True, exist_ok=True)
    path = args.destination / name
    partial = path.with_suffix(".partial")
    digest = hashlib.sha256()
    with urlopen(base + name, timeout=180) as response, partial.open("wb") as handle:
        for block in iter(lambda: response.read(8 * 1024 * 1024), b""):
            handle.write(block)
            digest.update(block)
    if digest.hexdigest() != entry["sha256"] or partial.stat().st_size != entry["bytes"]:
        raise RuntimeError("Download checksum/size mismatch; archive was not extracted")
    partial.replace(path)
    with tarfile.open(path) as archive:
        archive.extractall(args.destination, filter="data")
    print(f"Verified and extracted {name} to {args.destination.resolve()}")


if __name__ == "__main__":
    main()
