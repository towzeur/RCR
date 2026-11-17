#!/usr/bin/env python3
"""Create a GitHub release draft and upload checksummed reproducibility assets.

Uses GH_TOKEN/GITHUB_TOKEN or the configured Git credential helper. No token is
written to disk. Publication requires the explicit --publish flag.
"""

import argparse
import hashlib
import http.client
import json
import os
import subprocess
from pathlib import Path
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen


def token():
    value = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if value:
        return value
    result = subprocess.run(["git", "credential", "fill"],
        input="protocol=https\nhost=github.com\n\n", text=True,
        capture_output=True, check=True, timeout=30)
    fields = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    return fields["password"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--repo", default="towzeur/RCR")
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    directory = args.directory.resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    headers = {"Authorization": "Bearer " + token(), "Accept": "application/vnd.github+json",
               "User-Agent": "RCR-release", "X-GitHub-Api-Version": "2022-11-28"}

    def api(path, data=None, method=None):
        body = json.dumps(data).encode() if data is not None else None
        request = Request("https://api.github.com/repos/" + args.repo + path,
            data=body, headers={**headers, "Content-Type": "application/json"}, method=method)
        with urlopen(request, timeout=60) as response:
            return json.load(response)

    releases = api("/releases?per_page=100")
    release = next((r for r in releases if r["tag_name"] == manifest["tag"]), None)
    if release is None:
        release = api("/releases", {"tag_name": manifest["tag"], "name": "RCR: code and reproducibility artifacts",
            "draft": True, "body": "Checkpoints and per-task evaluation artifacts for the BMVC 2026 paper, across three datasets, buffers 200/500, and seeds 0/1/2. Each archive contains three final inference checkpoints and all task feature artifacts needed for FAA/FF. Training images, replay images, and optimizer state are excluded from the checkpoints. Verify downloads with SHA256SUMS. See manifest.json for per-file checksums and provenance."})
    if not release["draft"]:
        raise RuntimeError("Refusing to modify an already published release")
    remote = {a["name"]: a for a in release["assets"]}
    upload_base = release["upload_url"].split("{", 1)[0]
    assets = manifest["assets"] + [{"name": "manifest.json"}, {"name": "SHA256SUMS"}]
    for entry in assets:
        path = directory / entry["name"]
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        expected = "sha256:" + digest.hexdigest()
        if entry.get("sha256") and expected != "sha256:" + entry["sha256"]:
            raise ValueError(f"Local checksum mismatch: {path.name}")
        if path.name in remote:
            existing = remote[path.name]
            if existing["size"] != path.stat().st_size or existing.get("digest") != expected:
                raise ValueError(f"Remote asset differs; refusing to overwrite: {path.name}")
            print("Already verified:", path.name, flush=True)
            continue
        url = urlsplit(upload_base + "?name=" + quote(path.name))
        conn = http.client.HTTPSConnection(url.netloc, timeout=180)
        conn.putrequest("POST", url.path + "?" + url.query)
        for key, value in {**headers, "Content-Type": "application/octet-stream",
                           "Content-Length": str(path.stat().st_size)}.items():
            conn.putheader(key, value)
        conn.endheaders()
        print("Uploading:", path.name, flush=True)
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                conn.send(chunk)
        response = conn.getresponse()
        result = json.loads(response.read())
        conn.close()
        if response.status != 201:
            raise RuntimeError(f"Upload failed ({response.status}): {result.get('message')}")
        if result.get("digest") != expected or result["size"] != path.stat().st_size:
            raise RuntimeError(f"Server-side checksum verification failed: {path.name}")
        print("Verified on GitHub:", path.name, flush=True)
    if args.publish:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        release = api(f"/releases/{release['id']}", {"draft": False, "target_commitish": commit}, "PATCH")
    print("Published:" if args.publish else "Draft ready:", release["html_url"])


if __name__ == "__main__":
    main()
