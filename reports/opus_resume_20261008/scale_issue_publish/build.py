#!/usr/bin/env python3
"""Replay the recorded CPU HIP builds without changing their exact argv.

The checked-in build_manifest.json is an immutable input. Running this script
rebuilds the selected experiments.so outputs at the paths recorded there and
prints fresh status/hash records; it never rewrites the original manifest.
Use --dry-run to inspect argv without starting a compiler.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", choices=["baseline", "candidate"], action="append",
                        help="Replay only this side; repeat to select both.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    manifest_path = HERE / "build_manifest.json"
    manifest_before = sha(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "passed" and manifest["cpu_only"]
    selected = set(args.side or ["baseline", "candidate"])
    builds = [b for b in manifest["builds"] if b["side"] in selected]
    assert {b["side"] for b in builds} == selected
    env = dict(os.environ, ROCR_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="",
               CUDA_VISIBLE_DEVICES="")
    records = []
    for build in builds:
        argv = build["command"]
        assert isinstance(argv, list) and argv and all(isinstance(x, str) for x in argv)
        # Inspect paths, but pass the manifest's list unchanged to subprocess.
        assert argv.count("-o") == 1
        output = Path(argv[argv.index("-o") + 1])
        assert output == HERE / build["side"] / "experiments.so"
        assert str(HERE / build["side"] / "launch.hip") in argv
        if args.dry_run:
            records.append({"side": build["side"], "command": argv})
            continue
        started = time.monotonic()
        completed = subprocess.run(argv, cwd=WORKSPACE, env=env, check=False)
        record = {"side": build["side"], "command": argv,
                  "returncode": completed.returncode,
                  "seconds": time.monotonic() - started}
        if completed.returncode == 0:
            record["library_sha256"] = sha(output)
            record["matches_recorded_library_sha256"] = record["library_sha256"] == build["library_sha256"]
        records.append(record)
        assert sha(manifest_path) == manifest_before, "Original build manifest changed"
        if completed.returncode:
            print(json.dumps({"status": "failed", "cpu_only": True, "builds": records}, indent=2))
            raise SystemExit(completed.returncode)
    assert sha(manifest_path) == manifest_before
    matches = args.dry_run or all(r["matches_recorded_library_sha256"] for r in records)
    print(json.dumps({"status": "dry_run" if args.dry_run else "passed" if matches else "hash_mismatch",
                      "cpu_only": True, "build_manifest_unchanged": True,
                      "build_manifest_sha256": manifest_before, "builds": records}, indent=2))
    if not matches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
