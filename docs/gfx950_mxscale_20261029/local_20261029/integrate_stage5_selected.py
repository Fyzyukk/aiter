#!/usr/bin/env python3
"""Integrate only final-six-round-selected new COs with stable paired gains."""
import argparse
import copy
import dataclasses
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

from prepare_co_experiment import load_common, registry
from prepare_opus_repeat import csv_rows, key

LOCAL = Path(__file__).resolve().parent
ROOT = LOCAL.parents[2]
REL = Path("csrc/opus_gemm/gen_co/co_kernels_gfx950.json")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    out = campaign / "integration"
    out.mkdir()
    repo = campaign / "co_round3/.cache/experiment_repo"
    comparisons = csv_rows(campaign / "result/final160_comparisons.csv")
    selected = defaultdict(list)
    for row in comparisons:
        if (row["new_experimental_co"] == "True" and row["stable_candidate_improvement"] == "True"
                and float(row["paired_median_speedup_pct"]) >= .5):
            selected[int(row["winner_kid"])].append(list(key(row)[2:5]))
    receipt = {"selected_ids": sorted(selected), "shape_scopes": dict(selected),
               "selection": "Final fresh-six-round winner; correct all six; faster than original control in at least4/6 paired rounds; paired median improvement>=0.5%"}
    if not selected:
        (out / "receipt.json").write_text(json.dumps({**receipt, "applied": False, "reason": "No new CO meets the integration threshold"}, indent=2) + "\n")
        print(json.dumps(receipt))
        return
    # The experiment changed only the CO manifest, never pipeline source.
    for relative in ("csrc/opus_gemm", "csrc/include/opus"):
        for source in (ROOT / relative).rglob("*"):
            if source.is_file() and source.suffix in (".py", ".cuh", ".h", ".hpp", ".cc", ".cpp", ".hip", ".cu"):
                copied = repo / source.relative_to(ROOT)
                assert copied.is_file() and sha(source) == sha(copied), str(source)
    source = ROOT / REL
    original_bytes = source.read_bytes()
    original = load_common(ROOT, label="stage5_integration_original")
    experiment = load_common(repo, label="stage5_integration_experiment")
    before, experimental = registry(original), registry(experiment)
    assert len(before) == 48234
    assert all(dataclasses.asdict(instance) == dataclasses.asdict(experimental[kid]) for kid, instance in before.items())
    document = json.loads(original_bytes)
    extended = json.loads((repo / REL).read_text())
    entries = {entry["kid"]: entry for entry in extended["gfx950"]}
    images = []
    for kid, shapes in sorted(selected.items()):
        entry = copy.deepcopy(entries[kid])
        entry["supported_shapes"] = sorted(shapes)
        document["gfx950"].append(entry)
        document["image_suffixes"][str(kid)] = extended["image_suffixes"][str(kid)]
        name = experimental[kid].name + ".co"
        image = repo / REL.parent / "gfx950" / name
        assert image.is_file() and not image.is_symlink()
        data = image.read_bytes()
        assert data[:4] == b"\x7fELF" and int.from_bytes(data[18:20], "little") == 224
        images.append({"source": str(image), "destination": str(ROOT / REL.parent / "gfx950" / name), "sha256": sha(image)})
    document["_integrated_stage5"] = {**receipt, "source_campaign": str(campaign)}
    proposal = out / "proposed_manifest.json"
    proposal.write_text(json.dumps(document, indent=2) + "\n")
    integrated = load_common(ROOT, manifest=proposal, label="stage5_integration_proposed")
    after = registry(integrated)
    assert set(after) == set(before) | set(selected)
    assert all(dataclasses.asdict(instance) == dataclasses.asdict(after[kid]) for kid, instance in before.items())
    assert len({instance.name for instance in after.values()}) == len(after)
    assert integrated._bpreshuffle_jit_kid_base == 780
    for kid in selected:
        lhs, rhs = dataclasses.asdict(after[kid]), dataclasses.asdict(experimental[kid])
        lhs["bpreshuffle_co"].pop("supported_shapes", None)
        rhs["bpreshuffle_co"].pop("supported_shapes", None)
        assert lhs == rhs
    backup = out / "public_manifest_before.json"
    backup.write_bytes(original_bytes)
    receipt.update(previous_ids_verified=48234, new_ids_count=len(selected), images=images,
                   manifest_before_sha256=sha(source), manifest_after_sha256=sha(proposal), applied=False)
    created = []
    try:
        assert source.read_bytes() == original_bytes
        for image in images:
            destination = Path(image["destination"])
            assert not destination.exists()
            shutil.copy2(image["source"], destination)
            created.append(destination)
            assert sha(destination) == image["sha256"]
        source.write_bytes(proposal.read_bytes())
        applied = load_common(ROOT, label="stage5_integration_applied")
        assert all(dataclasses.asdict(instance) == dataclasses.asdict(registry(applied)[kid]) for kid, instance in after.items())
        receipt["applied"] = True
    except BaseException:
        source.write_bytes(original_bytes)
        for destination in created:
            destination.unlink()
        raise
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"integrated_co_ids": sorted(selected), "previous48234_preserved": True}))


if __name__ == "__main__":
    main()
