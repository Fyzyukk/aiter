#!/usr/bin/env python3
"""Repeat the short-K integration review using Python's standard library only.

No device discovery, runtime imports, compilation, or kernel launch occurs.
Run offline_compile.py first to regenerate the codegen artifacts checked here.
Only review.json and REVIEW.md beneath --output-dir are written.
"""

import argparse
import ast
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import types


REFERENCE_COMMIT = "10ab50645d1f25e11844b814b66002b27181dfbf"
NEW_KIDS = {9040: 384, 9041: 768, 9042: 1024}
HISTORICAL_KIDS = {9000, 9010, 9011, 9012, 9020}
COMMON = "csrc/opus_gemm/opus_gemm_common.py"
GENERATOR = "csrc/opus_gemm/codegen/gen_instances_gfx950.py"
REPORT = "reports/opus_shortk_20260925"
HISTORICAL = "reports/opus_local_gap_current_20260925"
REFERENCE_INPUTS = {
    f"{HISTORICAL}/local_gap_details.csv":
        "34aa7d45f000175ae60f64dd24235207c686fccffccc92f988fb65d84101cd22",
    f"{HISTORICAL}/expected_all_candidates.csv":
        "2d424253c948b6aa325cc69ac7b99cfd3245f3525b59d1904ec7a76453049f36",
    f"{HISTORICAL}/deferred_shapes.csv":
        "ab2d31aba61bd6c4759efc45d9c88eba801c757e7b8d6bf325e7e7ee098b2ba7",
    f"{HISTORICAL}/tune_adapter.py":
        "7adecd5624ee24ea8852c8231e16a95d016237a73b6622678c656c8c3d24df8b",
}


class NoRuntimeImports:
    """Fail before any accidental runtime import can initialize a device."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"aiter", "torch", "triton", "cupy"}:
            raise RuntimeError(f"CPU-only review forbids importing {fullname}")
        return None


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def read_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def shape(row):
    return tuple(int(row[key]) for key in ("M", "N", "K"))


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args])


def load_registry(source, source_path, module_name):
    # Registry imports only stdlib modules and reads local CO metadata.
    # The import guard remains installed during this exec.
    module = types.ModuleType(module_name)
    module.__file__ = str(source_path)
    sys.modules[module_name] = module
    exec(compile(source, str(source_path), "exec"), module.__dict__)
    return module


def extract_candidates(path, registry, canonical_dtype, shape_helper=None):
    # Deliberately do not execute the adapter module and its runtime imports.
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "candidate_kids_for_shape")
    namespace = {
        "a8w8_mxscale_gemm_bpreshuffle_kernels_list": registry,
        "canonical_output_dtype": canonical_dtype,
        "a8w8_mxscale_bpreshuffle_supports_shape": shape_helper,
        "_TAG": "a8w8_mxscale_gemm_bpreshuffle",
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[function.name]


def review(root):
    failures = []
    checks = []

    def check(name, condition, detail=None):
        record = {"name": name, "passed": bool(condition)}
        if detail is not None:
            record["detail"] = detail
        checks.append(record)
        if not condition:
            failures.append(record)

    def file_hash(relative):
        return sha256((root / relative).read_bytes())

    baseline_path = root / REPORT / "baseline_identity.json"
    baseline = json.loads(baseline_path.read_text())
    current = load_registry((root / COMMON).read_text(), root / COMMON, "shortk_review_current")
    reference_source = git(root, "show", f"{REFERENCE_COMMIT}:{COMMON}").decode()
    reference = load_registry(reference_source, root / COMMON, "shortk_review_reference")
    registry = current.a8w8_mxscale_gemm_bpreshuffle_kernels_list
    old_ids = {int(kid) for kid in baseline["kernels"]}
    frozen_registry = {}
    old_identity_failures = []
    for kid_text, entry in baseline["kernels"].items():
        kid = int(kid_text)
        fields = dict(entry["fields"])
        # Derive the pre-edit m_align property entirely from frozen metadata.
        fields["m_align"] = 64 if fields["pad_m"] else (
            1 if fields["has_oob"] else fields["B_M"])
        frozen_registry[kid] = types.SimpleNamespace(**fields)
        instance = registry[kid]
        if instance.name != entry["name"]:
            old_identity_failures.append([kid, "name"])
        for key, expected in entry["fields"].items():
            actual = json.loads(json.dumps(getattr(instance, key)))
            if actual != expected:
                old_identity_failures.append([kid, key, expected, actual])
        if instance.fixed_k is not None:
            old_identity_failures.append([kid, "fixed_k must default to None"])
    check("all frozen old kernel names/metadata preserved", not old_identity_failures,
          old_identity_failures)
    check("registry adds exactly 9040/9041/9042", set(registry) == old_ids | NEW_KIDS.keys())
    new_names = [registry[kid].name for kid in NEW_KIDS]
    check("new names are unique and separate from old names",
          len(set(new_names)) == 3 and not set(new_names) &
          {entry["name"] for entry in baseline["kernels"].values()})

    source_hashes = {}
    for path, expected in baseline["files"].items():
        actual = file_hash(path)
        changed = actual != expected
        source_hashes[path] = {"before": expected, "current": actual, "changed": changed}
        if path not in {COMMON, GENERATOR}:
            check(f"protected source hash: {path}", not changed)

    reference_hashes = {}
    for path, expected in REFERENCE_INPUTS.items():
        actual = file_hash(path)
        reference_hashes[path] = {"expected": expected, "current": actual}
        check(f"historical input hash: {path}", actual == expected)
    provenance = json.loads((root / REPORT / "plan/source_manifest.json").read_text())
    # Check the first-25 frozen source snapshot through its existing provenance.
    entries = provenance["source_snapshots"]
    if isinstance(entries, dict):
        entries = list(entries.values())
    first25_provenance = [entry for entry in entries if isinstance(entry, dict)
                          and entry.get("snapshot") == "frozen_short_k_targets_25.csv"]
    check("first25 frozen provenance entry exists", len(first25_provenance) == 1)
    if first25_provenance:
        entry = first25_provenance[0]
        path = f"{REPORT}/plan/{entry['snapshot']}"
        check("first25 frozen CSV matches provenance SHA", file_hash(path) == entry["sha256"])

    old_candidates = extract_candidates(root / HISTORICAL / "tune_adapter.py",
                                        frozen_registry, reference.canonical_output_dtype)
    candidates = extract_candidates(root / REPORT / "tune_adapter.py", registry,
                                    current.canonical_output_dtype,
                                    current.a8w8_mxscale_bpreshuffle_supports_shape)
    details_rows = read_rows(root / HISTORICAL / "local_gap_details.csv")
    shapes295 = [shape(row) for row in details_rows]
    check("original supported shape set has 295 unique entries",
          len(shapes295) == len(set(shapes295)) == 295)
    historical_candidates = defaultdict(set)
    historical_count = 0
    for row in read_rows(root / HISTORICAL / "expected_all_candidates.csv"):
        if row["libtype"] == "opus":
            historical_candidates[shape(row)].add(int(row["kernelId"]))
            historical_count += 1
    check("historical candidate inventory contains 1275 OPUS records",
          historical_count == 1275)
    check("historical candidate inventory covers exactly the 295 supported shapes",
          set(historical_candidates) == set(shapes295))
    shape_records = []
    old_candidate_mismatches = []
    historical_mismatches = []
    for m, n, k in shapes295:
        before = old_candidates("gfx950", m, n, k)
        after = candidates("gfx950", m, n, k)
        retained = sorted(set(after) & old_ids)
        measured = sorted(historical_candidates[(m, n, k)])
        if retained != before:
            old_candidate_mismatches.append([m, n, k, before, retained])
        if sorted(set(retained) & HISTORICAL_KIDS) != measured:
            historical_mismatches.append([m, n, k, measured, retained])
        shape_records.append({"M": m, "N": n, "K": k, "old_candidates": before,
                              "retained_old_candidates": retained,
                              "historical_measured_candidates": measured,
                              "new_candidates": sorted(set(after) & NEW_KIDS.keys())})
    check("old 9-ID candidate sets unchanged on all 295 supported shapes",
          not old_candidate_mismatches, old_candidate_mismatches)
    check("historical 5-ID measured candidate inventory preserved on all 295 shapes",
          not historical_mismatches, historical_mismatches)

    frozen25 = read_rows(root / REPORT / "plan/frozen_short_k_targets_25.csv")
    planned25 = read_rows(root / REPORT / "plan/first25_candidates.csv")
    check("first25 frozen/planned shape sets agree and contain 25 unique shapes",
          len(frozen25) == len(planned25) == 25 and
          len({shape(row) for row in frozen25}) == 25 and
          {shape(row) for row in frozen25} == {shape(row) for row in planned25})
    check("all first25 targets belong to the original supported295 set",
          {shape(row) for row in frozen25} <= set(shapes295))
    first25_records = []
    for row in planned25:
        m, n, k = shape(row)
        expected = int(row["plan_expected_candidate"])
        accepted = sorted(set(candidates("gfx950", m, n, k)) & NEW_KIDS.keys())
        passed = accepted == [expected] and NEW_KIDS.get(expected) == k
        check(f"first25 unique new kid: M={m}, N={n}, K={k}", passed)
        first25_records.append({"M": m, "N": n, "K": k,
                                "expected_kid": expected, "accepted_new_kids": accepted,
                                "passed": passed})
    deferred = read_rows(root / HISTORICAL / "deferred_shapes.csv")
    check("10 pre-existing address-range exclusions stay excluded",
          len(deferred) == 10 and all(not candidates("gfx950", *shape(row)) for row in deferred))

    boundary_cases = 0
    boundary_failures = []
    limit = 2**31 - 1
    for kid, fixed_k in NEW_KIDS.items():
        instance = registry[kid]
        for m in (-64, 0, 1, 63, 64, 65, 128, 192, 256, 320, 384, 448, 512, 4096, 2**31):
            for n in (-256, 0, 1, 128, 255, 256, 257, 512, 4096, 2**31):
                for k in (-128, 0, 128, 256, 384, 512, 768, 1024, 1152):
                    expected = (min(m, n, k) > 0 and m % 64 == 0 and n % 256 == 0
                                and k == fixed_k and m <= limit // k and n <= limit // k
                                and m <= (limit // 2) // n)
                    actual = current.a8w8_mxscale_bpreshuffle_supports_shape(instance, m, n, k)
                    boundary_cases += 1
                    if bool(actual) != bool(expected):
                        boundary_failures.append([kid, m, n, k, expected, actual])
        for n in (256, 512, 4096, 65536):
            max_m = min(limit // fixed_k, (limit // 2) // n) // 64 * 64
            for m, expected in ((max_m, True), (max_m + 64, False)):
                boundary_cases += 1
                if current.a8w8_mxscale_bpreshuffle_supports_shape(instance, m, n, fixed_k) != expected:
                    boundary_failures.append([kid, m, n, fixed_k, expected])
        for m in (64, 128, 192, 4096):
            max_n = min(limit // fixed_k, (limit // 2) // m) // 256 * 256
            for n, expected in ((max_n, True), (max_n + 256, False)):
                boundary_cases += 1
                if current.a8w8_mxscale_bpreshuffle_supports_shape(instance, m, n, fixed_k) != expected:
                    boundary_failures.append([kid, m, n, fixed_k, expected])
        check(f"kid {kid} architecture/output rejection",
              not candidates("gfx942", 64, 256, fixed_k) and
              not candidates("gfx950", 64, 256, fixed_k, "fp32"))
    check(f"scalar contract matches launcher arithmetic in {boundary_cases} boundary cases",
          not boundary_failures, boundary_failures)

    defaults = {}
    for arch in ("gfx950", "gfx942", "gfx1250"):
        name = f"DEFAULT_COMPILED_KIDS_{arch.upper()}"
        before, after = sorted(getattr(reference, name)), sorted(getattr(current, name))
        defaults[arch] = {"reference": before, "current": after}
        check(f"default compile floor unchanged: {arch}", before == after)
    check("new IDs absent from all default compiled sets",
          not NEW_KIDS.keys() & current.DEFAULT_COMPILED_KIDS)
    config_hashes = {}
    for line in git(root, "ls-tree", "-r", REFERENCE_COMMIT, "--", "aiter/configs").decode().splitlines():
        metadata, path = line.split("\t", 1)
        object_id = metadata.split()[2]
        data = (root / path).read_bytes()
        git_blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        config_hashes[path] = {"reference_git_blob": object_id,
                               "current_git_blob": git_blob, "sha256": sha256(data)}
    check("all tracked production config bytes match fixed reference commit",
          all(item["reference_git_blob"] == item["current_git_blob"]
              for item in config_hashes.values()), {"files": len(config_hashes)})
    check("no untracked production config files added",
          not git(root, "ls-files", "--others", "--exclude-standard", "--", "aiter/configs").strip())

    aggregate_names = {"opus_gemm_manifest.h", "opus_gemm_a8w8_kid_dispatch.h",
                       "instances/all_instances_host_gfx950.cu"}
    generated_hashes = {}
    for path, expected in baseline["generated"].items():
        actual = file_hash(f"{REPORT}/codegen/{path}")
        generated_hashes[path] = {"before": expected, "current": actual,
                                  "changed": actual != expected}
        if path not in aggregate_names:
            check(f"old generated artifact byte identity: {path}", actual == expected)
        else:
            old_file = root / REPORT / "before_codegen" / path
            check(f"frozen aggregate provenance: {path}", sha256(old_file.read_bytes()) == expected)
            # Appending macro entries adds a continuation to the previous last
            # entry; normalize that spelling before comparing declarations.
            normalize = lambda text: [line.rstrip(" \\") for line in text.splitlines()]
            removed = Counter(normalize(old_file.read_text())) - Counter(normalize(
                (root / REPORT / "codegen" / path).read_text()))
            expected_removed = Counter()
            if path == "opus_gemm_a8w8_kid_dispatch.h":
                expected_removed["#define GENERATE_A8W8_BLOCKSCALE_BPRESHUFFLE_KID_DISPATCH_GFX950_BF16_SIZE 9"] = 1
            check(f"aggregate retains old declarations: {path}", removed == expected_removed,
                  dict(removed))

    dispatch = (root / REPORT / "codegen/opus_gemm_a8w8_kid_dispatch.h").read_text()
    dispatch_pairs = {int(kid): name for kid, name in re.findall(r"\{\s*(\d+),\s*&([^<\s]+)<bf16_t>", dispatch)}
    check("generated BF16 dispatch has exactly all 12 expected ID/name pairs",
          dispatch_pairs == {kid: instance.name for kid, instance in registry.items()})
    for kid, fixed_k in NEW_KIDS.items():
        instance = registry[kid]
        impl = root / REPORT / "codegen/impl" / f"{instance.name}.cuh"
        device = root / REPORT / "codegen/instances" / f"{instance.name}_Cbf16_t.device.cu"
        text = impl.read_text()
        snippets = [f"opus_gemm_mxscale_bpreshuffle_shortk_traits_gfx950<{fixed_k}>",
                    f'AITER_CHECK(k == {fixed_k}, entry, ": requires K == {fixed_k}");',
                    "m % 64 == 0 && n % 256 == 0 && k % 128 == 0",
                    "x_scale.stride(0) == 1 && x_scale.stride(1) == m",
                    "w_scale.is_contiguous()", "byte_limit = 2147483647",
                    "m <= byte_limit / k && n <= byte_limit / k &&",
                    "m <= (byte_limit / sizeof(D_C)) / n",
                    "gemm_a8w8_mxfp8_bpreshuffle_shortk_kernel<", "grid, dim3(512)"]
        check(f"new generated wrapper contract: kid {kid}", all(item in text for item in snippets)
              and text.index(f"AITER_CHECK(k == {fixed_k}") < text.index("args{}"))
        check(f"new generated device symbol: kid {kid}",
              "template __global__ void gemm_a8w8_mxfp8_bpreshuffle_shortk_kernel<" in device.read_text())

    input_hashes = {COMMON: file_hash(COMMON), GENERATOR: file_hash(GENERATOR),
                    f"{REPORT}/baseline_identity.json": sha256(baseline_path.read_bytes()),
                    f"{REPORT}/tune_adapter.py": file_hash(f"{REPORT}/tune_adapter.py")}
    for path in sorted((root / REPORT / "codegen").rglob("*")):
        if path.is_file():
            input_hashes[str(path.relative_to(root))] = sha256(path.read_bytes())
    for suffix in ("pipeline", "traits"):
        path = f"csrc/opus_gemm/include/gfx950/opus_gemm_{suffix}_a8w8_mxscale_bpreshuffle_shortk_gfx950.cuh"
        input_hashes[path] = file_hash(path)
    check("no forbidden runtime module imported",
          not any(name.split(".")[0] in {"aiter", "torch", "triton", "cupy"} for name in sys.modules))
    return {
        "status": "passed" if not failures else "failed", "cpu_only": True,
        "reference_commit": REFERENCE_COMMIT,
        "method": "Frozen old metadata + isolated AST adapter functions; generated artifacts inspected, not regenerated.",
        "limits": "No GPU numerical correctness, race validation on hardware, or performance claim.",
        "checks": checks, "failures": failures, "boundary_case_count": boundary_cases,
        "old_ids": sorted(old_ids), "new_ids": NEW_KIDS,
        "first25": first25_records, "supported295": shape_records,
        "historical_opus_candidate_records": historical_count,
        "default_compile_sets": defaults, "production_config_hashes": config_hashes,
        "protected_source_hashes": source_hashes, "old_generated_hashes": generated_hashes,
        "historical_input_hashes": reference_hashes, "current_input_hashes": input_hashes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    sys.meta_path.insert(0, NoRuntimeImports())
    result = review(args.root.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result["review_script_sha256"] = sha256(Path(__file__).read_bytes())
    (args.output_dir / "review.json").write_text(json.dumps(result, indent=2) + "\n")
    passed = sum(item["passed"] for item in result["checks"])
    count_by_kid = Counter(row["expected_kid"] for row in result["first25"])
    report = f"""# Short-K CPU-only integration review

Status: **{result['status']}** ({passed}/{len(result['checks'])} checks).

- All first 25 frozen target shapes admit exactly their planned new ID: {dict(count_by_kid)}.
- All 295 supported shapes retain their pre-edit 9-ID candidate sets. The historical 1,275 candidate records for IDs 9000/9010/9011/9012/9020 also match exactly.
- The 10 pre-existing address-range exclusions remain excluded.
- {result['boundary_case_count']} scalar cases cover exact K, M64/N256 alignment, invalid dimensions, and signed-int byte-limit boundaries. Unsupported architecture/output dtype is rejected.
- Frozen old names/metadata, protected source/helper hashes, and old per-kernel generated files remain unchanged. Aggregate generated files preserve existing declarations and add only the new family.
- Default compile sets match commit `{REFERENCE_COMMIT}`; all {len(result['production_config_hashes'])} tracked production configuration files match its bytes. New IDs are not defaults.
- New generated wrappers select the correct fixed-K traits/device symbol and enforce the exact-K/shape/scale/byte-range contract before constructing kernel arguments.

Re-run from the repository root after `offline_compile.py` has regenerated codegen artifacts:

```sh
python -B reports/opus_shortk_20260925/integration/review.py
```

The script uses only the standard library, reads old adapter functions through AST extraction, and blocks runtime imports. It does not discover a GPU, launch a kernel, or import `aiter`/`torch`. Full records and input hashes are in `review.json`. This is integration evidence; GPU numerical correctness and performance remain unmeasured.
"""
    if result["failures"]:
        report = (f"# Short-K CPU-only integration review\n\nStatus: **failed** "
                  f"({passed}/{len(result['checks'])} checks).\n\n"
                  "This rerun does not establish integration compatibility. "
                  "See review.json for input hashes and detailed results.\n\nFailures:\n\n" +
                  "\n".join(f"- {item['name']}" for item in result["failures"]) + "\n")
    (args.output_dir / "REVIEW.md").write_text(report)
    print(json.dumps({"status": result["status"], "checks_passed": passed,
                      "checks_total": len(result["checks"]), "output_dir": str(args.output_dir)}, indent=2))
    raise SystemExit(bool(result["failures"]))


if __name__ == "__main__":
    main()
