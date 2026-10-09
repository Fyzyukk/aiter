#!/usr/bin/env python3
"""Incremental CPU-only landing audit; reuses the existing exact 745 inventory.

No GPU imports, builds, timing, production edits, or repeated ISA extraction.
Outputs scope and missing boundary checks for two existing experiments.
"""
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
SOURCE = ROOT / "csrc/opus_gemm/include/gfx950"


def read(rel):
    return json.loads((OUT / rel).read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parameters(value):
    args = value.split("<", 1)[1].rsplit(">", 1)[0]
    return [x.strip() for x in args.split(",")]


def compact(row):
    keys = ["M", "N", "K", "parent_id", "actual_configuration_id",
            "actual_branch_index", "actual_branch_label", "actual_conditions",
            "canonical_traits_type", "BM", "BN", "waves", "wave_k", "split_k",
            "stages", "cluster", "register_prefetch", "fixed_k",
            "producer_workgroups", "k128_tiles_per_global_split",
            "official_instruction_sha256"]
    return {key: row[key] for key in keys}


def counted(rows, key):
    return dict(sorted(Counter(row[key] for row in rows).items()))


def main():
    inventory = read("shape_inventory.json")
    probe = read("shape_inventory_evidence/traits_probe.json")["traits"]
    official = read("shape_inventory_evidence/official_resources.json")
    register_audit = read("register_reuse/device_audit.json")
    fine_audits = {side: read(f"fine_wait/{side}/device_audit.json")
                   for side in ["baseline", "candidate"]}
    fine_resources = {side: {row["name"]: row for row in
                     read(f"fine_wait/{side}/resource_isa.json")}
                      for side in ["baseline", "candidate"]}
    assert inventory["source_head"] == "b152ab834e68f8fde18adbd7b200c587770b9fb5"
    assert inventory["summary"]["rows"] == 745
    assert all(row["metadata_match"] == "exact_official"
               for row in inventory["rows"])
    assert not inventory["gpu_imported_or_initialized"]
    assert "torch" not in sys.modules

    # Reuse the already reviewed generated scalar branch interpreter.
    spec = importlib.util.spec_from_file_location("landing_inventory", OUT / "shape_inventory.py")
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    sys.path.insert(0, str(ROOT / "csrc/opus_gemm"))
    import opus_gemm_common as registry
    from codegen import gen_instances_gfx950 as codegen
    assert "torch" not in sys.modules

    register_ids = {9040, 9041, 9042, 9050, 9051, 9052, 9053, 9054}
    trait_header = SOURCE / "opus_gemm_traits_a8w8_mxscale_bpreshuffle_small_gfx950.cuh"
    fine_header = SOURCE / "opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_small_lds_gfx950.cuh"
    baseline_trait = OUT / "register_reuse/baseline" / trait_header.name
    candidate_trait = OUT / "register_reuse/candidate" / trait_header.name
    baseline_fine = OUT / "fine_wait/baseline/include/gfx950" / fine_header.name
    candidate_fine = OUT / "fine_wait/candidate/include/gfx950" / fine_header.name
    assert trait_header.read_bytes() == baseline_trait.read_bytes()
    assert fine_header.read_bytes() == baseline_fine.read_bytes()
    old = "bool ReuseBScale = false>"
    new = "bool ReuseBScale = (128 % BlockN == 0)>"
    assert baseline_trait.read_text().count(old) == 1
    assert baseline_trait.read_text().replace(old, new) == candidate_trait.read_text()
    alias_matches = re.findall(
        r"using\s+(\w+)\s*=\s*opus_gemm_small_register_traits_gfx950<([^;]+)>;",
        trait_header.read_text())
    assert len(alias_matches) == len(register_ids)
    assert all(len(args.split(",")) <= 8 for _, args in alias_matches)

    register_libraries = {lib["mode"]: {row["variant"]: row for row in lib["kernels"]}
                          for lib in register_audit["libraries"]}
    assert set(register_libraries["baseline"]) == register_ids
    register_rows = []
    register_effective = []
    register_noop = []
    register_explicit = []
    for row in inventory["rows"]:
        if not row["canonical_traits_type"].startswith("opus_gemm_small_register_traits_gfx950<"):
            continue
        args = parameters(row["canonical_traits_type"])
        assert len(args) == 11
        if row["actual_configuration_id"] not in register_ids:
            assert row["actual_configuration_id"] in {9070, 9071, 9072, 9073}
            assert args[-1] == ("true" if row["BN"] == 32 else "false")
            register_explicit.append(compact(row))
            continue
        assert args[-1] == "false" and row["fixed_k"] == 0
        assert 128 % row["BN"] == 0
        b = register_libraries["baseline"][row["actual_configuration_id"]]
        c = register_libraries["candidate"][row["actual_configuration_id"]]
        assert b["name"] == row["official_kernel_symbol"]
        assert b["instruction_sha256"] == row["official_instruction_sha256"]
        entry = compact(row)
        entry["baseline_reuse_b_scale"] = False
        entry["candidate_reuse_b_scale"] = True
        en = int(args[1]) // (int(args[3]) * 16)
        entry["E_N"] = en
        entry["bytecode_changed"] = b["instruction_sha256"] != c["instruction_sha256"]
        entry["static_b8_loads"] = [b["b8_vmem_count"], c["b8_vmem_count"]]
        entry["candidate_instruction_sha256"] = c["instruction_sha256"]
        register_rows.append(entry)
        if en == 1:
            assert not entry["bytecode_changed"]
            assert b["resources"] == c["resources"]
            register_noop.append(entry)
        else:
            assert entry["bytecode_changed"]
            assert c["b8_vmem_count"] < b["b8_vmem_count"]
            register_effective.append(entry)
    assert (len(register_rows), len(register_effective), len(register_noop),
            len(register_explicit)) == (58, 41, 17, 38)

    # Enumerate the actual B scale address of every lane's N fragment. This
    # checks all supported N tile origins, including N128 group boundaries.
    address_checks = 0
    for bn in [16, 32, 64]:
        for tm, tn in [(1, 1)]:
            for n_total in [128, 256, 384, 768, 7168, 65536]:
                for col in range(0, n_total, bn):
                    for wn in range(tn):
                        for ni in range(bn // (tn * 16)):
                            nr = (ni * tn + wn) * 16
                            assert (col + nr) // 128 == col // 128
                            address_checks += 1

    fine_baseline = {row["name"]: row for row in fine_audits["baseline"]["kernels"]}
    fine_candidate = {row["name"]: row for row in fine_audits["candidate"]["kernels"]}
    fine_rows = []
    fine_unchanged = []
    for row in inventory["rows"]:
        if not row["canonical_traits_type"].startswith("opus_gemm_mxscale_bpreshuffle_fine_traits_gfx950<"):
            continue
        traits = probe[row["actual_traits"]]
        partial = (traits["B_M"] // 8) % traits["NUM_WAVES"]
        gated = bool(traits["FINE_M_LOADS"] and not traits["REGISTER_SCALES"]
                     and traits["EARLY_SCALE_LOADS"] and partial)
        b = fine_baseline[row["official_kernel_symbol"]]
        c = fine_candidate[b["name"]]
        assert b["instruction_sha256"] == row["official_instruction_sha256"]
        assert gated == (b["instruction_sha256"] != c["instruction_sha256"])
        entry = compact(row)
        entry["gated_change"] = gated
        entry["partial_a_copy_waves"] = partial
        awave = [traits["B_M"] // 8 // traits["NUM_WAVES"] + (wave < partial)
                 for wave in range(traits["NUM_WAVES"])]
        bwave = traits["B_N"] * 128 // (traits["BLOCK_SIZE"] * 16)
        entry["vmem_matrix_requests_per_wave_per_tile"] = [a + bwave for a in awave]
        # All 745 fine winners are beyond the short-K branch, so the changed
        # nonzero steady wait can execute; this is not measured stall time.
        entry["steady_wait_reachable"] = any(n > traits["NUM_STAGES"]
                                              for n in row["k128_tiles_per_global_split"])
        entry["candidate_instruction_sha256"] = c["instruction_sha256"]
        entry["sgpr"] = [fine_resources["baseline"][b["name"]]["sgpr"],
                         fine_resources["candidate"][b["name"]]["sgpr"]]
        (fine_rows if gated else fine_unchanged).append(entry)
    assert (len(fine_rows), len(fine_unchanged)) == (53, 2)
    assert all(row["steady_wait_reachable"] for row in fine_rows)
    assert all(row["sgpr"][1] == row["sgpr"][0] + 2 for row in fine_rows)

    # Existing official object matches include emitted leaves not chosen by
    # 745 winners: M80 runtime split2 and M48 fixed S6/C2, plus reducers.
    full_fine_matches = []
    for name, b in fine_baseline.items():
        matches = [{"parent_id": int(parent), "instruction_equal":
                    item["instruction_sha256"] == b["instruction_sha256"]}
                   for parent, data in official.items() for item in data["variants"]
                   if item["name"] == name]
        assert matches and all(match["instruction_equal"] for match in matches)
        full_fine_matches.append({"name": name, "matches": matches,
                                  "candidate_instruction_equal":
                                  b["instruction_sha256"] == fine_candidate[name]["instruction_sha256"]})
    assert len(full_fine_matches) == 20

    # Also retain emitted branch scope, including current745-unselected leaves.
    # This distinguishes parent9063's unreachable K7168 legacy specialization
    # from a valid explicit9069 call; both compile the same producer type.
    branch_scope = {"register_reuse": [], "fine_wait": []}
    for parent, data in inventory["parents"].items():
        for leaf in data["branches"]:
            traits = probe[leaf["traits_cpp"]]
            canonical = traits["canonical_type"]
            if canonical.startswith("opus_gemm_small_register_traits_gfx950<"):
                scope = "register_reuse"
                defaulted = leaf["traits_cpp"].startswith("opus_gemm_mxscale_bpreshuffle_small_register")
                params = parameters(canonical)
                effective = defaulted and int(params[1]) // (int(params[3]) * 16) > 1
            elif canonical.startswith("opus_gemm_mxscale_bpreshuffle_fine_traits_gfx950<"):
                scope = "fine_wait"
                defaulted = False
                effective = bool(traits["FINE_M_LOADS"] and not traits["REGISTER_SCALES"]
                    and traits["EARLY_SCALE_LOADS"] and (traits["B_M"] // 8) % traits["NUM_WAVES"])
            else:
                continue
            selected = [row for row in inventory["rows"]
                        if row["parent_id"] == int(parent) and row["actual_branch_index"] == leaf["index"]]
            unreachable = any("k >= 8192" in cond for cond in leaf["conditions"]) and "k == 7168" in leaf["conditions"]
            branch_scope[scope].append({"parent_id": int(parent), "branch_index": leaf["index"],
                "branch_label": leaf["branch_label"], "conditions": leaf["conditions"],
                "canonical_traits_type": canonical, "defaulted_reuse_parameter": defaulted,
                "effective_bytecode_change": effective, "current745_selected_rows": len(selected),
                "parent_path_unreachable_K_contradiction": unreachable})

    alias_to_id = {codegen._bpreshuffle_compact_traits(value): kid
                   for kid, value in registry.a8w8_mxscale_gemm_bpreshuffle_kernels_list.items()}
    canonical_to_traits = {value["canonical_type"]: value for value in probe.values()}
    aliases_to_canonical = {key: value["canonical_type"] for key, value in probe.items()}

    def branch_for(kid, shape):
        instance = registry.a8w8_mxscale_gemm_bpreshuffle_kernels_list[kid]
        assert registry.a8w8_mxscale_bpreshuffle_supports_shape(instance, *shape)
        if str(kid) in inventory["parents"]:
            leaves = inventory["parents"][str(kid)]["branches"]
            selected = [leaf for leaf in leaves if all(
                helpers.cpp_eval(condition, *shape) for condition in leaf["conditions"])]
            assert len(selected) == 1
            leaf = selected[0]
            actual_id = alias_to_id.get(leaf["traits_cpp"], kid)
            canonical = aliases_to_canonical[leaf["traits_cpp"]]
        else:
            # Explicit historical call: use that ID's specialization directly.
            expression = codegen._bpreshuffle_compact_traits(instance)
            for specialization in instance.bpreshuffle_specializations:
                if specialization[0] == shape[2]:
                    expression = codegen._bpreshuffle_compact_traits(instance, specialization)
                    break
            actual_id, canonical = kid, aliases_to_canonical[expression]
        return actual_id, canonical

    def case(kid, shape, expected, reasons, *, kind="production_dispatch"):
        actual, canonical = branch_for(kid, shape)
        assert actual == expected, (kid, shape, actual, expected)
        value = {"kid": kid, "shape": list(shape), "expected_actual_id": actual,
                 "canonical_traits_type": canonical, "reasons": reasons, "call_kind": kind}
        if canonical.startswith("opus_gemm_mxscale_bpreshuffle_fine_traits_gfx950<"):
            traits = canonical_to_traits[canonical]
            total = shape[2] // 128
            per, extra = divmod(total, traits["SPLIT_K"])
            value["split_loops"] = [per + (s < extra) for s in range(traits["SPLIT_K"])]
            value["gated_wait_change"] = bool((traits["B_M"] // 8) % traits["NUM_WAVES"])
        return value

    register_regressions = [
        case(9041, (64, 896, 4096), 9050, ["grid224 below256 -> deep prefetch"]),
        case(9041, (64, 1024, 4096), 9041, ["grid256 boundary -> base prefetch"]),
        case(9041, (64, 1024, 8192), 9050, ["K>=8192 overrides grid"]),
        case(9041, (17, 256, 128), 9041, ["base N16 output0", "empty K-wave and M tail"]),
        case(9052, (17, 384, 7168), 9070, ["explicit reuse already true", "fixed output4 M tail"]),
        case(9053, (33, 384, 7168), 9072, ["explicit reuse already true", "fixed output4 M tail"]),
        case(9042, (256, 768, 7168), 9071, ["M16N48 grid256 inclusive boundary", "N-tail scale groups"]),
        case(9042, (257, 768, 7168), 9073, ["M16N48 grid272 -> M32N48", "M tail, explicit false"]),
    ]
    # Existing register_validation has 42 cases spanning 6 effective aliases.
    # Its N384 tests multiple N128 groups and randomized native scale values.
    # Keep them if their binary/runner evidence matches; only the production
    # dispatch cases above are additional after formal integration.
    existing_register = read("plans/register_validation.json")
    assert {target["kid"] for target in existing_register["targets"]} == register_ids - {9041, 9050}
    register_queue_checks = []
    for kid in sorted(register_ids - {9041, 9050}):
        b = register_libraries["baseline"][kid]
        bm, bn, tm, tn, prefetch, wave_k = b["template_ints"][:6]
        for total in [prefetch * wave_k, prefetch * wave_k + 1]:
            m = 17 if bm == 16 else 33
            shape = (m, 384, total * 128)
            # Reuse a plan case with this same K when already available;
            # every existing case uses N384 and several M-tail values.
            existing = next((target for target in existing_register["targets"]
                             if target["kid"] == kid and target["shape"][2] == shape[2]), None)
            if existing:
                shape = tuple(existing["shape"])
            entry = case(kid, shape, kid, [
                f"waveK={wave_k}, prefetch={prefetch}: {'exact full queue' if total == prefetch * wave_k else 'first post-queue tail, uneven K-wave partition'}",
                "N384 covers adjacent N128 scale groups"])
            entry["already_in_existing_private_plan"] = existing is not None
            entry["per_wave_loops"] = [total // wave_k + (w < total % wave_k) for w in range(wave_k)]
            register_queue_checks.append(entry)
    existing_fine = read("plans/fine_validation.json")
    fine_boundaries = []
    for kid, m, split in [(9060, 81, 1), (9061, 97, 1), (9062, 145, 2), (9063, 81, 4), (9068, 113, 4)]:
        for local in [4, 5]:
            total = local * split
            fine_boundaries.append(case(kid, (m, 256, total * 128), kid,
                [f"each split loops={local}: {'S' if local == 4 else 'S+1'} boundary", "M tail SFA byte fallback"]))
    fine_boundaries.extend([
        case(9062, (145, 256, 1152), 9062, ["9 K128 tiles ->5/4 split loops: ring/short coexist"]),
        case(9063, (81, 256, 1664), 9063, ["13 K128 tiles ->4/3/3/3: short split boundary"]),
        case(9063, (81, 256, 2176), 9063, ["17 K128 tiles ->5/4/4/4: first ring split"]),
        case(9062, (145, 256, 128), 9062, ["empty split must overwrite finite FP32 partials"]),
        case(9063, (81, 256, 128), 9063, ["3 empty splits must overwrite finite FP32 partials"]),
        case(9060, (81, 256, 3072), 9060, ["fixedK3072 specialization and M tail"]),
        case(9060, (81, 256, 7168), 9060, ["fixedK7168 specialization and M tail"]),
        case(9062, (145, 256, 16384), 9062, ["M80 fixedK16384 split2 and reducer"]),
        case(9063, (79, 256, 16384), 9063, ["M80 fixedK16384 split4 uses different reducer vec/cache", "M tail SFA byte fallback"]),
        case(9062, (80, 256, 1152), 9062, ["M80/M96 dispatch equality side"]),
        case(9062, (81, 256, 1152), 9064, ["M96 reduces M tiles", "split2 reducer16/cache2"]),
        case(9063, (48, 256, 8192), 9069, ["M<=48 dispatch to runtime M48"]),
        case(9063, (49, 256, 8192), 9063, ["first M80 at M49"]),
        case(9063, (81, 256, 8192), 9066, ["M96/4wave unaffected branch"]),
        case(9063, (97, 256, 8192), 9068, ["M112 first row and SFA byte fallback"]),
        case(9063, (113, 256, 8192), 9067, ["M128 unaffected branch"]),
        case(9063, (129, 4096, 8192), 9063, ["M80split4 grid256 equality side"]),
        case(9063, (129, 4224, 8192), 9065, ["grid264 >256 dispatch M96/8wave"]),
        case(9069, (49, 256, 7168), 9069, ["explicit legacy fixed S6/C2", "not reachable through parent K>=8192"], kind="explicit_legacy"),
        case(9069, (49, 256, 3072), 9069, ["explicit legacy runtime K, no clustered specialization"], kind="explicit_legacy"),
    ])

    forced = read("fine_wait/variants.json")

    def private_fine_traits(kid, shape):
        if str(kid) not in forced["forced_runtime_ids"]:
            return branch_for(kid, shape)[1]
        named = forced["forced_runtime_ids"][str(kid)]
        expression = "opus_gemm_mxscale_bpreshuffle_fine_traits_gfx950<" + ", ".join(
            str(value) for value in forced["traits"][named]) + ">"
        return aliases_to_canonical[expression]

    def planned(row, plan):
        matches = [target["kid"] for target in plan["targets"]
                   if target["shape"] == row["shape"] and
                   private_fine_traits(target["kid"], target["shape"]) == row["canonical_traits_type"]]
        return matches

    for row in fine_boundaries:
        matches = planned(row, existing_fine)
        row["equivalent_existing_private_plan_ids"] = matches
        row["already_in_existing_private_plan"] = bool(matches)

    landing_plans = []
    register_targets = []
    seen = set()
    for row in register_queue_checks:
        if row["already_in_existing_private_plan"]:
            continue
        key = (row["kid"], tuple(row["shape"]))
        if key not in seen:
            seen.add(key)
            register_targets.append({"kid": row["kid"], "shape": row["shape"],
                                     "seed": 101 + len(register_targets), "signed": True,
                                     "reasons": row["reasons"]})
    plan_path = OUT / "plans/register_landing_incremental.json"
    plan_path.write_text(json.dumps({"libraries": existing_register["libraries"], "workspace": False,
        "targets": register_targets}, indent=2) + "\n")
    landing_plans.append(str(plan_path.relative_to(OUT)))
    fine_targets = []
    seen = set()
    for row in fine_boundaries:
        if row["already_in_existing_private_plan"]:
            continue
        key = (row["kid"], tuple(row["shape"]))
        if key not in seen:
            seen.add(key)
            fine_targets.append({"kid": row["kid"], "shape": row["shape"],
                                 "seed": 201 + len(fine_targets), "signed": True,
                                 "reasons": row["reasons"]})
    plan_path = OUT / "plans/fine_landing_incremental.json"
    plan_path.write_text(json.dumps({"libraries": existing_fine["libraries"], "workspace": True,
        "targets": fine_targets}, indent=2) + "\n")
    landing_plans.append(str(plan_path.relative_to(OUT)))
    summaries = {
        "register_reuse": {"template_flag_changed_rows": len(register_rows),
            "effective_bytecode_changed_rows": len(register_effective),
            "N16_same_bytecode_rows": len(register_noop),
            "explicit_fixed_unchanged_rows": len(register_explicit),
            "effective_by_actual_id": counted(register_effective, "actual_configuration_id"),
            "template_flag_by_parent": counted(register_rows, "parent_id"),
            "all_relevant_current745_baseline_exact_official": True},
        "fine_wait": {"gated_bytecode_changed_rows": len(fine_rows),
            "gated_traits_selected_by_current745": len({row["canonical_traits_type"] for row in fine_rows}),
            "unchanged_fine_rows": len(fine_unchanged),
            "changed_by_actual_id": counted(fine_rows, "actual_configuration_id"),
            "changed_by_parent": counted(fine_rows, "parent_id"),
            "all_selected_steady_wait_reachable": True,
            "all20_private_baseline_entry_instruction_hashes_exact_official": True}}
    result = {"status": "cpu_landing_scope_passed_gpu_acceptance_pending", "gpu_executed": False,
        "production_modified": False, "source_head": inventory["source_head"],
        "summary": summaries,
        "prepared_incremental_private_plans": landing_plans,
        "reused_artifacts_sha256": {rel: sha(OUT / rel) for rel in [
            "shape_inventory.json", "shape_inventory_evidence/traits_probe.json",
            "shape_inventory_evidence/official_resources.json", "register_reuse/device_audit.json",
            "fine_wait/baseline/device_audit.json", "fine_wait/candidate/device_audit.json",
            "fine_wait/static_validation.json", "plans/register_validation.json", "plans/fine_validation.json"]},
        "register_reuse": {
            "minimal_change": {"file": str(trait_header), "old": old, "new": new,
                "one_line_only": True, "unchanged_codegen_and_dispatch": True},
            "source_matches_existing_frozen_baseline": True,
            "defaulted_aliases": [name for name, _ in alias_matches],
            "emitted_parent_branch_scope": branch_scope["register_reuse"],
            "scale_group_address_checks_passed": address_checks,
            "scale_group_condition": "128%BN==0 and col=tile_x*BN keep every N16 fragment within the same N128 scale group; T_N=1 for current aliases",
            "template_flag_rows": register_rows, "effective_rows": register_effective,
            "N16_same_bytecode_rows": register_noop, "explicit_fixed_unchanged_rows": register_explicit,
            "mandatory_regression": {"reuse_existing_private_plan": "plans/register_validation.json",
                "existing_plan_targets": len(existing_register["targets"]),
                "incremental_private_plan": "plans/register_landing_incremental.json",
                "production_dispatch_list_is_not_runnable_with_direct_private_launcher": True,
                "production_dispatch_checks": register_regressions,
                "queue_boundary_checks": register_queue_checks,
                "new_queue_shapes_excluding_existing_plan": sum(not row["already_in_existing_private_plan"] for row in register_queue_checks),
                "required_data": "signed native FP8, randomized native E8M0 per N128/K128 group; N384 covers groups0,1,2; repeated runs and output guards",
                "not_required": "No non-N128 or non-K128 input is newly supported. N16 code-identical branches need smoke/symbol verification, not a large new performance campaign."},
            "acceptance_gate": [
                "Clean shared-pool Event timing for the six affected runtime aliases; private launch bypasses fixed-K production dispatch.",
                "If any alias regresses, retain its false value explicitly or use per-alias true opt-in after exact candidate recompile; do not assume one default benefits every alias.",
                "Rebuild official host and device together because the defaulted true flag changes mangled template symbols, including N16 code-identical entries.",
                "Compare adopted official candidate instructions/resources to the tested private candidate and keep fixed9070-9073 bytecode unchanged.",
                "A changed default is an optimization only after correctness and complete-call timing pass; CPU load/VGPR reductions are not adoption evidence."]},
        "fine_wait": {
            "minimal_change": {"file": str(fine_header), "frozen_candidate": str(candidate_fine),
                "source_sha256": sha(fine_header), "candidate_sha256": sha(candidate_fine),
                "scope": "one helper lambda plus three existing steady-state wait call replacements",
                "guard": "FINE_M_LOADS && !REGISTER_SCALES && EARLY_SCALE_LOADS && ((B_M/8)%NUM_WAVES!=0)",
                "unchanged": "traits, dispatch, LDS allocation, matrix/scales loads, zero VM waits, LGKM waits, barriers, ring protection, drain, output/reducers"},
            "source_matches_existing_frozen_baseline": True,
            "emitted_parent_branch_scope": branch_scope["fine_wait"],
            "changed_rows": fine_rows, "unchanged_fine_rows": fine_unchanged,
            "full_entry_official_matches": full_fine_matches,
            "current745_coverage_limit": "All 53 changed winners execute long-K S4/C1 paths and M divisible by16; current745 alone omits short/empty/uneven split, M-byte fallback, and explicit M48 S6/C2.",
            "mandatory_regression": {"reuse_existing_private_plan": "plans/fine_validation.json",
                "existing_plan_targets": len(existing_fine["targets"]),
                "incremental_private_plan": "plans/fine_landing_incremental.json",
                "boundary_and_dispatch_checks": fine_boundaries,
                "new_unique_boundary_shapes_excluding_existing_plan": sum(not row["already_in_existing_private_plan"] for row in fine_boundaries),
                "unchanged_common_pipeline_controls": [9043, 9044, 9045, 9046, 9047, 9049, 9055],
                "control_requirement": "Official code-object hash/resource comparison for non-Fine, REGISTER_SCALES, early=false, C>1, M96W4/M128 and all reducer entries; add only a smoke if identical.",
                "required_data": "signed native FP8, variable finite E8M0, full shared-pool output/workspace finite/guard/repeatability checks; split2/4 include complete FP32 producer+reducer"},
            "acceptance_gate": [
                "Clean same-pool Event confirmation by affected actual geometry and fixed/runtime branch; all timing includes producer and reducer.",
                "The existing source-level schedule proof and LLVM same-class ordering argument are necessary background, not a replacement for GPU boundary checks.",
                "Verify adopted official compiler output retains VMEM load order and the exact intended per-wave thresholds; no added spills or changed barriers/drain.",
                "If only selected geometries win, add a compile-time allowlist/trait guard, rebuild and retest that scoped candidate instead of applying the whole frozen mutation unconditionally.",
                "Counter profiling is separate; absence of counters limits root-cause claims without invalidating a clean completed timing comparison."]},
        "limits": ["Current745 inventory contains historical winning parents, not all alternate candidates for every shape.",
                   "No GPU result or production adoption is established by this CPU artifact.",
                   "Boundary list supplements existing plans; do not rerun completed valid checks merely because they appear here."]}
    timing_summary = OUT / "register_timing_summary.json"
    if timing_summary.exists():
        observed = json.loads(timing_summary.read_text())
        result["register_reuse"]["subsequent_gpu_evidence"] = {
            "path": str(timing_summary.relative_to(OUT)), "sha256": sha(timing_summary),
            "global_default_change_allowed": False,
            "rejected_aliases": [9040],
            "positive_signal_aliases_pending_second_window_and_boundaries": [9042, 9053, 9054],
            "no_resolved_first_window_gain_aliases": [9051, 9052],
            "adoption_requires_new_scoped_mutation": "Keep9040 false; original one-line default describes the rejected full experiment, not a viable production landing. Opt in only independently verified aliases/traits and rebuild/retest."}
        result["register_reuse"]["minimal_change"]["describes_original_experiment_only"] = True
        result["register_reuse"]["acceptance_gate"].insert(0,
            "The first clean Event window rejected9040/globaldefault; do not apply the original one-line mutation. Second-window/boundary evidence and a scoped opt-in are required.")
    fine_timing_summary = OUT / "fine_timing_summary.json"
    if fine_timing_summary.exists():
        result["fine_wait"]["subsequent_gpu_evidence"] = {
            "path": str(fine_timing_summary.relative_to(OUT)), "sha256": sha(fine_timing_summary),
            "full_mutation_allowed": False,
            "decision": "Nine clean Event targets reject the global fine_wait version; five targets regressed in all5 rounds, remaining mixed near-zero signals do not establish gains.",
            "no_expanded_regression_of_rejected_candidate": True,
            "future_distinct_design": "Prepared boundary plans describe required acceptance coverage for a future distinct design; rejection itself needs no further GPU tests."}
        result["fine_wait"]["minimal_change"]["describes_rejected_experiment_only"] = True
        result["fine_wait"]["acceptance_gate"].insert(0,
            "Do not adopt this original helper/three-site version; clean9target GPU results rejected it. Preserve the current conservative wait.")
    positive_summary = OUT / "register_positive_winner_summary.json"
    scoped_audit_path = OUT / "register_reuse_scoped/device_audit.json"
    support_path = OUT / "register_scoped_support_regression.json"
    if positive_summary.exists() and scoped_audit_path.exists() and support_path.exists():
        scoped_audit = json.loads(scoped_audit_path.read_text())
        assert scoped_audit["status"] == "scoped_aliases_match_tested_machine_code"
        result["register_reuse"]["scoped_candidate_current_status"] = {
            "opt_in_aliases": [9042, 9053, 9054], "all_others_default_false": True,
            "six_actual_winner_confirmation": "register_positive_winner_summary.json",
            "six_actual_winners_all5rounds_faster": True,
            "scoped_private_device_audit": "register_reuse_scoped/device_audit.json",
            "sixteen_entry_expected_isa_and_resource_comparisons_passed": True,
            "support_regression": "register_scoped_support_regression.json",
            "runtime_cases_pending": 646, "runtime_shards": 22,
            "fixed_unchanged_domain": 224, "optional_fixed_smokes": 3,
            "production_adopted": False,
            "required_next": "Runtime support/queue/M-tail/N128scale correctness and sufficient performance regression, then official parent TU instruction/resource identity check."}
    closing_path = OUT / "results/register_minimal_closing_analysis.json"
    if closing_path.exists():
        closing = json.loads(closing_path.read_text())
        coverage = closing["exact_current_winner_coverage"]
        assert coverage["affected_actual_winner_count"] == coverage["Event_covered_count"] == 6
        assert not coverage["remaining_event_targets"]
        result["status"] = "cpu_landing_scope_with_finite_register_GPU_evidence_pending_final_integration"
        result["register_reuse"]["subsequent_gpu_evidence"].pop("positive_signal_aliases_pending_second_window_and_boundaries", None)
        result["register_reuse"]["subsequent_gpu_evidence"]["validated_scoped_aliases_pending_final_integration"] = [9042, 9053, 9054]
        result["register_reuse"]["acceptance_gate"][0] = (
            "Reject9040/globaldefault. Scoped9042/9053/9054 have complete6actualwinner Event,5mechanism,6screen Event and full device identity evidence; record final integration separately.")
        current = result["register_reuse"]["scoped_candidate_current_status"]
        current.pop("runtime_cases_pending", None)
        current.update(status="validated_scoped_pending_final_integration", required_next="Final integration source/artifact manifest; official API completion is audited independently.",
            optional_runtime_domain=646, optional_runtime_sweep_stopped=True,
            optional_runtime_verified_cases=closing["optional_support_snapshot"]["summary"]["clean_parent_shape_cases"],
            complete_runtime_support_claimed=False, current_actual_winners_effectively_changed=6,
            current_actual_winners_clean_Event_covered=6, remaining_actual_winner_Event_targets=[],
            new_mechanism_cases_passed=5, new_screen_Event_targets_passed=6,
            material_screen_Event_losses_ge5percent=0, small_nonwinner_Event_losses_percent=[0.810911115234747,0.641480600739075],
            closing_analysis=str(closing_path.relative_to(OUT)), closing_analysis_sha256=sha(closing_path))
        result["limits"].append("Finite register checks completed and are referenced here; optional646 fallback sweep stopped at210 verified cases, not claimed exhaustive.")
    (OUT / "landing_conditions.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": result["status"], "summary": summaries,
                      "register_production_checks": len(register_regressions),
                      "register_queue_boundary_checks": len(register_queue_checks),
                      "register_new_queue_vs_existing": result["register_reuse"]["mandatory_regression"]["new_queue_shapes_excluding_existing_plan"],
                      "fine_boundary_checks": len(fine_boundaries),
                      "fine_new_vs_existing": result["fine_wait"]["mandatory_regression"]["new_unique_boundary_shapes_excluding_existing_plan"]},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
