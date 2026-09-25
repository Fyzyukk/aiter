"""Audit the new full sweep and export native-E8M0 tuner selections."""
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
KEYS = ("M", "N", "K")
SCHEMA = ["gfx", "cu_num", "M", "N", "K", "libtype", "kernelId", "splitK",
          "us", "kernelName", "tflops", "bw", "errRatio"]


def read(name):
    with (HERE / name).open(newline="") as f:
        return list(csv.DictReader(f))


def write(name, rows, fields=None):
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    with (HERE / name).open("w", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)


def shape(row):
    return tuple(int(row[k]) for k in KEYS)


def identity(row):
    return (*shape(row), row["lib"], int(row["kid"]), int(row["splitK"]), row["kernelName"])


run = json.loads((HERE / "full_r3_run.json").read_text())
assert run["status"] == "passed", "Only summarize a completed full sweep"
assert run["rounds"] == 3 and run["completed_shapes"] == 295
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
for path, digest in run["source_sha256"].items():
    assert sha(ROOT / path) == digest, path
for path, digest in run["binary_sha256"].items():
    assert sha(path) == digest, path
for kind, digest in run["output_sha256"].items():
    assert sha(run["outputs"][kind]) == digest, kind
expected = read("expected_all_candidates.csv")
expected_ids = {(*shape(r), r["libtype"], int(r["kernelId"]), int(r["splitK"]), r["kernelName"]) for r in expected}
candidates = read("full_r3_candidates.csv")
assert len(candidates) == len(expected_ids) == 27690
assert {identity(r) for r in candidates} == expected_ids
raw = read("full_r3_raw.csv")
checks = read("full_r3_correctness.csv")
raw_by = defaultdict(list)
checks_by = defaultdict(list)
for row in raw:
    raw_by[identity(row)].append(row)
for row in checks:
    checks_by[identity(row)].append(row)
assert set(checks_by) == expected_ids
assert all(sum(r["phase"] == "preflight" for r in rs) == 1 for rs in checks_by.values())
medians, rejected, profile = [], [], []
for candidate in candidates:
    key = identity(candidate)
    samples = raw_by[key]
    assert len({int(r["round"]) for r in samples}) == len(samples)
    failures = [r for r in checks_by[key] if r["status"] != "passed"]
    valid = not failures and len(samples) == 3
    out = {k: candidate[k] for k in (*KEYS, "name", "lib", "kid", "splitK", "kernelName")}
    if valid:
        values = [float(r["us"]) for r in samples]
        assert {int(r["round"]) for r in samples} == {0, 1, 2}
        assert all(r["status"] == "passed" and float(r["error"]) == 0 for r in samples)
        assert all(math.isfinite(v) and v > 0 for v in values)
        out.update(median_us=statistics.median(values), min_us=min(values), max_us=max(values), samples=3)
        medians.append(out)
        us, error = out["median_us"], 0
    else:
        assert failures, "Missing measurements without a recorded failure"
        out.update(reason=failures[0]["reason"], error=failures[0]["error"], phase=failures[0]["phase"])
        rejected.append(out)
        us, error = "", failures[0]["error"] or 1
    m, n, k = shape(candidate)
    profile.append(dict(gfx="gfx950", cu_num=256, M=m, N=n, K=k,
                        libtype=candidate["lib"], kernelId=int(candidate["kid"]),
                        splitK=int(candidate["splitK"]), us=us, kernelName=candidate["kernelName"],
                        tflops=(2*m*n*k/us/1e6 if us else ""),
                        bw=((m*k+n*k+2*m*n)/us/1000 if us else ""), errRatio=error))
assert not any(r["lib"] == "opus" for r in rejected), "OPUS correctness failure needs investigation"
assert len([r for r in medians if r["lib"] == "opus"]) == 1275
write("candidate_medians.csv", medians)
write("rejected_candidates.csv", rejected)
write("profile.csv", profile, SCHEMA)
all_valid = {(*shape(r), r["libtype"], r["kernelId"], r["splitK"]): r for r in profile if r["us"] != ""}
by_shape = defaultdict(list)
for row in medians:
    by_shape[shape(row)].append(row)
assert set(by_shape) == {shape(r) for r in read("shapes.csv")}
previous = {shape(r): r for r in read("previous_progress_ledger.csv")}
comparisons, tuned, opus_best, close = [], [], [], []
for key, rows in sorted(by_shape.items(), key=lambda kv: (kv[0][1], kv[0][2], kv[0][0])):
    best = min(rows, key=lambda r: (r["median_us"], identity(r)))
    opus = min((r for r in rows if r["lib"] == "opus"), key=lambda r: r["median_us"])
    reference = min((r for r in rows if r["lib"] != "opus"), key=lambda r: r["median_us"])
    for result, collection in ((best, tuned), (opus, opus_best)):
        collection.append(all_valid[(*key, result["lib"], int(result["kid"]), int(result["splitK"]))])
    ref_us, opus_us = reference["median_us"], opus["median_us"]
    margin = (opus_us / ref_us - 1) * 100
    round_status = []
    for turn in range(3):
        o = min(float(r["us"]) for c in rows if c["lib"] == "opus" for r in raw_by[identity(c)] if int(r["round"]) == turn)
        b = min(float(r["us"]) for c in rows if c["lib"] != "opus" for r in raw_by[identity(c)] if int(r["round"]) == turn)
        round_status.append(o < b)
    old_won = previous[key]["opus_wins"] == "True"
    won = opus_us < ref_us
    status = "still_wins" if won and old_won else "new_win" if won else "new_loss" if old_won else "still_loses"
    comparison = dict(zip(KEYS, key), opus=opus["name"], opus_us=opus_us,
                      reference=reference["name"], reference_us=ref_us,
                      reference_over_opus=ref_us/opus_us, opus_slower_pct=margin,
                      opus_wins=won, exact_tie=opus_us == ref_us,
                      selected=best["name"], selected_us=best["median_us"],
                      selected_backend=best["lib"], opus_faster_rounds=sum(round_status),
                      round_status_consistent=len(set(round_status)) == 1,
                      previous_ledger_opus_wins=old_won, status_vs_previous_ledger=status)
    for lib in ("ck", "cktile", "asm"):
        local = [r for r in rows if r["lib"] == lib]
        winner = min(local, key=lambda r: r["median_us"]) if local else None
        comparison[lib+"_best"] = winner["name"] if winner else ""
        comparison[lib+"_us"] = winner["median_us"] if winner else ""
    for kid in (9000, 9010, 9011, 9012, 9020):
        winner = next((r for r in rows if r["name"] == f"opus_{kid}"), None)
        comparison[f"opus_{kid}_us"] = winner["median_us"] if winner else ""
    comparisons.append(comparison)
    if abs(margin) <= 3 or len(set(round_status)) != 1:
        close.append(dict(gfx="gfx950", cu_num=256, **dict(zip(KEYS, key))))
write("comparison.csv", comparisons)
write("tuned.csv", tuned, SCHEMA)
write("opus_best.csv", opus_best, SCHEMA)
write("close_shapes.csv", close, ["gfx", "cu_num", *KEYS])
losses = [r for r in comparisons if not r["opus_wins"]]
write("remaining_shapes.csv", losses, list(comparisons[0]))
groups = defaultdict(list)
for row in comparisons:
    groups[row["N"], row["K"]].append(row)
group_rows = []
for (n, k), rows in groups.items():
    slow = [r for r in rows if not r["opus_wins"]]
    group_rows.append(dict(N=n, K=k, total=len(rows), opus_wins=len(rows)-len(slow),
                           remaining=len(slow), remaining_M=";".join(str(r["M"]) for r in slow),
                           geomean_reference_over_opus=math.exp(statistics.mean(math.log(r["reference_over_opus"]) for r in rows))))
write("group_summary.csv", group_rows)
summary = dict(status="passed", shapes=295, deferred_addressing_shapes=10,
               candidates=len(candidates), timing_records=len(raw), correctness_checks=len(checks), rounds=3,
               candidates_by_backend=dict(Counter(r["lib"] for r in candidates)),
               valid_candidates_by_backend=dict(Counter(r["lib"] for r in medians)),
               rejected_candidates_by_backend=dict(Counter(r["lib"] for r in rejected)),
               selected_backends=dict(Counter(r["selected_backend"] for r in comparisons)),
               opus_wins=295-len(losses), remaining=len(losses),
               exact_ties=sum(r["exact_tie"] for r in comparisons),
               best_opus_kids=dict(Counter(r["opus"] for r in comparisons)),
               winning_opus_kids=dict(Counter(r["opus"] for r in comparisons if r["opus_wins"])),
               transitions_vs_previous_ledger=dict(Counter(r["status_vs_previous_ledger"] for r in comparisons)),
               close_or_inconsistent_shapes=len(close),
               geomean_reference_over_opus=math.exp(statistics.mean(math.log(r["reference_over_opus"]) for r in comparisons)),
               source_sha256_match=True, binary_sha256_match=True,
               native_scale_dtype="e8m0", performance_full_retest=True)
(HERE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
