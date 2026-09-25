"""Apply entire confirmation batches, without selecting minima across batches."""
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


def read(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def write(name, rows, fields):
    with (HERE / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(rows)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def shape(r):
    return tuple(int(r[k]) for k in KEYS)


def identity(r):
    return (*shape(r), r["lib"], int(r["kid"]), int(r["splitK"]), r["kernelName"])


full = json.loads((HERE / "full_r3_run.json").read_text())
confirm = json.loads((HERE / "confirmation.json").read_text())
assert full["status"] == confirm["status"] == "passed"
comparison = {shape(r): r for r in read(HERE / "comparison.csv")}
expected_shapes = {shape(r) for r in read(HERE / "close_shapes.csv")}
updated, manifests = set(), []
effective = defaultdict(list)
effective_raw = defaultdict(list)
for row in read(HERE / "candidate_medians.csv"):
    if shape(row) not in expected_shapes:
        row["median_us"] = float(row["median_us"])
        effective[shape(row)].append(row)
for row in read(HERE / "full_r3_raw.csv"):
    if shape(row) not in expected_shapes and row["status"] == "passed":
        effective_raw[identity(row)].append(row)
for worker in confirm["workers"]:
    path = Path(worker["prefix"] + "_run.json")
    run = json.loads(path.read_text())
    assert run["status"] == "passed" and run["mode"] == "final" and run["rounds"] == 5
    assert run["completed_shapes"] == worker["shapes"]
    own = {tuple(s) for s in run["shapes"]}
    assert own <= expected_shapes and not updated.intersection(own)
    updated.update(own)
    sweep = json.loads(Path(run["sweep_input"]["path"]).read_text())
    assert sha(run["sweep_input"]["path"]) == run["sweep_input"]["sha256"]
    assert run["gpu_uuid"] == sweep["gpu_uuid"]
    for key in ("source_sha256", "binary_sha256", "reference_sha256", "harness_sha256"):
        assert run[key] == full[key], key
    for kind, digest in run["output_sha256"].items():
        assert sha(run["outputs"][kind]) == digest, (path, kind)
    for s in own:
        assert int(comparison[s]["gpu"]) == run["gpu"]
    candidates = read(run["outputs"]["candidates"])
    selected = {identity(r): r for r in candidates if r["selected"] == "True"}
    assert len(selected) == sum(r["selected"] == "True" for r in candidates)
    checks, raw = defaultdict(list), defaultdict(list)
    for r in read(run["outputs"]["correctness"]):
        checks[identity(r)].append(r)
    for r in read(run["outputs"]["raw"]):
        raw[identity(r)].append(r)
    assert set(checks) == set(selected)
    for key, row in selected.items():
        failures = [r for r in checks[key] if r["status"] != "passed" or float(r["error"] or "nan") != 0]
        assert sum(r["phase"] == "preflight" for r in checks[key]) == 1
        if failures:
            assert row["lib"] != "opus", failures
            continue
        samples = raw[key]
        assert len(samples) == 5 and {int(r["round"]) for r in samples} == set(range(5))
        assert all(r["status"] == "passed" and float(r["error"]) == 0 for r in samples)
        values = [float(r["us"]) for r in samples]
        assert all(math.isfinite(v) and v > 0 for v in values)
        effective[shape(row)].append(dict(row, median_us=statistics.median(values),
                                          min_us=min(values), max_us=max(values), samples=5))
        effective_raw[key] = samples
    manifests.append(dict(path=str(path), sha256=sha(path), gpu=run["gpu"], shapes=len(own)))
assert updated == expected_shapes and set(effective) == set(comparison)
for p, h in full["source_sha256"].items():
    assert sha(ROOT / p) == h, p
for p, h in full["binary_sha256"].items():
    assert sha(p) == h, p

final, tuned, opus_best = [], [], []
for s, rows in sorted(effective.items(), key=lambda x: (x[0][1], x[0][2], x[0][0])):
    opus = min((r for r in rows if r["lib"] == "opus"), key=lambda r: r["median_us"])
    ref = min((r for r in rows if r["lib"] != "opus"), key=lambda r: r["median_us"])
    best = min(rows, key=lambda r: (r["median_us"], identity(r)))
    old = comparison[s]
    won = opus["median_us"] < ref["median_us"]
    previous = old["previous_ledger_opus_wins"] == "True"
    rounds = 5 if s in updated else 3
    wins_by_round = []
    for turn in range(rounds):
        o = min(float(sample["us"]) for r in rows if r["lib"] == "opus"
                for sample in effective_raw[identity(r)] if int(sample["round"]) == turn)
        b = min(float(sample["us"]) for r in rows if r["lib"] != "opus"
                for sample in effective_raw[identity(r)] if int(sample["round"]) == turn)
        wins_by_round.append(o < b)
    row = dict(zip(KEYS, s), gpu=int(old["gpu"]), gpu_uuid=old["gpu_uuid"],
               measurement_phase="confirmation" if s in updated else "full_sweep", rounds=rounds,
               opus=opus["name"], opus_us=opus["median_us"], reference=ref["name"],
               reference_us=ref["median_us"], reference_over_opus=ref["median_us"]/opus["median_us"],
               opus_slower_pct=(opus["median_us"]/ref["median_us"]-1)*100, opus_wins=won,
               exact_tie=opus["median_us"] == ref["median_us"], selected=best["name"],
               selected_us=best["median_us"], selected_backend=best["lib"],
               opus_faster_rounds=sum(wins_by_round), round_status_consistent=len(set(wins_by_round)) == 1,
               previous_ledger_opus_wins=previous,
               status_vs_previous_ledger=("still_wins" if previous else "new_win") if won else
                                         ("new_loss" if previous else "still_loses"),
               full_sweep_opus_wins=old["opus_wins"])
    for lib in ("ck", "cktile", "asm"):
        local = [r for r in rows if r["lib"] == lib]
        winner = min(local, key=lambda r: r["median_us"]) if local else None
        row[lib+"_best"] = winner["name"] if winner else ""
        row[lib+"_us"] = winner["median_us"] if winner else ""
    for kid in (9000, 9010, 9011, 9012, 9020):
        winner = next((r for r in rows if r["name"] == f"opus_{kid}"), None)
        row[f"opus_{kid}_us"] = winner["median_us"] if winner else ""
    final.append(row)
    for r, dest in ((best, tuned), (opus, opus_best)):
        m, n, k = s
        us = r["median_us"]
        dest.append(dict(gfx="gfx950", cu_num=256, M=m, N=n, K=k, libtype=r["lib"],
                         kernelId=int(r["kid"]), splitK=int(r["splitK"]), us=us,
                         kernelName=r["kernelName"], tflops=2*m*n*k/us/1e6,
                         bw=(m*k+n*k+2*m*n)/us/1000, errRatio=0))
write("final_comparison.csv", final, list(final[0]))
write("final_tuned.csv", tuned, SCHEMA)
write("final_opus_best.csv", opus_best, SCHEMA)
losses = [r for r in final if not r["opus_wins"]]
write("final_remaining_shapes.csv", losses, list(final[0]))
close = [r for r in final if abs(r["opus_slower_pct"]) <= 3 or not r["round_status_consistent"]]
write("final_close_shapes.csv", close, list(final[0]))
groups = defaultdict(list)
for r in final:
    groups[r["N"], r["K"]].append(r)
group_rows = []
for (n, k), rows in groups.items():
    slow = [r for r in rows if not r["opus_wins"]]
    group_rows.append(dict(N=n, K=k, total=len(rows), opus_wins=len(rows)-len(slow),
                           remaining=len(slow), remaining_M=";".join(str(r["M"]) for r in slow)))
write("final_group_summary.csv", group_rows, list(group_rows[0]))
stats = dict(status="passed", shapes=len(final), deferred_addressing_shapes=10,
             full_sweep=json.loads((HERE / "summary.json").read_text()),
             confirmation_shapes=len(updated), confirmation_rounds=5, confirmation_runs=manifests,
             policy="Use five-round confirmation batch for all selected shapes; three-round full sweep elsewhere; no cross-batch minimum",
             opus_wins=len(final)-len(losses), remaining=len(losses),
             selected_backends=dict(Counter(r["selected_backend"] for r in final)),
             best_opus_kids=dict(Counter(r["opus"] for r in final)),
             winning_opus_kids=dict(Counter(r["opus"] for r in final if r["opus_wins"])),
             transitions_vs_previous_ledger=dict(Counter(r["status_vs_previous_ledger"] for r in final)),
             changes_from_full_sweep=sum(r["opus_wins"] != (r["full_sweep_opus_wins"] == "True") for r in final),
             exact_ties=sum(r["exact_tie"] for r in final), close_or_inconsistent_shapes=len(close),
             losses_within_3pct=sum(abs(r["opus_slower_pct"]) <= 3 for r in losses),
             losses_over_3pct=sum(r["opus_slower_pct"] > 3 for r in losses),
             geomean_reference_over_opus=math.exp(statistics.mean(math.log(r["reference_over_opus"]) for r in final)))
(HERE / "final_summary.json").write_text(json.dumps(stats, indent=2) + "\n")
print(json.dumps({k:v for k,v in stats.items() if k not in ("full_sweep", "confirmation_runs")}, indent=2))
