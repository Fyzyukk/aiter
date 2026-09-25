"""Complete audit, confirmation, export, replay, and report after the sweep."""
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
state = dict(status="running", stage="waiting_for_full_sweep", start_time=time.time(), steps=[])


def save():
    tmp = HERE / "pipeline.json.tmp"
    tmp.write_text(json.dumps(state, indent=2) + "\n")
    tmp.replace(HERE / "pipeline.json")


try:
    assert not (HERE / "pipeline.json").exists()
    save()
    while True:
        run = json.loads((HERE / "launcher.json").read_text())
        if run["status"] != "running":
            assert run["status"] == "passed", run
            break
        time.sleep(10)
    for name in ("merge", "summarize", "confirm", "finalize", "parallel_replay", "report"):
        state["stage"] = name
        save()
        print("Starting", name, flush=True)
        with (HERE / f"{name}.log").open("x") as log:
            result = subprocess.run([sys.executable, "-u", str(HERE / f"{name}.py")],
                                    stdout=log, stderr=subprocess.STDOUT)
        state["steps"].append(dict(step=name, exit_code=result.returncode, end_time=time.time()))
        save()
        assert result.returncode == 0, f"{name} failed; see {name}.log"
    state.update(status="passed", stage="complete")
except BaseException as exc:
    state.update(status="failed", error=repr(exc))
    raise
finally:
    state["end_time"] = time.time()
    save()
print(json.dumps(state), flush=True)
