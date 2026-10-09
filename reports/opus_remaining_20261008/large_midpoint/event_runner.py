#!/usr/bin/env python3
"""Use the exact official Oct8 baseline and isolated private candidate in graphs.

All timing/check machinery is the retained experiment_runner. Baseline uses
its public API wrapper, verifying the actually loaded module. Both captured
graphs contain the same exact selected device dispatch for each target.
"""
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OLD=ROOT/'reports/opus_bound_analysis_20261007'
sys.path.insert(0,str(OLD))
import experiment_runner as runner
from official_smoke import OfficialRunner

inventory=json.loads((HERE.parent/'inventory.json').read_text())
module=inventory['official_module']
private_runner=runner.Runner


def mixed_runner(path, workspace=False):
    if Path(path).resolve()==Path(module['path']).resolve():
        assert not workspace
        result=OfficialRunner(1,module['path'],module['sha256'])
        result.split_count=None
        return result
    return private_runner(path,workspace)


runner.Runner=mixed_runner
if __name__=='__main__':runner.main()
