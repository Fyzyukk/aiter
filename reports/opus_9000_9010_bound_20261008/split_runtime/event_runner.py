from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,'/root/workspace/aiter-opus-mxfp8-bpreshuffle/reports/opus_9000_9010_bound_20261008/split_runtime')
import experiment_runner as runner
from official_smoke import OfficialRunner
module=json.loads(Path('/root/workspace/aiter-opus-mxfp8-bpreshuffle/reports/opus_9000_9010_bound_20261008/source_manifest.json').read_text())['baseline'];private=runner.Runner

def mixed(path,workspace=False):
 if Path(path).resolve()==Path(module['path']).resolve():
  r=OfficialRunner(1,module['path'],module['sha256']);r.split_count=None;return r
 return private(path,workspace)
runner.Runner=mixed
if __name__=='__main__':runner.main()
