"""Check the preparation and default execution gates without GPU imports."""

import ast
import contextlib
import hashlib
import importlib.abc
import importlib.util
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

BUNDLE = Path(__file__).resolve().parent
FORBIDDEN = {'torch', 'aiter', 'csrc'}


class RejectGpuImports(importlib.abc.MetaPathFinder):
    def __init__(self):
        self.attempts = []

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in FORBIDDEN:
            self.attempts.append(fullname)
            raise AssertionError(f'CPU command attempted a forbidden import: {fullname}')
        return None


def main():
    blocker = RejectGpuImports()
    if any(name.split('.')[0] in FORBIDDEN for name in sys.modules):
        raise AssertionError('Run this CPU check in a fresh Python process')
    sys.meta_path.insert(0, blocker)
    sys.path.insert(0, str(BUNDLE))
    checks = []
    sources = sorted(BUNDLE.glob('*.py'))
    for source in sources:
        ast.parse(source.read_text(), filename=str(source))
    checks.append({'name': 'all_python_sources_parse', 'passed': True,
                   'files': [path.name for path in sources]})
    spec = importlib.util.spec_from_file_location('prepared_recheck', BUNDLE / 'run_recheck.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for args, shapes, pairs in [([], 2, 4),
                               (['check', '--scope', 'all'], 14, 26),
                               (['plan', '--scope', 'priority'], 2, 4),
                               (['plan', '--scope', 'remaining'], 12, 22)]:
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = module.main(args)
        if code != 0:
            raise AssertionError(f'CPU command failed: {args}, {stderr.getvalue()}')
        plan = json.loads(stdout.getvalue())
        if plan['gpu_used'] or plan['selected_shapes'] != shapes or plan['selected_pairs'] != pairs:
            raise AssertionError(f'Unexpected CPU plan: {args}')
        checks.append({'name': 'default_CPU_plan' if not args else ' '.join(args),
                       'passed': True, 'shapes': shapes, 'pairs': pairs,
                       'gpu_used': False})
    # These intentionally omit a required gate. None can enter execute_gpu().
    for args in [['execute'], ['execute', '--gpu', '6'], ['--execute-gpu']]:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            try:
                module.main(args)
            except SystemExit as error:
                code = error.code
            else:
                raise AssertionError(f'Missing GPU gate was accepted: {args}')
        if code != 2:
            raise AssertionError(f'Unexpected rejection code: {args}, {code}')
        checks.append({'name': 'reject incomplete GPU gate: ' + ' '.join(args),
                       'passed': True, 'exit_code': code, 'gpu_used': False})
    loaded = [name for name in sys.modules if name.split('.')[0] in FORBIDDEN]
    if blocker.attempts or loaded:
        raise AssertionError(f'Forbidden imports: attempts={blocker.attempts}, loaded={loaded}')
    analyzer = json.loads((BUNDLE / 'cpu_analyzer_checks.json').read_text())
    if not analyzer['passed'] or not analyzer['synthetic'] or analyzer['gpu_tests_launched']:
        raise AssertionError('The separate analyzer checks did not pass purely on CPU')
    report = {'report_kind': 'CPU preparation import and execution-gate checks',
              'checked_utc': datetime.now(timezone.utc).isoformat(),
              'passed': True, 'gpu_used': False, 'synthetic': True,
              'gpu_results_written': False, 'forbidden_import_attempts': blocker.attempts,
              'forbidden_modules_loaded': loaded, 'checks': checks,
              'analyzer_synthetic_cases_passed': analyzer['case_count'],
              'bundle_python_sha256': {source.name: hashlib.sha256(source.read_bytes()).hexdigest()
                                       for source in sources}}
    (BUNDLE / 'cpu_preparation_checks.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'passed': True, 'GPU_used': False, 'checks': len(checks),
                      'analyzer_synthetic_cases': analyzer['case_count'],
                      'forbidden_imports': loaded, 'report': str(BUNDLE / 'cpu_preparation_checks.json')}))


if __name__ == '__main__':
    main()
