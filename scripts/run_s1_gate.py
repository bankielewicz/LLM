"""Retain the bounded S1 source and installed-service checks for one candidate."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

from run_s0_gate import authoring_environment, git, source_manifest, write_new, qualified_python
from check_custody import audit, canonical, sha256

PLAN = ('python-runtime', 'authoring-environment', 'custody-before', 's0-regression-gate',
        'bundled-contracts', 'bundled-reader', 's1-wsl-unit-tests', 'native-windows-control',
        'unit-denominator', 'build-wheel', 'wheel-source-binding', 'install-wheel', 'runtime-environment',
        'installed-service-checks', 'custody-after', 'candidate-stability')


class Gate:
    def __init__(self, repo, reports):
        self.repo, self.reports = repo, reports
        self.rows = []
        self.env = os.environ.copy()
        self.env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1', PYTHONPATH=str(repo/'companion/src'))
        self.env['SOURCE_DATE_EPOCH'] = git(repo, 'show', '-s', '--format=%ct', 'HEAD')

    def observe(self, name, action):
        try:
            value = action()
            row = {'gate_id': name, 'status': 'PASS', 'observation': value}
        except Exception as exc:
            row = {'gate_id': name, 'status': 'FAIL', 'reason': str(exc)}
        self.rows.append(row)
        print(name + ': ' + row['status'], flush=True)
        return row

    def command(self, name, argv, *, timeout=300):
        started = time.time()
        try:
            result = subprocess.run(argv, cwd=self.repo, env=self.env, capture_output=True, timeout=timeout)
            stdout, stderr, code = result.stdout, result.stderr, result.returncode
            status = 'PASS' if code == 0 else 'FAIL'
        except subprocess.TimeoutExpired as exc:
            stdout, stderr, code, status = exc.stdout or b'', exc.stderr or b'', None, 'BLOCKED'
        except OSError as exc:
            stdout, stderr, code, status = b'', str(exc).encode(), None, 'BLOCKED'
        row = {'gate_id': name, 'status': status, 'command': argv, 'exit_code': code, 'seconds': round(time.time()-started,3)}
        for label, data in [('stdout', stdout), ('stderr', stderr)]:
            path = self.reports / (name + '.' + label + '.txt')
            path.write_bytes(data)
            row[label] = {'path': path.name, 'size_bytes': len(data), 'sha256': sha256(data)}
        self.rows.append(row)
        print(name + ': ' + status, flush=True)
        return row


def unit_denominator(path, windows_passed):
    root = ET.parse(path).getroot()
    rows = []
    for case in root.iter('testcase'):
        failed = case.find('failure') is not None or case.find('error') is not None
        skipped = case.find('skipped')
        name = case.get('classname', '') + '::' + case.get('name', '')
        state = 'FAIL' if failed else 'NOT_RUN' if skipped is not None else 'PASS'
        rows.append({'test': name, 'status': state, 'reason': skipped.get('message') if skipped is not None else None})
    if not rows or any(row['status'] == 'FAIL' for row in rows):
        raise ValueError('Unit denominator is empty or contains a failure')
    skipped = [row for row in rows if row['status'] == 'NOT_RUN']
    allowed = {'companion.tests.test_auth_control::test_actual_windows_named_pipe_has_no_tcp_fallback', 'test_auth_control::test_actual_windows_named_pipe_has_no_tcp_fallback'}
    # The exact Windows-only test is retained as NOT_RUN in this WSL report.
    # Its independent native process probe is a separately required gate.
    if skipped and (not windows_passed or any(row['test'] not in allowed for row in skipped)):
        raise ValueError('Unexplained unit skips or missing native Windows control evidence: ' + str(skipped))
    return {'total': len(rows), 'passed': len(rows)-len(skipped), 'failed': 0, 'not_run': len(skipped), 'cases': rows,
            'scope': 'WSL unit suite; separately executed native Windows IPC probe does not replace a skipped pytest case or qualify a runtime profile.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir', type=Path, required=True)
    parser.add_argument('--runtime-python', type=Path, required=True)
    parser.add_argument('--windows-python', required=True)
    parser.add_argument('--node', required=True)
    parser.add_argument('--canonical-course', type=Path, required=True)
    parser.add_argument('--legacy-python', required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    reports = args.report_dir.resolve()
    if reports.is_relative_to(repo) or repo.is_relative_to(reports):
        parser.error('Keep retained S1 reports outside the source worktree')
    reports.mkdir(parents=True, exist_ok=False)
    gate = Gate(repo, reports)
    before = source_manifest(repo, reports)
    source_id = sha256(canonical(before))
    write_new(reports/'source-manifest.json', {'candidate_id': source_id, 'files': before})
    head, tree = git(repo, 'rev-parse', 'HEAD'), git(repo, 'rev-parse', 'HEAD^{tree}')
    try:
        gate.observe('python-runtime', qualified_python)
        gate.observe('authoring-environment', lambda: authoring_environment(repo))
        gate.observe('custody-before', lambda: audit(repo, repo/'docs/implementation/s0/authority.json'))
        gate.command('s0-regression-gate', [sys.executable, '-B', 'scripts/run_s0_gate.py', '--report-dir', str(reports/'s0'), '--node', args.node, '--canonical-course', str(args.canonical_course), '--legacy-python', args.legacy_python], timeout=600)
        gate.command('bundled-contracts', [sys.executable, '-B', 'scripts/build_runtime_contracts.py', '--check'])
        gate.command('bundled-reader', [sys.executable, '-B', 'scripts/build_runtime_assets.py', '--check'])
        gate.command('s1-wsl-unit-tests', [sys.executable, '-B', '-m', 'pytest', 'companion/tests', '-p', 'no:cacheprovider', '-v', '--junitxml', str(reports/'unit-tests.xml')], timeout=600)
        native_probe = repo/'companion/tests/native_windows_control_probe.py'
        native_path = subprocess.run(['wslpath', '-w', str(native_probe)], check=True, capture_output=True, text=True).stdout.strip()
        windows = gate.command('native-windows-control', [args.windows_python, native_path], timeout=60)
        gate.observe('unit-denominator', lambda: unit_denominator(reports/'unit-tests.xml', windows['status']=='PASS'))
        built = gate.command('build-wheel', [sys.executable, '-B', '-m', 'build', '--wheel', '--no-isolation', '--outdir', str(reports/'wheel'), 'companion'])
        wheels = list((reports/'wheel').glob('*.whl')) if built['status']=='PASS' else []
        if len(wheels) == 1:
            wheel = wheels[0]
            binding = gate.command('wheel-source-binding', [sys.executable, '-B', 'scripts/check_s1_wheel.py', str(wheel), str(repo)])
            if binding['status'] != 'PASS':
                raise ValueError('Candidate wheel does not match source')
            installed = gate.command('install-wheel', [str(args.runtime_python), '-I', '-m', 'pip', 'install', '--no-deps', '--force-reinstall', str(wheel)])
            if installed['status']=='PASS':
                gate.command('runtime-environment', [str(args.runtime_python), '-I', str(repo/'scripts/check_s1_runtime.py'), str(repo/'companion/locks/wsl-cpu.wheels.json')])
                gate.command('installed-service-checks', [sys.executable, '-B', 'scripts/run_s1_installed_checks.py', '--python', str(args.runtime_python), '--wheel', str(wheel), '--report-dir', str(reports/'installed')], timeout=300)
        gate.observe('custody-after', lambda: audit(repo, repo/'docs/implementation/s0/authority.json'))
        def stable():
            assert source_manifest(repo, reports) == before, 'Source candidate changed while checks ran'
            assert git(repo,'rev-parse','HEAD') == head, 'Commit changed while checks ran'
            return {'candidate_id': source_id, 'git_commit': head, 'git_tree': tree}
        gate.observe('candidate-stability', stable)
    except Exception as exc:
        gate.rows.append({'gate_id': 'gate-execution', 'status': 'FAIL', 'reason': str(exc)})
    seen = {row['gate_id'] for row in gate.rows}
    gate.rows.extend({'gate_id': name, 'status': 'BLOCKED', 'reason': 'A required prerequisite did not complete.'} for name in PLAN if name not in seen)
    passed = len(gate.rows)==len(PLAN) and all(row['status']=='PASS' for row in gate.rows)
    artifacts = []
    for path in sorted(reports.rglob('*')):
        if path.is_file():
            artifacts.append({'path': path.relative_to(reports).as_posix(), 'size_bytes': path.stat().st_size, 'sha256':sha256(path.read_bytes())})
    write_new(reports/'result.json', {'format':'llm-foundations-s1-gate-v1', 's1_source_service_gate':'PASS' if passed else 'FAIL',
        'candidate_id':source_id,'git_commit':head,'git_tree':tree,'gates':gate.rows,'artifacts':artifacts,
        'acceptance_execution_units':605,'product_qualification':'NOT_RUN',
        'native_profile_qualification':{p:'NOT_RUN' for p in ('win-cpu','win-cuda','wsl-cpu','wsl-cuda')}})
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
