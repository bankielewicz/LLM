"""Malformed backend reports never become a successful preflight or escape parsing."""
import copy
import json

import pytest

from llm_foundations_companion import preflight
from test_preflight import cuda_report, selection


@pytest.mark.parametrize(('path', 'bad'), [
    (('format',), 'unknown'),
    (('backend_status',), 'unknown'), (('backend_status',), []),
    (('device_available',), 1),
    (('versions', 'torch'), 'x'*81), (('versions', 'peft'), 1),
    (('error_text',), 'x'*2001), (('error_text',), None),
    (('cuda', 'result'), 'unknown'), (('cuda', 'result'), []),
    (('cuda', 'reason_code'), 'unknown'), (('cuda', 'reason_code'), []),
    (('cuda', 'reason_code'), 'CUDA_DEVICE_NOT_FOUND'),
    (('cuda', 'device'), None),
    (('cuda', 'device', 'name'), ''),
    (('cuda', 'device', 'count'), True),
    (('cuda', 'device', 'compute_capability'), '120'),
    (('cuda', 'device', 'arch_list'), ['sm_75']*33),
    (('cuda', 'device', 'arch_list'), ['arbitrary']),
    (('cuda', 'device', 'total_memory_mib'), -1),
    (('cuda', 'device', 'free_memory_mib'), True),
    (('cuda', 'device', 'max_relative_difference'), -0.1),
    (('cuda', 'device', 'max_relative_difference'), True),
    (('cuda', 'device', 'max_relative_difference'), None),
])
def test_malformed_child_report_is_downgraded_without_exception(path, bad):
    report = copy.deepcopy(cuda_report())
    parent = report
    for name in path[:-1]:
        parent = parent[name]
    parent[path[-1]] = bad
    raw = json.dumps(report).encode()
    execution = preflight.run_preflight_child(selection('wsl-cuda'),
        process_runner=lambda *args, **kwargs: preflight.ProcessResult(returncode=0, stdout=raw))
    assert execution.valid_child_output is False
    assert execution.report['backend_status'] == 'missing'
    assert execution.report['device_available'] is False


def test_failed_cuda_report_requires_reason():
    report = cuda_report(result='failed', reason_code=None)
    with pytest.raises(ValueError, match='lacks a reason'):
        preflight.validate_child_report(report)


def test_duplicate_keys_in_backend_child_json_are_rejected():
    raw = b'{"format":"one","format":"two"}'
    execution = preflight.run_preflight_child(selection(),
        process_runner=lambda *args, **kwargs: preflight.ProcessResult(returncode=0, stdout=raw))
    assert not execution.valid_child_output
    assert execution.report['backend_status'] == 'missing'
