"""Pure preflight policy/report checks; these do not qualify GPU hardware."""
from pathlib import Path

import pytest

from llm_foundations_companion import preflight
from test_preflight import (child_execution, cuda_report, gpu_result,
    passing_cpu_report, selection, storage_probe)


def test_unexpected_distribution_metadata_failure_is_backend_missing():
    def broken(name): raise OSError('metadata inaccessible')
    chosen = preflight.select_profile(None, platform_kind='wsl', torch_version_resolver=broken)
    assert chosen.profile == 'wsl-cpu' and chosen.backend_reason_code == 'BACKEND_MISSING'


@pytest.mark.parametrize('operation', ['select', 'query'])
def test_unknown_platform_cannot_select_or_query_gpu(operation):
    with pytest.raises(preflight.UnsupportedPlatformError):
        if operation == 'select': preflight.select_profile(None, platform_kind='other')
        else: preflight.run_gpu_query('other')


@pytest.mark.parametrize('name', ['', 'x'*201])
def test_gpu_query_rejects_empty_or_excessive_name(name):
    with pytest.raises(ValueError, match='name'):
        preflight.parse_gpu_query_line(name+', 12.0, 616.92, 12227')


@pytest.mark.parametrize(('call', 'value'), [(preflight._version_tuple,'unknown'), (preflight._capability_tuple,'12')])
def test_hardware_version_parsers_do_not_accept_unstructured_values(call, value):
    with pytest.raises(ValueError): call(value)


@pytest.mark.parametrize(('profile','relative'), [('wsl-cpu','bin/python'), ('win-cpu','Scripts/python.exe')])
def test_sibling_cuda_environment_detection_requires_existing_interpreter(tmp_path, profile, relative):
    cpu = tmp_path/'envs'/profile/relative
    cpu.parent.mkdir(parents=True); cpu.write_bytes(b'placeholder')
    chosen = selection(profile)
    assert not preflight.detect_cuda_environment_installed(chosen, interpreter=cpu)
    cuda = tmp_path/'envs'/profile.replace('cpu','cuda')/relative
    cuda.parent.mkdir(parents=True); cuda.write_bytes(b'placeholder')
    assert preflight.detect_cuda_environment_installed(chosen, interpreter=cpu)


def test_cuda_offer_requires_successful_fixed_query():
    query = preflight.GpuQueryResult(None,None,'CUDA_DEVICE_NOT_FOUND',preflight.ProcessResult(1))
    with pytest.raises(preflight.PreflightError) as raised:
        preflight.gpu_offer(selection('wsl-cuda'),query,cuda_environment_installed=False)
    assert raised.value.reason_code == 'CUDA_DEVICE_NOT_FOUND'


@pytest.mark.parametrize(('status','reason','fragment'), [
    ('not_detected',None,'CPU environment'),
    ('available',None,'installed'),
    ('unsupported','CUDA_DRIVER_TOO_OLD','572.61'),
    ('unsupported','CUDA_CAPABILITY_UNSUPPORTED','7.5'),
    ('unsupported','CUDA_MEMORY_INSUFFICIENT','3,584'),
    ('in_use',None,'using the GPU'),
])
def test_gpu_offer_explanation_reflects_actionable_reason(status, reason, fragment):
    offer = {'status':status,'reason_code':reason,'cuda_profile':'wsl-cuda',
        'cuda_environment_installed':True,'gpu':{'name':'Test GPU','driver_version':'570.0',
        'compute_capability':'6.0','memory_total_mib':2048}}
    assert fragment in preflight.gpu_check_message(offer)


@pytest.mark.parametrize('offer', [{'status':'unknown'}, {'status':'unsupported','reason_code':'unknown'}])
def test_gpu_offer_explanation_rejects_unknown_status_or_reason(offer):
    with pytest.raises(ValueError): preflight.gpu_check_message(offer)


@pytest.mark.parametrize('mode', ['old_driver','missing_backend','wrong_backend'])
def test_cuda_startup_failure_reports_exact_reason_and_cpu_command(mode):
    report = cuda_report()
    query = gpu_result('Test GPU, 12.0, 570.0, 12227') if mode=='old_driver' else gpu_result()
    if mode=='missing_backend': report['backend_status']='missing'
    elif mode=='wrong_backend': report['backend_status']='version_mismatch'
    expected = {'old_driver':'CUDA_DRIVER_TOO_OLD','missing_backend':'BACKEND_MISSING','wrong_backend':'BACKEND_VERSION_MISMATCH'}[mode]
    with pytest.raises(preflight.ProfileSelectionError) as raised:
        preflight.require_startup_allowed(selection('win-cuda'),child_execution(report),query,storage_root_display='lesson-data')
    assert raised.value.reason_code == expected
    assert 'envs\\win-cpu\\Scripts\\python.exe' in raised.value.cpu_command
    assert '--storage lesson-data' in raised.value.cpu_command


def test_architecture_report_skips_malformed_entries_and_accepts_compatible_native_code():
    assert not preflight._compiled_architecture_supports(['compute_120','sm_invalid',7],(12,0))
    assert preflight._compiled_architecture_supports(['sm_invalid','sm_120'],(12,0))


@pytest.mark.parametrize('reason', ['CUDA_DEVICE_NOT_FOUND',None,'unknown'])
def test_cuda_result_without_device_is_bounded_failure(reason):
    report = cuda_report(device_available=False,result='failed',reason_code=reason)
    if reason != 'CUDA_DEVICE_NOT_FOUND':
        with pytest.raises(ValueError):
            preflight.evaluate_cuda_preflight(gpu_result(),report)
        return
    result = preflight.evaluate_cuda_preflight(gpu_result(),report)
    assert result['result']=='failed' and result['device'] is None
    assert result['reason_code']=='CUDA_DEVICE_NOT_FOUND'


def test_preflight_receipt_requires_query_for_cuda_and_marks_unavailable_cpu_failed():
    args={'storage_root_display':'test','started_at':'2026-10-01T12:00:00.000Z',
          'finished_at':'2026-10-01T12:00:00.000Z','python_version':'3.12.3'}
    with pytest.raises(ValueError,match='GPU query'):
        preflight.build_preflight_receipt(selection('wsl-cuda'),child_execution(cuda_report()),storage_probe(),**args)
    report=passing_cpu_report(); report['device_available']=False
    receipt=preflight.build_preflight_receipt(selection(),child_execution(report),storage_probe(),**args)
    assert receipt['status']=='fail'
    assert any(row['status']=='fail' and 'BACKEND_MISSING' in row['observed'] for row in receipt['tests'])
