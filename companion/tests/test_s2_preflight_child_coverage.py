"""Bounded backend-probe failure behavior using explicit device doubles.

These tests do not establish CUDA or Windows runtime support.
"""
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from llm_foundations_companion import preflight_main as child


def modules(monkeypatch, *, cuda=False, missing=None, wrong=None, cpu_failure=False):
    torch = SimpleNamespace(__version__='2.8.0+cu128' if cuda else '2.8.0+cpu', float32='float32')
    def allocate(*args, **kwargs):
        if cpu_failure:
            raise RuntimeError('private driver error')
        return SimpleNamespace(zero_=lambda: None)
    torch.empty = allocate
    backend = {'torch': torch, **{name: SimpleNamespace(__version__=version)
        for name, version in child.COMMON_PINS.items()}}
    if wrong:
        backend[wrong].__version__ = '0.0.0'
    def load(name):
        if name == missing:
            raise ImportError('private file path')
        return backend[name]
    monkeypatch.setattr(child, '_load_backend', load)
    return torch


def test_backend_loader_accepts_only_the_five_fixed_names(monkeypatch):
    names = tuple(child._empty_versions())
    assert names == ('torch', 'transformers', 'peft', 'accelerate', 'safetensors')
    fixed = {}
    for name in names:
        module = ModuleType(name)
        monkeypatch.setitem(sys.modules, name, module)
        fixed[name] = module
    monkeypatch.setitem(sys.modules, 'learner_controlled', ModuleType('learner_controlled'))

    assert {name: child._load_backend(name) for name in names} == fixed
    with pytest.raises(ValueError, match='not fixed'):
        child._load_backend('learner_controlled')


@pytest.mark.parametrize('profile', ['wsl-cpu', 'win-cpu'])
@pytest.mark.parametrize('cpu_failure', [False, True])
def test_cpu_child_checks_all_pins_then_reports_allocation_failure(monkeypatch, profile, cpu_failure):
    modules(monkeypatch, cpu_failure=cpu_failure)
    report = child.build_child_report(profile)
    assert report['backend_status'] == 'passed'
    assert report['device_available'] is not cpu_failure
    assert report['cuda'] is None
    assert 'private' not in report['error_text']


@pytest.mark.parametrize('cuda', [False, True])
@pytest.mark.parametrize('problem', ['missing', 'wrong'])
def test_backend_import_and_version_failures_are_bounded(monkeypatch, cuda, problem):
    modules(monkeypatch, cuda=cuda, **{problem: 'peft'})
    report = child.build_child_report('wsl-cuda' if cuda else 'wsl-cpu')
    assert report['backend_status'] == ('missing' if problem == 'missing' else 'version_mismatch')
    assert not report['device_available'] and 'peft' in report['error_text']
    assert 'private' not in report['error_text']
    if cuda:
        assert report['cuda']['reason_code'] == 'CUDA_DEVICE_NOT_FOUND'


def device(**changes):
    values = dict(is_available=lambda: True, device_count=lambda: 1,
        get_device_name=lambda index: 'Test GPU', get_device_capability=lambda index: (7, 5),
        get_arch_list=lambda: ['compute_75', 'sm_7', 'sm_9999', 'sm_75'],
        get_device_properties=lambda index: SimpleNamespace(total_memory=4096*1024*1024),
        mem_get_info=lambda index: (3072*1024*1024, 4096*1024*1024))
    values.update(changes)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(('changes', 'reason'), [
    ({'is_available': lambda: False}, 'CUDA_DEVICE_NOT_FOUND'),
    ({'device_count': lambda: 0}, 'CUDA_DEVICE_NOT_FOUND'),
    ({'get_device_capability': lambda index: (7, 4)}, 'CUDA_CAPABILITY_UNSUPPORTED'),
    ({'get_arch_list': lambda: ['compute_75', 'sm_7', 'sm_9999', 'sm_90']}, 'CUDA_ARCH_NOT_IN_BUILD'),
    ({'get_device_properties': lambda index: SimpleNamespace(total_memory=3583*1024*1024)}, 'CUDA_MEMORY_INSUFFICIENT'),
    ({'mem_get_info': lambda index: (2047*1024*1024, 4096*1024*1024)}, 'CUDA_MEMORY_INSUFFICIENT'),
])
def test_cuda_metadata_boundaries_reject_before_numeric_probe(monkeypatch, changes, reason):
    torch = modules(monkeypatch, cuda=True)
    torch.cuda = device(**changes)
    report = child.build_child_report('wsl-cuda')
    assert report['backend_status'] == 'passed' and not report['device_available']
    assert report['cuda']['reason_code'] == reason


def test_cuda_metadata_exception_is_bounded(monkeypatch):
    torch = modules(monkeypatch, cuda=True)
    def broken(index):
        raise RuntimeError('private driver path')
    torch.cuda = device(get_device_name=broken)
    report = child.build_child_report('wsl-cuda')
    assert report['cuda']['reason_code'] == 'CUDA_DEVICE_NOT_FOUND'
    assert 'private' not in report['error_text']


def test_numeric_failure_restores_original_tf32_setting(monkeypatch):
    torch = modules(monkeypatch, cuda=True)
    torch.cuda = device()
    matmul = SimpleNamespace(allow_tf32=True)
    torch.backends = SimpleNamespace(cuda=SimpleNamespace(matmul=matmul))
    def broken(*args, **kwargs):
        assert matmul.allow_tf32 is False
        raise RuntimeError('private allocation failure')
    torch.arange = broken
    report = child.build_child_report('wsl-cuda')
    assert report['cuda']['reason_code'] == 'CUDA_NUMERIC_CHECK_FAILED'
    assert matmul.allow_tf32 is True and 'private' not in report['error_text']


def test_child_main_fallback_is_closed_json_without_exception_details(monkeypatch, capsys):
    def broken():
        raise RuntimeError('private environment')
    monkeypatch.setattr(child, 'build_child_report', broken)
    assert child.main() == 0
    result = capsys.readouterr()
    report = json.loads(result.out)
    assert not result.err and report['backend_status'] == 'missing'
    assert report['versions'] == {name: None for name in ('torch', 'transformers', 'peft', 'accelerate', 'safetensors')}
    assert 'private' not in report['error_text']


def test_child_main_emits_one_json_record_and_bounds_version(monkeypatch, capsys):
    modules(monkeypatch)
    monkeypatch.setenv('LLM_FOUNDATIONS_PROFILE', 'wsl-cpu')
    assert child.main() == 0
    assert json.loads(capsys.readouterr().out)['device_available'] is True
    assert child._version(SimpleNamespace(__version__='x'*100)) == 'x'*80
    assert child._version(SimpleNamespace()) is None
