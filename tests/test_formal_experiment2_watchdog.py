import importlib.util
import json
from pathlib import Path
from unittest.mock import Mock

SPEC = importlib.util.spec_from_file_location('watch', Path(__file__).resolve().parents[1] / 'scripts/watch_formal_experiment2.py')
w = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(w)


def setup(monkeypatch, tmp_path):
    monkeypatch.setattr(w, 'RUN', tmp_path)
    monkeypatch.setattr(w, 'processes', lambda: ([], []))
    monkeypatch.setattr(w.time, 'sleep', lambda _: None)
    popen = Mock(return_value=Mock(pid=321, poll=lambda: None))
    monkeypatch.setattr(w.subprocess, 'Popen', popen)
    monkeypatch.setattr(w.subprocess, 'run', lambda *a, **kw: Mock(stdout=''))
    return popen


def test_running_never_restarts(monkeypatch, tmp_path):
    start = setup(monkeypatch, tmp_path)
    monkeypatch.setattr(w, 'processes', lambda: ([Mock(pid=123)], [Mock(pid=123)]))
    assert w.check() is False
    start.assert_not_called()


def test_stopped_restarts_append_log(monkeypatch, tmp_path):
    start = setup(monkeypatch, tmp_path)
    (tmp_path / 'train.log').write_text('original\n')
    w.check()
    start.assert_called_once()
    assert start.call_args.kwargs['env']['RUN_ROOT'] == str(tmp_path)
    assert (tmp_path / 'train.log').read_text().startswith('original\n')
    assert (tmp_path / 'launcher.pid').read_text() == '321\n'


def test_busy_gpu_defers(monkeypatch, tmp_path):
    start = setup(monkeypatch, tmp_path)
    monkeypatch.setattr(w.subprocess, 'run', lambda *a, **kw: Mock(stdout='987'))
    w.check()
    start.assert_not_called()


def test_pause_and_completion(monkeypatch, tmp_path):
    start = setup(monkeypatch, tmp_path)
    (tmp_path / 'watchdog.pause').touch()
    w.check()
    start.assert_not_called()
    (tmp_path / 'watchdog.pause').unlink()
    for i in range(5):
        d = tmp_path / f'round-{i:04d}'
        d.mkdir()
        (d / 'adapter_model.safetensors').touch()
        (d / 'summary.json').write_text(json.dumps({'checkpoints': {'alice': str(d), 'bob': str(d)}}))
    assert w.check() is True
    start.assert_not_called()
