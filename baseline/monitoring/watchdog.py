"""Hourly recovery for the fixed XML campaign; never retries normal task failures."""
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = ROOT / 'baseline/results/inspire-20261005-xml-final'
FORMAL = CAMPAIGN / 'formal'
MONITOR = CAMPAIGN / 'monitoring'
PYTHON = ROOT / 'baseline/.runtime/coordinator/bin/python'
MODELS = {'qwen35_4b': (0, 8103, 8200), 'qwen35_9b': (2, 8102, 8202), 'qwen38_27b': (1, 8101, 8201)}


def read(path):
    return json.loads(path.read_text())


def processes(*tokens):
    found = []
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            argv = (proc / 'cmdline').read_bytes().decode().split('\0')
            if all(t in argv for t in tokens) and (proc / 'cwd').resolve() == ROOT:
                found.append(int(proc.name))
        except (OSError, UnicodeError):
            pass
    return found


def listening(port):
    try:
        with socket.create_connection(('127.0.0.1', port), timeout=3):
            return True
    except OSError:
        return False


def ready(port):
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(f'http://127.0.0.1:{port}/health', timeout=5) as response:
            return response.status == 200
    except Exception:
        return False


def launch(name, argv, **env):
    with (MONITOR / f'{name}.log').open('a') as log:
        child = subprocess.Popen(argv, cwd=ROOT, env={**os.environ, **env},
                                 stdin=subprocess.DEVNULL, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
    return child.pid


def check_model(model, gpu, api, sim):
    rows = read(ROOT / 'baseline/manifests/all_40.json')['episodes']
    finals = [FORMAL / model / row['episode_id'] / 'final.json' for row in rows]
    good = [p for p in finals if p.exists() and read(p)['status'] != 'infrastructure_error']
    result = {'completed': len(good), 'planned': len(rows),
              'success': sum(bool(read(p).get('success')) for p in good), 'actions': []}
    if len(good) == len(rows):
        result['state'] = 'complete'
        return result
    # The evaluator owns this same lock for its entire lifetime. Never alter live runs.
    with (FORMAL / f'.{model}.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            result['state'] = 'running'
            progress = list((FORMAL / model).glob('*/attempt-*/progress.json'))
            result['last_progress_age_seconds'] = round(time.time() - max(p.stat().st_mtime for p in progress)) if progress else None
            return result
        if processes('baseline/scripts/launch_model.py', model, str(FORMAL)):
            result['state'] = 'starting'
            return result
        # Only start missing services. Existing unhealthy processes are reported, not duplicated.
        if not ready(api):
            if not processes('vllm.entrypoints.openai.api_server', str(api)) and not listening(api):
                pid = launch(f'model-{model}', ['bash', 'baseline/scripts/serve_vllm.sh', model], CUDA_VISIBLE_DEVICES=str(gpu))
                result['actions'].append({'start_model_service': pid})
            for _ in range(120):
                if ready(api):
                    break
                time.sleep(5)
            else:
                result['state'] = 'waiting_for_model_service'
                return result
        if not listening(sim):
            if not processes('sim.mcp_server', str(sim)):
                pid = launch(f'simulator-{model}', ['bash', 'baseline/scripts/serve_simulator.sh'],
                             BASELINE_SIM_PORT=str(sim), OPENETA_WORKER_GPUS='3', OPENETA_WORKER_POOL_MAX='1')
                result['actions'].append({'start_simulator': pid})
            for _ in range(30):
                if listening(sim):
                    break
                time.sleep(2)
            else:
                result['state'] = 'waiting_for_simulator'
                return result
        # Keep failed infrastructure evidence separate; valid task failures are never retried.
        for final in finals:
            if final.exists() and read(final)['status'] == 'infrastructure_error':
                archive = MONITOR / 'interrupted' / model / str(time.time_ns())
                archive.mkdir(parents=True)
                final.parent.rename(archive / final.parent.name)
                result['actions'].append({'archived_infrastructure_failure': str(archive)})
    # Release the evaluator lock before spawning; the watchdog itself has a separate lock.
    pid = launch(f'evaluator-{model}', [str(PYTHON), 'baseline/scripts/launch_model.py',
                 '--model', model, '--sim-url', f'http://127.0.0.1:{sim}/mcp', '--root', str(FORMAL)])
    result.update(state='restarted', pid=pid)
    return result


def main():
    os.chdir(ROOT)
    MONITOR.mkdir(parents=True, exist_ok=True)
    with (MONITOR / 'watchdog.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        report = {'checked_at': time.time(), 'models': {}}
        for model, args in MODELS.items():
            try:
                report['models'][model] = check_model(model, *args)
            except Exception as exc:
                report['models'][model] = {'state': 'check_error', 'error': f'{type(exc).__name__}: {exc}'}
        report['complete'] = all(r['state'] == 'complete' for r in report['models'].values())
        report['finished_at'] = time.time()
        tmp = MONITOR / 'latest.tmp'
        tmp.write_text(json.dumps(report, indent=2) + '\n')
        tmp.replace(MONITOR / 'latest.json')
        with (MONITOR / 'history.jsonl').open('a') as log:
            log.write(json.dumps(report) + '\n')
        print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
