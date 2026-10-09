"""Keep root-container LoRA checkpoints owned by the run directory's user."""
import argparse
import json
import os
from pathlib import Path
import stat
import time


def repair(root):
    owner=root.stat()
    changed=[]
    for path in root.glob('formal/trainer/*/v*/checkpoint-*/adapter_model.safetensors'):
        if not (path.parent/'trainer_state.json').is_file():
            continue
        info=path.lstat()
        if not stat.S_ISREG(info.st_mode) or not path.resolve().is_relative_to(root):
            raise ValueError('Refusing to change ownership outside the run directory')
        if (info.st_uid,info.st_gid)!=(owner.st_uid,owner.st_gid):
            os.chown(path,owner.st_uid,owner.st_gid,follow_symlinks=False)
            changed.append(str(path))
    return changed


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--watch',action='store_true')
    a=p.parse_args();root=a.run_root.resolve()
    deadline=time.monotonic()+72*3600
    while True:
        for path in repair(root):
            print(json.dumps({'owner_repaired':path}),flush=True)
        state=root/'curriculum_status.json'
        done=state.exists() and json.loads(state.read_text()).get('status') in ('complete','failed','needs_reward_signal')
        if not a.watch or done or time.monotonic()>deadline or any((root/'coordination').glob('failed-*.json')):
            break
        time.sleep(5)


if __name__=='__main__':main()
