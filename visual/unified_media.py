"""Extract one saved demonstration frame on CPU; never imports simulators."""
import argparse
import json
import subprocess
from pathlib import Path

from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--index', required=True, type=int)
parser.add_argument('--camera', required=True)
parser.add_argument('--kind', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
p = Path(args.dataset)
if args.kind == 'libero_demo_state':
    import h5py
    camera = {'agentview': 'agentview_rgb', 'wrist': 'eye_in_hand_rgb'}[args.camera]
    with h5py.File(p, 'r') as f:
        frame = f[f'data/demo_{args.index}/obs/{camera}'][0]
    Image.fromarray(frame).save(args.output)
elif args.kind == 'robocasa_demo_state':
    camera = {'agentview_left': 'robot0_agentview_left', 'agentview_right': 'robot0_agentview_right', 'wrist': 'robot0_eye_in_hand'}[args.camera]
    info = json.loads((p / 'meta/info.json').read_text())
    relative = info['video_path'].format(episode_chunk=args.index // info['chunks_size'],
        episode_index=args.index, video_key='observation.images.' + camera)
    video = (p / relative).resolve()
    if not video.is_relative_to(p.resolve()):
        raise ValueError('Video path outside dataset')
    subprocess.run(['ffmpeg', '-nostdin', '-loglevel', 'error', '-i', str(video),
                    '-frames:v', '1', '-y', args.output], check=True, timeout=12)
else:
    raise ValueError('Unknown cached media kind')
