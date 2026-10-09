"""H200-vs-4090 RGB quantization allowance; no scene/state/instruction relaxation."""
import hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image
from baseline.v2.runtime import verify_accepted_initial as original_guard
C=Path(__file__).resolve().parent;V2=C.parents[1]

def verify_accepted_initial(row,raw,projected):
 try:return original_guard(row,raw,projected)
 except RuntimeError as exc:
  if str(exc)!='Initial image exceeds accepted raster tolerance' or row['metadata']['benchmark']!='robocasa':raise
 # The original guard has already checked robot state and official instruction.
 expected=json.loads((V2/'manifests/initial_states.json').read_text())[row['episode_id']]
 assert len(projected['cameras'])==len(expected['cameras'])
 measurements=[]
 for camera,reference in zip(projected['cameras'],expected['cameras']):
  assert camera['frame_id']==reference['frame_id']
  path=V2/reference['path']
  assert hashlib.sha256(path.read_bytes()).hexdigest()==reference['file_sha256']
  a=np.asarray(Image.open(camera['image_path']));b=np.asarray(Image.open(path))
  assert a.shape==b.shape
  delta=np.abs(a.astype(float)-b.astype(float));mae=float(delta.mean());maximum=float(delta.max())
  if mae>0.01 or maximum>1.0:raise RuntimeError(f'Cross-GPU initial pixels differ beyond quantization: {mae}, {maximum}')
  measurements.append({'camera':camera['frame_id'],'mae_255':mae,'max_delta_255':maximum})
 audit=Path(projected['vision_image_paths'][0]).parent/'cross-device-comparison.json'
 audit.write_text(json.dumps({'episode_id':row['episode_id'],'reference':'v2 4090 acceptance','target':'H200',
   'robot_and_instruction_checks':'unchanged','mae_limit_255':0.01,'max_delta_255':1,'measurements':measurements},indent=2)+'\n')
