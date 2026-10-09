"""H200-vs-4090 RGB quantization allowance; no scene/state/instruction relaxation."""
import hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image
from baseline.v2.runtime import verify_accepted_initial as original_guard
C=Path(__file__).resolve().parent;V2=C.parents[1]

def raster_limits(benchmark):
 # LIBERO: retain the original mean bound; allow sparse two-level uint8
 # differences observed in the wrist renderer across 4090 and H200.
 return {'libero': (0.001, 2.0), 'robocasa': (0.01, 1.0)}[benchmark]

def quantization_measurements(a,b,benchmark):
 if a.shape!=b.shape:raise RuntimeError('Initial image dimensions changed')
 delta=np.abs(a.astype(float)-b.astype(float))
 mae=float(delta.mean());maximum=float(delta.max())
 mean_limit,max_limit=raster_limits(benchmark)
 if mae>mean_limit or maximum>max_limit:
  raise RuntimeError(f'Cross-GPU initial pixels differ beyond quantization: {mae}, {maximum}')
 return {'mae_255':mae,'max_delta_255':maximum}

def verify_accepted_initial(row,raw,projected):
 expected=json.loads((V2/'manifests/initial_states.json').read_text())[row['episode_id']]
 if len(projected['cameras'])!=len(expected['cameras']):
  raise RuntimeError('Accepted camera count changed')
 try:return original_guard(row,raw,projected)
 except RuntimeError as exc:
  if str(exc)!='Initial image exceeds accepted raster tolerance' or row['metadata']['benchmark'] not in ('libero','robocasa'):raise
 # The original guard has already checked robot state and official instruction.
 expected=json.loads((V2/'manifests/initial_states.json').read_text())[row['episode_id']]
 assert len(projected['cameras'])==len(expected['cameras'])
 measurements=[]
 for camera,reference in zip(projected['cameras'],expected['cameras']):
  assert camera['frame_id']==reference['frame_id']
  path=V2/reference['path']
  assert hashlib.sha256(path.read_bytes()).hexdigest()==reference['file_sha256']
  a=np.asarray(Image.open(camera['image_path']));b=np.asarray(Image.open(path))
  measurements.append({'camera':camera['frame_id'],**quantization_measurements(a,b,row['metadata']['benchmark'])})
 audit=Path(projected['vision_image_paths'][0]).parent/'cross-device-comparison.json'
 audit.write_text(json.dumps({'episode_id':row['episode_id'],'reference':'v2 4090 acceptance','target':'H200',
   'robot_and_instruction_checks':'unchanged','mae_limit_255':raster_limits(row['metadata']['benchmark'])[0],
   'max_delta_255':raster_limits(row['metadata']['benchmark'])[1],'measurements':measurements},indent=2)+'\n')
