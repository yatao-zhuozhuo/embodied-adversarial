"""Start the explicitly assigned mixed checkpoint on its assigned physical GPUs."""
import json,os,sys,hashlib
from pathlib import Path
C=Path(__file__).resolve().parent;ROOT=C.parents[3]
e=json.loads((C/'models.json').read_text())[sys.argv[1]];p=Path(e['model_path'])
assert hashlib.sha256((p/'openeta_sft_provenance.json').read_bytes()).hexdigest()==e['provenance_sha256']
assert hashlib.sha256((p/'model.safetensors.index.json').read_bytes()).hexdigest()==e['index_sha256']
assert all((p/name).stat().st_size==size for name,size in e['shards'].items())
os.environ.update(CUDA_VISIBLE_DEVICES=','.join(map(str,e['gpus'])),NO_PROXY='*',no_proxy='*',
                  OMP_NUM_THREADS='4',MAX_JOBS='4',PYTHONPATH=str(ROOT/'baseline/.runtime/api-compat')+':'+str(ROOT))
os.environ['LD_LIBRARY_PATH']='/usr/local/nvidia/lib64:'+os.environ.get('LD_LIBRARY_PATH','')
python=str(ROOT/'baseline/.runtime/swift/bin/python')
argv=[python,'-m','vllm.entrypoints.openai.api_server','--model',str(p),'--served-model-name',e['model'],
      '--host','127.0.0.1','--port',str(e['port']),'--dtype','bfloat16','--tensor-parallel-size',str(e['tensor_parallel_size']),
      '--max-model-len','32768','--max-num-seqs','1','--gpu-memory-utilization','0.70',
      '--kv-cache-memory-bytes','4294967296','--mm-processor-cache-gb','0','--limit-mm-per-prompt','{"image":5,"video":0}',
      '--mm-processor-kwargs','{"size":{"shortest_edge":4096,"longest_edge":262144}}',
      '--gdn-prefill-backend','triton','--reasoning-parser','qwen3','--generation-config','vllm','--enforce-eager']
os.chdir(ROOT);os.execv(python,argv)
