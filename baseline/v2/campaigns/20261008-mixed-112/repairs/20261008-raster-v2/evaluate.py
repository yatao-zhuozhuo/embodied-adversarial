"""Run the unchanged frozen v2 evaluator with explicitly bound mixed checkpoints."""
import asyncio,os,sys
from pathlib import Path
C=Path(__file__).resolve().parent;ROOT=C.parents[3]
sys.path.insert(0,str(ROOT))
os.environ.update(BASELINE_SNAPSHOT=str(C/'snapshot'),NO_PROXY='*',no_proxy='*')
from backend import build_backend
from baseline.v2 import runtime
from initial_guard import verify_accepted_initial
runtime.build_backend=build_backend
runtime.verify_accepted_initial=verify_accepted_initial
if __name__=='__main__':asyncio.run(runtime.main())
