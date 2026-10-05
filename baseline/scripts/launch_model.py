"""Load the existing DeepSeek credential into this process, then run evaluation."""
import asyncio
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if '--model' in sys.argv and sys.argv[sys.argv.index('--model')+1] == 'deepseek' and not os.environ.get('DEEPSEEK_API_KEY') and not os.environ.get('BASELINE_DEEPSEEK_RELAY_DIR'):
    p = Path(os.environ.get('BASELINE_DEEPSEEK_ENV_FILE', '/mnt/shared-storage-user/self-evolve/auto-train-pipline/runs/_evaluation_api_bridge/deepseek.env'))
    for line in p.read_text().splitlines():
        if line.startswith('DEEPSEEK_API_KEY='):
            os.environ['DEEPSEEK_API_KEY'] = line.split('=', 1)[1].strip().strip('"\'')
from baseline.runtime.run import main
asyncio.run(main())
