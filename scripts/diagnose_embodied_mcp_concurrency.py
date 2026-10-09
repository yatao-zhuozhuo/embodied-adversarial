"""Exercise concurrent PickCube MCP clients without loading the policy."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json
import sys
import traceback

from adapter.maniskill_mcp import McpPickCubeAdapter


def probe(index: int, root: Path) -> dict:
    adapter = None
    phase = "create"
    try:
        adapter = McpPickCubeAdapter(camera_resolution=64)
        phase = "reset"
        adapter.reset(seed=31000 + index)
        phase = "capture"
        snapshot = adapter.capture_snapshot(snapshot_dir=str(root / "snapshots"), seed=31000 + index)
        phase = "restore"
        adapter.restore_snapshot(snapshot)
        phase = "task_state"
        state = adapter.task_state()
        return {"index": index, "ok": True, "phase": phase, "state": state}
    except BaseException as exc:
        return {"index": index, "ok": False, "phase": phase,
                "error": repr(exc), "traceback": traceback.format_exc()[-2000:]}
    finally:
        if adapter is not None:
            try:
                adapter.close()
            except Exception:
                pass


def main() -> int:
    root = Path(sys.argv[1])
    root.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(probe, index, root) for index in range(8)]
        for future in as_completed(futures):
            result = future.result()
            print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
