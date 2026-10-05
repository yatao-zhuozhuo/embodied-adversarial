"""Start/restart this baseline's loopback MCP service as a detached process."""
import argparse
import json
from pathlib import Path
import socket
import subprocess
import time
import psutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    python = str(root / "baseline/.runtime/coordinator/bin/python")
    owned = []
    for process in psutil.process_iter(["cmdline"]):
        cmd = process.info["cmdline"] or []
        if cmd and cmd[0] == python and "sim.mcp_server" in cmd and "8200" in cmd:
            owned.append(process)
    if owned and not args.restart:
        print(json.dumps({"already_running": [p.pid for p in owned]}))
        return
    for process in owned:
        process.terminate()
    _, alive = psutil.wait_procs(owned, timeout=15)
    if alive:
        raise RuntimeError("Previous service did not stop cleanly")
    log = root / "baseline/logs/simulator.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as output:
        child = subprocess.Popen(["bash", str(root / "baseline/scripts/serve_simulator.sh")],
                                 cwd=root, stdin=subprocess.DEVNULL, stdout=output,
                                 stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(100):
        if child.poll() is not None:
            raise RuntimeError(f"Simulator exited; inspect {log}")
        try:
            with socket.create_connection(("127.0.0.1", 8200), timeout=1):
                record = {"pid": child.pid, "url": "http://127.0.0.1:8200/mcp"}
                (root / "baseline/.runtime/simulator.json").write_text(json.dumps(record) + "\n")
                print(json.dumps(record))
                return
        except OSError:
            time.sleep(0.2)
    raise RuntimeError("Simulator did not start within 20 seconds")


if __name__ == "__main__":
    main()
