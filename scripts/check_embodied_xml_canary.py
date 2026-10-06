"""Fail closed on real-model XML validation before starting a new run."""
import json
from pathlib import Path
import sys


def check(root: Path):
    for role in ("alice", "bob"):
        summary = json.loads((root / f"{role}_result.json").read_text())
        artifact = json.loads(Path(summary["artifact"]).read_text())
        if artifact.get("protocol_version") != "openeta.two_tool_xml.v1":
            raise ValueError(f"{role}: wrong protocol")
        if artifact.get("thinking_enabled") is not True:
            raise ValueError(f"{role}: thinking was not enabled")
        if len(artifact.get("steps", [])) < 2 or any(not s.get("valid_action") for s in artifact["steps"]):
            raise ValueError(f"{role}: XML canary contains invalid outputs")
        if not any(s.get("executed_control_steps", 0) > 0 for s in artifact["steps"]):
            raise ValueError(f"{role}: no real action executed")
        if artifact.get("final_info", {}).get("infrastructure_error"):
            raise ValueError(f"{role}: infrastructure error")
        print(f"{role}: {len(artifact['steps'])} valid XML turns, real MCP action verified")


if __name__ == "__main__":
    check(Path(sys.argv[1]))
