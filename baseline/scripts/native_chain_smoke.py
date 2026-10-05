"""Verify the actual MCP-to-isolated-worker path without an LLM."""
import asyncio
import argparse
import json
import datetime
from pathlib import Path

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", default="metaworld")
    parser.add_argument("--output")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    row = json.loads((root / f"baseline/manifests/{args.bench}_10.json").read_text())["episodes"][0]
    record = {"benchmark": args.bench, "passed": False}
    async with streamablehttp_client("http://127.0.0.1:8200/mcp") as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            catalog = await session.list_tools()
            record["tools"] = [t.name for t in catalog.tools]
            for tool in catalog.tools:
                if tool.name in {"create_env", "reset_env", "observe_env", "close_env", "step_env"}:
                    record.setdefault("schemas", {})[tool.name] = tool.inputSchema
            output = Path(args.output) if args.output else root / f"baseline/results/native-chain-{args.bench}.json"
            output.parent.mkdir(parents=True, exist_ok=True)
            sid = "baseline-preflight-" + args.bench + "-" + datetime.datetime.now().strftime("%H%M%S")
            handle = None

            async def call(name, values):
                response = await session.call_tool(name, {**values, "session_id": sid})
                if response.isError:
                    raise RuntimeError(str(response.content))
                data = response.structuredContent
                if data is None:
                    data = json.loads(next(c.text for c in response.content if c.type == "text"))
                if data.get("error"):
                    raise RuntimeError(data["error"])
                return data

            try:
                created = await call("create_env", {"env_id": row["env_id"], "seed": row["seed"],
                                                     "task": row["task"], "image_width": 512, "image_height": 512})
                record["create"] = created
                handle = created["handle"]
                record["reset"] = await call("reset_env", {"handle": handle, "seed": row["seed"], "settle": False})
                record["observe"] = await call("observe_env", {"handle": handle})
                record["step"] = await call("step_env", {"handle": handle,
                                                          "action": [0.0] * created["action_dim"], "num_steps": 1})
                native = record["step"].get("info", {}).get("native_success")
                assert isinstance(native, dict) and isinstance(native.get("success"), bool), "missing native checker"
                assert len(created["action_low"]) == created["action_dim"]
                record["passed"] = True
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                if handle:
                    record["close"] = await call("close_env", {"handle": handle})
                    if not record["close"].get("ok"):
                        record["passed"] = False
                output.write_text(json.dumps(record, indent=2) + "\n")
            print(json.dumps({"benchmark": args.bench, "passed": record["passed"],
                              "error": record.get("error"), "output": str(output)}))
            if not record["passed"]:
                raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
