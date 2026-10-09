#!/usr/bin/env python3
"""Read-only browser for successful two-tool SFT episodes; default port 8770."""

from __future__ import annotations

import argparse
import json
import re
import os
from pathlib import Path

from starlette.applications import Starlette
from starlette.responses import FileResponse, HTMLResponse, JSONResponse
from starlette.routing import Route
import uvicorn

DATA_ROOT = Path(__file__).resolve().parents[1]
DATASET = os.environ.get("OPENETA_TWO_TOOL_DATASET", "deepseek_xml_v1")
if DATASET not in {"qwen_xml_v1", "deepseek_xml_v1"}:
    raise ValueError("Unsupported two-tool viewer dataset")
ROOTS = {
    "metaworld": DATA_ROOT / DATASET / "metaworld",
    "robocasa": DATA_ROOT / DATASET / "robocasa",
    "libero": DATA_ROOT / DATASET / "libero",
    "maniskill3": DATA_ROOT / DATASET / "maniskill",
}
PAGE = Path(__file__).with_name("static") / "two_tool.html"
SAFE_NAME = re.compile(r"[A-Za-z0-9_.-]+\Z")
SAFE_ATTEMPT = re.compile(r"attempt-[0-9]{4}\Z")
SAFE_FRAME = re.compile(r"[0-9]{4}-[A-Za-z0-9_.-]+\.png\Z")


def _episode_path(bench: str, task: str, attempt: str) -> Path | None:
    root = ROOTS.get(bench)
    if root is None or not SAFE_NAME.fullmatch(task) or not SAFE_ATTEMPT.fullmatch(attempt):
        return None
    path = root / task / attempt / "episode.json"
    if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        return None
    return path


def page(_request):
    return HTMLResponse(PAGE.read_text(encoding="utf-8"))


def episodes(request):
    bench_filter = request.query_params.get("bench")
    items = []
    for bench, root in ROOTS.items():
        if bench_filter and bench_filter != bench:
            continue
        if not root.is_dir():
            continue
        for path in root.glob("*/attempt-*/episode.json"):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue  # an active collector may not have finished writing
            if (record.get("success") is not True
                    or record.get("stop_reason") != "official_success"
                    or (record.get("schema_version"), record.get("action_source")) not in {
                        ("openeta.qwen_live_two_tool_xml.v1", "qwen_api"),
                        ("openeta.deepseek_live_two_tool_xml.v1", "deepseek_api")}):
                continue
            items.append({
                "bench": bench, "task_id": path.parent.parent.name,
                "attempt": path.parent.name, "task": record.get("task"),
                "turns": len(record.get("turns") or []),
                "action_source": record.get("action_source"),
                "modified": path.stat().st_mtime,
            })
    items.sort(key=lambda row: row["modified"], reverse=True)
    return JSONResponse({"total": len(items), "items": items})


def episode(request):
    p = request.path_params
    path = _episode_path(p["bench"], p["task"], p["attempt"])
    if path is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    record = json.loads(path.read_text(encoding="utf-8"))
    return JSONResponse(record)


def frame(request):
    p = request.path_params
    path = _episode_path(p["bench"], p["task"], p["attempt"])
    name = p["name"]
    if path is None or not SAFE_FRAME.fullmatch(name):
        return JSONResponse({"error": "not found"}, status_code=404)
    image = path.parent / "frames" / name
    if not image.is_file() or not image.resolve().is_relative_to(path.parent.resolve()):
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(image, media_type="image/png")


app = Starlette(routes=[
    Route("/", page),
    Route("/api/episodes", episodes),
    Route("/api/episode/{bench}/{task}/{attempt}", episode),
    Route("/api/frame/{bench}/{task}/{attempt}/{name}", frame),
])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
