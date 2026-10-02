#!/usr/bin/env python3
"""ManiSkill SFT 轨迹查看器服务 (只读, 默认仅监听 127.0.0.1)。

用法:
  python3 server.py [--root SFT_data] [--port 8765] [--reindex]

不连接/控制仿真器, 不提供任何写接口; 媒体通过不透明 id 访问,
真实路径解析后必须落在 --root 之内。
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import threading
import time
from pathlib import Path

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import (FileResponse, JSONResponse, PlainTextResponse)
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

import indexer

FRONTED_DIR = Path(__file__).resolve().parent
ROOT = FRONTED_DIR.parent.resolve()
DB_PATH = ROOT / "viewer" / "index.sqlite"
THUMB_DIR = ROOT / "viewer" / "thumbs"

_reindex_lock = threading.Lock()
_reindex_state = {"running": False, "log": [], "finished_at": None}


def db() -> sqlite3.Connection:
    con = sqlite3.connect(str(DB_PATH))
    con.row_factory = sqlite3.Row
    return con


def rows(cur) -> list:
    return [dict(r) for r in cur.fetchall()]


def _json_loads(s, default=None):
    if s is None:
        return default
    try:
        return json.loads(s)
    except (TypeError, json.JSONDecodeError):
        return default


def _safe_media_path(rel_path: str) -> Path | None:
    """解析真实路径并拒绝目录逃逸。"""
    p = (ROOT / rel_path).resolve()
    try:
        p.relative_to(ROOT)
    except ValueError:
        return None
    return p if p.is_file() else None


# ---------------------------------------------------------------- summary

def api_summary(request: Request):
    con = db()
    catalog = _json_loads(indexer._meta_get(con, "task_catalog"), [])
    quotas = {t.get("task_slug"): t.get("quota") for t in catalog}
    teachers = {}
    for row in con.execute(
            "SELECT teacher, status, task_slug, reject_reasons, updated_at_s"
            " FROM episodes"):
        t = teachers.setdefault(row["teacher"], {
            "accepted": 0, "rejected": 0, "other": 0, "total": 0,
            "official_success_unaccepted": 0,
            "latest_s": None, "tasks": {}, "failure_reasons": {},
        })
        t["total"] += 1
        t[row["status"]] = t.get(row["status"], 0) + 1
        ts = row["updated_at_s"]
        if ts and (t["latest_s"] is None or ts > t["latest_s"]):
            t["latest_s"] = ts
        if row["status"] == "accepted":
            slug = row["task_slug"] or "unknown"
            tk = t["tasks"].setdefault(slug, {"accepted": 0,
                                              "quota": quotas.get(slug)})
            tk["accepted"] += 1
        elif row["status"] == "rejected":
            for r in _json_loads(row["reject_reasons"], []):
                t["failure_reasons"][r] = t["failure_reasons"].get(r, 0) + 1
    for row in con.execute(
            "SELECT teacher, COUNT(*) c FROM episodes"
            " WHERE status!='accepted' AND official_success=1 GROUP BY teacher"):
        teachers.setdefault(row["teacher"], {}).update(
            official_success_unaccepted=row["c"])
    built = indexer._meta_get(con, "built_at_s")
    con.close()
    for t in teachers.values():
        t["quota_total"] = sum(q for q in quotas.values() if q)
    return JSONResponse({
        "generated_at": time.time(),
        "index_built_at": float(built) if built else None,
        "teachers": teachers,
        "task_catalog": catalog,
    })


# ---------------------------------------------------------------- episodes

SORTS = {
    "recent": "updated_at_s DESC",
    "turns_desc": "n_turns DESC",
    "turns_asc": "n_turns ASC",
    "duration_desc": "duration_s DESC",
    "seed_asc": "seed ASC",
    "tokens_desc": "total_tokens DESC",
}


def api_episodes(request: Request):
    q = request.query_params
    where, args = [], []
    if q.get("teacher"):
        where.append("teacher=?"); args.append(q["teacher"])
    if q.get("task"):
        where.append("task_slug=?"); args.append(q["task"])
    if q.get("status"):
        st = q["status"]
        if st == "failed":
            where.append("status!='accepted'")
        else:
            where.append("status=?"); args.append(st)
    if q.get("seed"):
        where.append("seed=?"); args.append(int(q["seed"]))
    if q.get("tool"):
        where.append("tools_used LIKE ?"); args.append(f'%"{q["tool"]}"%')
    if q.get("min_turns"):
        where.append("n_turns>=?"); args.append(int(q["min_turns"]))
    if q.get("max_turns"):
        where.append("n_turns<=?"); args.append(int(q["max_turns"]))
    if q.get("reason"):
        where.append("(reject_reasons LIKE ? OR failure_reason LIKE ? OR stop_reason LIKE ?)")
        args += [f'%{q["reason"]}%'] * 3
    if q.get("q"):
        where.append("(episode_id LIKE ? OR session_id LIKE ?)")
        args += [f'%{q["q"]}%'] * 2
    sql_where = ("WHERE " + " AND ".join(where)) if where else ""
    order = SORTS.get(q.get("sort", "recent"), SORTS["recent"])
    page = max(1, int(q.get("page", 1)))
    page_size = min(200, max(1, int(q.get("page_size", 30))))

    con = db()
    total = con.execute(f"SELECT COUNT(*) c FROM episodes {sql_where}", args).fetchone()["c"]
    items = rows(con.execute(
        f"SELECT session_id, episode_id, teacher, task_slug, seed, status,"
        f" accepted, official_success, reject_reasons, failure_reason,"
        f" stop_reason, n_turns, duration_s, elapsed_s, total_tokens,"
        f" updated_at_s, tools_used, first_frame, last_frame"
        f" FROM episodes {sql_where} ORDER BY {order}"
        f" LIMIT ? OFFSET ?", args + [page_size, (page - 1) * page_size]))
    con.close()
    for it in items:
        it["reject_reasons"] = _json_loads(it["reject_reasons"], [])
        it["tools_used"] = _json_loads(it["tools_used"], [])
    return JSONResponse({"total": total, "page": page, "page_size": page_size,
                         "items": items})


def _episode_row_or_404(con, session_id):
    row = con.execute("SELECT * FROM episodes WHERE session_id=?",
                      (session_id,)).fetchone()
    return row


def api_episode(request: Request):
    con = db()
    row = _episode_row_or_404(con, request.path_params["session_id"])
    if not row:
        con.close()
        return JSONResponse({"error": "not found"}, status_code=404)
    ep = dict(row)
    for k in ("reject_reasons", "tools_used", "warnings", "budget"):
        ep[k] = _json_loads(ep[k], [] if k != "budget" else {})
    ep.pop("fingerprint", None)
    con.close()
    return JSONResponse(ep)


def api_turns(request: Request):
    con = db()
    sid = request.path_params["session_id"]
    ep = _episode_row_or_404(con, sid)
    if not ep:
        con.close()
        return JSONResponse({"error": "not found"}, status_code=404)
    turns = rows(con.execute(
        "SELECT turn_index, seq, started_at_s, completed_at_s, duration_s,"
        " reward, terminated, truncated, official_reward, env_success,"
        " task_success, tool_names, key_events, frames_before, frames_after"
        " FROM turns WHERE session_id=? ORDER BY turn_index", (sid,)))
    con.close()
    for t in turns:
        for k in ("tool_names", "key_events", "frames_before", "frames_after"):
            t[k] = _json_loads(t[k], [] if k in ("tool_names", "key_events") else {})
    return JSONResponse({
        "session_id": sid,
        "status": ep["status"],
        "accepted": ep["accepted"],
        "official_success": ep["official_success"],
        "warnings": _json_loads(ep["warnings"], []),
        "snapshot_note": "帧为逐 turn 决策前后快照, 非连续物理运动录像",
        "turns": turns,
    })


# ------------------------------------------------------------ turn detail

def _derived_fingerprint(rollout: Path) -> str:
    return f"{indexer._fp(rollout / 'model_calls.jsonl')}|{indexer._fp(rollout / 'tool_calls.jsonl')}"


def _ensure_derived(con, ep_row) -> list:
    """按需解析 model_calls/tool_calls 并与 transitions 对齐, 结果缓存于 SQLite。"""
    sid = ep_row["session_id"]
    rollout = Path(ep_row["bundle"])
    fp = _derived_fingerprint(rollout)
    row = con.execute("SELECT fingerprint, join_warnings FROM episode_derived"
                      " WHERE session_id=?", (sid,)).fetchone()
    if row and row["fingerprint"] == fp:
        return _json_loads(row["join_warnings"], [])

    calls = []
    mc = rollout / "model_calls.jsonl"
    if mc.is_file():
        for line in mc.open("r", encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            dec = o.get("parsed_decision") or {}
            val = o.get("validation") or {}
            res = o.get("result") or {}
            calls.append((
                sid, o.get("seq"), int(bool(val.get("accepted"))),
                json.dumps(val.get("errors") or [], ensure_ascii=False),
                dec.get("kind"), dec.get("name"), dec.get("reasoning"),
                json.dumps(dec.get("parameters"), ensure_ascii=False),
                res.get("payload"),
                o.get("duration_s"), o.get("started_at_s"), o.get("completed_at_s"),
            ))
    events = []
    tc = rollout / "tool_calls.jsonl"
    if tc.is_file():
        for line in tc.open("r", encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            ev = o.get("event") or {}
            details = ev.get("details") or {}
            outputs = details.get("outputs") or {}
            events.append((
                sid, o.get("seq"), ev.get("phase"), ev.get("name"),
                json.dumps(ev.get("parameters"), ensure_ascii=False),
                ev.get("content"),
                outputs.get("stdout"),
                (None if details.get("operational_success") is None
                 else int(bool(details.get("operational_success")))),
                o.get("timestamp_s"),
            ))

    # --- 对齐: 已验收的主规划 tool_call 决策 <-> transitions ---
    join_warnings = []
    accepted_calls = [c for c in calls
                      if c[2] and c[4] == "tool_call" and c[5]]
    turn_rows = con.execute(
        "SELECT turn_index, tool_names FROM turns WHERE session_id=?"
        " ORDER BY turn_index", (sid,)).fetchall()
    turn_call_map = {}
    used = set()
    for tr in turn_rows:
        tnames = _json_loads(tr["tool_names"], [])
        if not tnames:
            continue
        want = tnames[0]
        # 按顺序找下一个同名已验收调用 (执行顺序对齐)
        for i, c in enumerate(accepted_calls):
            if i in used:
                continue
            if c[5] == want:
                turn_call_map[tr["turn_index"]] = c[1]
                used.add(i)
                break
        else:
            join_warnings.append(
                f"turn {tr['turn_index']} 的工具 {want} 未找到对应的已验收模型调用")
    n_rejected = sum(1 for c in calls if not c[2])
    if n_rejected:
        join_warnings.append(f"{n_rejected} 次模型调用被校验拒绝 (重试), 对齐已跳过这些调用")
    if not calls and mc.is_file() is False:
        join_warnings.append("缺少 model_calls.jsonl")

    with con:
        con.execute("DELETE FROM model_calls WHERE session_id=?", (sid,))
        con.executemany("INSERT INTO model_calls VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", calls)
        con.execute("DELETE FROM tool_events WHERE session_id=?", (sid,))
        con.executemany("INSERT INTO tool_events VALUES(?,?,?,?,?,?,?,?,?)", events)
        con.execute("INSERT OR REPLACE INTO episode_derived VALUES(?,?,?,?)",
                    (sid, fp, json.dumps(join_warnings, ensure_ascii=False),
                     json.dumps(turn_call_map)))
    return join_warnings


def api_turn_detail(request: Request):
    sid = request.path_params["session_id"]
    n = int(request.path_params["n"])
    con = db()
    ep = _episode_row_or_404(con, sid)
    if not ep:
        con.close()
        return JSONResponse({"error": "not found"}, status_code=404)
    join_warnings = _ensure_derived(con, ep)
    t = con.execute("SELECT * FROM turns WHERE session_id=? AND turn_index=?",
                    (sid, n)).fetchone()
    if not t:
        con.close()
        return JSONResponse({"error": "turn not found"}, status_code=404)
    turn = dict(t)
    for k in ("tool_names", "key_events", "frames_before", "frames_after",
              "objects_before", "objects_after", "robot_before", "robot_after"):
        turn[k] = _json_loads(turn[k])

    map_row = con.execute("SELECT turn_call_map FROM episode_derived"
                          " WHERE session_id=?", (sid,)).fetchone()
    call_map = _json_loads(map_row["turn_call_map"], {}) if map_row else {}
    call_seq = call_map.get(str(n)) or call_map.get(n)

    decision = validation = None
    if call_seq is not None:
        mc = con.execute("SELECT * FROM model_calls WHERE session_id=? AND seq=?",
                         (sid, call_seq)).fetchone()
        if mc:
            decision = {
                "seq": mc["seq"], "kind": mc["kind"], "name": mc["name"],
                "reasoning": mc["reasoning"],
                "parameters": _json_loads(mc["parameters"]),
                "duration_s": mc["duration_s"],
                "started_at_s": mc["started_at_s"],
            }
            validation = {"accepted": bool(mc["accepted"]),
                          "errors": _json_loads(mc["errors"], [])}

    # 工具事件: 归属到 turn 的执行时间窗口
    tool_events = rows(con.execute(
        "SELECT seq, phase, name, parameters, content, stdout,"
        " operational_success, timestamp_s FROM tool_events"
        " WHERE session_id=? AND timestamp_s BETWEEN ? AND ?"
        " ORDER BY seq",
        (sid, (turn["started_at_s"] or 0) - 0.001,
         (turn["completed_at_s"] or 9e18) + 0.001)))
    for e in tool_events:
        e["parameters"] = _json_loads(e["parameters"])

    # SFT 样本反查
    sft = None
    if call_seq is not None and ep["episode_id"]:
        s = con.execute(
            "SELECT sample_id, teacher FROM sft_samples"
            " WHERE episode_id=? AND seq=?",
            (ep["episode_id"], call_seq)).fetchone()
        if s:
            sft = {"sample_id": s["sample_id"], "teacher": s["teacher"]}

    con.close()
    return JSONResponse({
        "turn": turn,
        "decision": decision,
        "validation": validation,
        "tool_events": tool_events,
        "join_warnings": join_warnings,
        "sft_sample": sft,
        "receipt": {
            "reward": turn["reward"],
            "official_reward": bool(turn["official_reward"]),
            "env_success": bool(turn["env_success"]),
            "task_success": bool(turn["task_success"]),
            "terminated": bool(turn["terminated"]),
            "truncated": bool(turn["truncated"]),
        },
    })


def api_turn_context(request: Request):
    """审计视图: 该 turn 对应模型调用的完整 prompt 与原始输出 (截断保护)。"""
    sid = request.path_params["session_id"]
    n = int(request.path_params["n"])
    con = db()
    ep = _episode_row_or_404(con, sid)
    if not ep:
        con.close()
        return JSONResponse({"error": "not found"}, status_code=404)
    _ensure_derived(con, ep)
    map_row = con.execute("SELECT turn_call_map FROM episode_derived"
                          " WHERE session_id=?", (sid,)).fetchone()
    call_map = _json_loads(map_row["turn_call_map"], {}) if map_row else {}
    call_seq = call_map.get(str(n)) or call_map.get(n)
    if call_seq is None:
        con.close()
        return JSONResponse({"error": "no aligned model call"}, status_code=404)
    mc = con.execute("SELECT payload FROM model_calls WHERE session_id=? AND seq=?",
                     (sid, call_seq)).fetchone()
    con.close()
    limit = 200_000
    payload = mc["payload"] if mc else None
    return JSONResponse({
        "seq": call_seq,
        "payload": payload[:limit] if payload else None,
        "payload_truncated": bool(payload and len(payload) > limit),
    })


def api_turn_raw(request: Request):
    """审计视图: transitions.jsonl 中该 turn 的原始记录。"""
    sid = request.path_params["session_id"]
    n = int(request.path_params["n"])
    con = db()
    ep = _episode_row_or_404(con, sid)
    t = con.execute("SELECT line_offset FROM turns WHERE session_id=? AND turn_index=?",
                    (sid, n)).fetchone()
    con.close()
    if not ep or not t:
        return JSONResponse({"error": "not found"}, status_code=404)
    f = Path(ep["bundle"]) / "transitions.jsonl"
    real = _safe_media_path(os.path.relpath(f, ROOT))
    if real is None:
        return JSONResponse({"error": "transitions.jsonl unavailable"}, status_code=404)
    with real.open("r", encoding="utf-8") as fh:
        fh.seek(t["line_offset"])
        line = fh.readline()
    try:
        return JSONResponse(json.loads(line))
    except json.JSONDecodeError:
        return JSONResponse({"error": "corrupt line"}, status_code=500)


def api_sft_sample(request: Request):
    sample_id = request.path_params["sample_id"]
    con = db()
    s = con.execute("SELECT * FROM sft_samples WHERE sample_id=?",
                    (sample_id,)).fetchone()
    con.close()
    if not s:
        return JSONResponse({"error": "not found"}, status_code=404)
    f = ROOT / "exported" / s["teacher"] / "train.v1.jsonl"
    real = _safe_media_path(os.path.relpath(f, ROOT))
    if real is None:
        return JSONResponse({"error": "export file unavailable"}, status_code=404)
    with real.open("r", encoding="utf-8") as fh:
        fh.seek(s["offset"])
        line = fh.readline()
    try:
        o = json.loads(line)
    except json.JSONDecodeError:
        return JSONResponse({"error": "corrupt sample"}, status_code=500)
    # 图片路径不直接暴露文件内容, 仅显示路径文本
    return JSONResponse(o)


# ---------------------------------------------------------------- media

def api_media(request: Request):
    mid = request.path_params["media_id"]
    con = db()
    m = con.execute("SELECT rel_path FROM media WHERE media_id=?", (mid,)).fetchone()
    con.close()
    if not m:
        return JSONResponse({"error": "not found"}, status_code=404)
    real = _safe_media_path(m["rel_path"])
    if real is None:
        return JSONResponse({"error": "file missing"}, status_code=404)
    q = request.query_params
    if "thumb" in q:
        width = min(1024, max(16, int(q.get("w", 256))))
        thumb = THUMB_DIR / f"{mid}_{width}.webp"
        if not thumb.is_file():
            try:
                from PIL import Image
                THUMB_DIR.mkdir(parents=True, exist_ok=True)
                with Image.open(real) as im:
                    im = im.convert("RGB")
                    ratio = width / im.width
                    im = im.resize((width, max(1, int(im.height * ratio))))
                    im.save(thumb, "WEBP", quality=80)
            except Exception as e:
                return JSONResponse({"error": f"thumbnail failed: {e}"},
                                    status_code=500)
        return FileResponse(thumb, media_type="image/webp")
    return FileResponse(real, media_type="image/png")


# ---------------------------------------------------------------- reindex

def _run_reindex():
    with _reindex_lock:
        _reindex_state.update(running=True, log=[], finished_at=None)
        try:
            indexer.build_index(ROOT, DB_PATH,
                                progress=lambda m: _reindex_state["log"].append(m))
        except Exception as e:
            _reindex_state["log"].append(f"[error] {e}")
        _reindex_state.update(running=False, finished_at=time.time())


def api_reindex(request: Request):
    if _reindex_state["running"]:
        return JSONResponse({"started": False, "running": True})
    threading.Thread(target=_run_reindex, daemon=True).start()
    return JSONResponse({"started": True})


def api_reindex_status(request: Request):
    return JSONResponse(_reindex_state)


# ---------------------------------------------------------------- app

routes = [
    Route("/api/summary", api_summary),
    Route("/api/episodes", api_episodes),
    Route("/api/episodes/{session_id}", api_episode),
    Route("/api/episodes/{session_id}/turns", api_turns),
    Route("/api/episodes/{session_id}/turns/{n:int}", api_turn_detail),
    Route("/api/episodes/{session_id}/turns/{n:int}/context", api_turn_context),
    Route("/api/episodes/{session_id}/turns/{n:int}/raw", api_turn_raw),
    Route("/api/sft/{sample_id}", api_sft_sample),
    Route("/api/media/{media_id}", api_media),
    Route("/api/reindex", api_reindex, methods=["POST"]),
    Route("/api/reindex/status", api_reindex_status),
    Mount("/", StaticFiles(directory=str(FRONTED_DIR / "static"), html=True)),
]

app = Starlette(routes=routes)


def main():
    global ROOT, DB_PATH, THUMB_DIR
    p = argparse.ArgumentParser(description="SFT 轨迹查看器")
    p.add_argument("--root", default=str(ROOT), help="SFT_data 根目录")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--reindex", action="store_true", help="启动前重建索引")
    args = p.parse_args()
    ROOT = Path(args.root).resolve()
    DB_PATH = ROOT / "viewer" / "index.sqlite"
    THUMB_DIR = ROOT / "viewer" / "thumbs"
    if args.reindex or not DB_PATH.is_file():
        indexer.build_index(ROOT, DB_PATH)
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
