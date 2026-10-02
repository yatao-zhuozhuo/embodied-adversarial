#!/usr/bin/env python3
"""SFT 轨迹查看器索引构建器。

扫描 SFT_data/ 下的验收索引、导出报告与原始 rollout bundle,
把轨迹摘要、turn 时间轴、帧引用写入 SQLite (SFT_data/viewer/index.sqlite)。
索引可删除重建, 原始 JSONL 始终是事实来源。按文件 mtime 增量更新。
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS episodes (
  session_id TEXT PRIMARY KEY,
  teacher TEXT, episode_id TEXT, env_id TEXT, task_slug TEXT, seed INTEGER,
  status TEXT, accepted INTEGER, official_success INTEGER,
  reject_reasons TEXT, bundle TEXT, batch_id TEXT, teacher_model TEXT,
  task_text TEXT, budget TEXT, created_at_s REAL, updated_at_s REAL,
  elapsed_s REAL, duration_s REAL, n_turns INTEGER, total_tokens INTEGER,
  tool_call_count INTEGER, stop_reason TEXT, failure_reason TEXT,
  tools_used TEXT, first_frame TEXT, last_frame TEXT, warnings TEXT,
  fingerprint TEXT
);
CREATE INDEX IF NOT EXISTS idx_ep_teacher ON episodes(teacher);
CREATE INDEX IF NOT EXISTS idx_ep_task ON episodes(task_slug);
CREATE INDEX IF NOT EXISTS idx_ep_status ON episodes(status);
CREATE INDEX IF NOT EXISTS idx_ep_updated ON episodes(updated_at_s);
CREATE TABLE IF NOT EXISTS turns (
  session_id TEXT, turn_index INTEGER, seq INTEGER,
  started_at_s REAL, completed_at_s REAL, duration_s REAL,
  reward REAL, terminated INTEGER, truncated INTEGER,
  official_reward INTEGER, env_success INTEGER, task_success INTEGER,
  tool_names TEXT, key_events TEXT,
  frames_before TEXT, frames_after TEXT,
  objects_before TEXT, objects_after TEXT,
  robot_before TEXT, robot_after TEXT,
  line_offset INTEGER,
  PRIMARY KEY (session_id, turn_index)
);
CREATE TABLE IF NOT EXISTS media (
  media_id TEXT PRIMARY KEY, session_id TEXT, rel_path TEXT,
  frame_id TEXT, kind TEXT
);
CREATE INDEX IF NOT EXISTS idx_media_session ON media(session_id);
CREATE TABLE IF NOT EXISTS model_calls (
  session_id TEXT, seq INTEGER, accepted INTEGER, errors TEXT,
  kind TEXT, name TEXT, reasoning TEXT, parameters TEXT, payload TEXT,
  duration_s REAL, started_at_s REAL, completed_at_s REAL,
  PRIMARY KEY (session_id, seq)
);
CREATE TABLE IF NOT EXISTS tool_events (
  session_id TEXT, seq INTEGER, phase TEXT, name TEXT, parameters TEXT,
  content TEXT, stdout TEXT, operational_success INTEGER, timestamp_s REAL,
  PRIMARY KEY (session_id, seq, phase)
);
CREATE TABLE IF NOT EXISTS episode_derived (
  session_id TEXT PRIMARY KEY, fingerprint TEXT, join_warnings TEXT,
  turn_call_map TEXT
);
CREATE TABLE IF NOT EXISTS sft_samples (
  sample_id TEXT PRIMARY KEY, teacher TEXT, episode_id TEXT,
  seq INTEGER, offset INTEGER
);
CREATE INDEX IF NOT EXISTS idx_sft_ep ON sft_samples(episode_id, seq);
"""

KEY_EVENT_TOOLS = ("ik_preview_check", "move_to", "gripper_control",
                   "observe", "python_exec", "save_memory")


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(SCHEMA)
    return con


def _fp(path: Path) -> str:
    try:
        st = path.stat()
        return f"{st.st_mtime_ns}:{st.st_size}"
    except OSError:
        return "missing"


def _meta_get(con, key):
    row = con.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row["value"] if row else None


def _meta_set(con, key, value):
    con.execute("INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)", (key, value))


def media_id_for(rel_path: str) -> str:
    return hashlib.sha1(rel_path.encode("utf-8")).hexdigest()[:24]


def _register_media(con, session_id, root: Path, abs_path, frame_id, kind):
    """把图像绝对路径登记为不透明 media id; 返回 id 或 None。"""
    if not abs_path:
        return None
    try:
        rel = os.path.relpath(abs_path, root)
    except ValueError:
        return None
    if rel.startswith(".."):
        return None
    mid = media_id_for(rel)
    con.execute(
        "INSERT OR IGNORE INTO media(media_id, session_id, rel_path, frame_id, kind)"
        " VALUES(?,?,?,?,?)",
        (mid, session_id, rel, frame_id, kind))
    return mid


def _frames_from_obs(con, session_id, root, obs):
    """从 observation.metadata.image_artifacts 提取 rgb 帧引用。"""
    out = {}
    md = (obs or {}).get("metadata") or {}
    for art in md.get("image_artifacts") or []:
        if art.get("kind") != "rgb":
            continue
        frame_id = art.get("frame_id") or "unknown"
        if frame_id in out:
            continue
        mid = _register_media(con, session_id, root, art.get("path"), frame_id, "rgb")
        if mid:
            out[frame_id] = mid
    return out


def _compact_objects(obs):
    objs = []
    for o in (obs or {}).get("objects") or []:
        objs.append({
            "name": o.get("name"),
            "position": o.get("position"),
            "orientation": o.get("orientation"),
        })
    return objs


def _compact_robot(obs):
    r = (obs or {}).get("robot") or {}
    return {
        "end_effector_pose": r.get("end_effector_pose"),
        "gripper_state": r.get("gripper_state"),
        "joint_positions": r.get("joint_positions"),
    }


def _key_events(tool_names, reward, terminated, truncated, task_success):
    ev = []
    for t in tool_names:
        if t in KEY_EVENT_TOOLS:
            ev.append(t)
        elif t and t not in ev:
            ev.append(t)
    if task_success:
        ev.append("official_success")
    if terminated and not task_success:
        ev.append("terminated")
    if truncated:
        ev.append("truncated")
    return ev


def load_accepted(root: Path):
    """accepted/<teacher>/episodes.v1.jsonl -> {session_id: row}"""
    out = {}
    acc_dir = root / "accepted"
    if not acc_dir.is_dir():
        return out
    for teacher_dir in sorted(acc_dir.iterdir()):
        f = teacher_dir / "episodes.v1.jsonl"
        if not f.is_file():
            continue
        for line in f.open("r", encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            if o.get("session_id"):
                out[o["session_id"]] = o
    return out


def load_rejected(root: Path):
    """reports/<teacher>.export.json -> {bundle_path: [reasons]}"""
    out = {}
    rep_dir = root / "reports"
    for f in sorted(rep_dir.glob("*.export.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        for item in d.get("rejected") or []:
            if item.get("bundle"):
                out[item["bundle"]] = item.get("reasons") or []
    return out


def load_task_catalog(root: Path):
    f = root / "configs" / "task_catalog.v1.json"
    if not f.is_file():
        return []
    try:
        return json.loads(f.read_text(encoding="utf-8")).get("tasks") or []
    except (json.JSONDecodeError, OSError):
        return []


def index_session(con, root: Path, teacher: str, rollout: Path,
                  accepted_map, rejected_map):
    session_id = rollout.parent.name
    manifest_p = rollout / "manifest.json"
    transitions_p = rollout / "transitions.jsonl"
    fp = f"{_fp(manifest_p)}|{_fp(transitions_p)}"
    old = con.execute("SELECT fingerprint FROM episodes WHERE session_id=?",
                      (session_id,)).fetchone()
    if old and old["fingerprint"] == fp:
        return False

    warnings = []
    manifest = {}
    if manifest_p.is_file():
        try:
            manifest = json.loads(manifest_p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            warnings.append(f"manifest 解析失败: {e}")
    else:
        warnings.append("缺少 manifest.json")

    md = manifest.get("metadata") or {}
    last_ep = (manifest.get("last_episode") or {}).get("metadata") or {}
    usage = last_ep.get("usage") or {}
    budget = {
        "max_turns": md.get("max_turns"),
        "max_tool_calls": md.get("max_tool_calls"),
        "max_total_tokens": md.get("max_total_tokens"),
        "timeout_s": md.get("timeout_s"),
    }

    # --- transitions ---
    turn_rows = []
    tools_used = set()
    if transitions_p.is_file():
        with transitions_p.open("r", encoding="utf-8") as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    t = json.loads(line)
                except json.JSONDecodeError:
                    warnings.append(f"transitions 第 {len(turn_rows)+1} 行解析失败")
                    continue
                obs = t.get("observation") or {}
                nobs = t.get("next_observation") or {}
                tnames = []
                for tc in ((t.get("action") or {}).get("command") or {}).get("tool_calls") or []:
                    if tc.get("name"):
                        tnames.append(tc["name"])
                        tools_used.add(tc["name"])
                info = t.get("info") or {}
                receipt = info.get("environment_receipt") or {}
                task_success = bool(receipt.get("task_success") or info.get("environment_success"))
                fb = _frames_from_obs(con, session_id, root, obs)
                fa = _frames_from_obs(con, session_id, root, nobs)
                if not fb and not fa:
                    warnings.append(f"turn {t.get('turn_index')} 无图像帧")
                turn_rows.append((
                    session_id, t.get("turn_index"), t.get("seq"),
                    t.get("started_at_s"), t.get("completed_at_s"), t.get("duration_s"),
                    t.get("reward"), int(bool(t.get("terminated"))),
                    int(bool(t.get("truncated"))),
                    int(bool(info.get("official_reward"))),
                    int(bool(info.get("environment_success"))),
                    int(task_success),
                    json.dumps(tnames, ensure_ascii=False),
                    json.dumps(_key_events(tnames, t.get("reward"),
                                           t.get("terminated"), t.get("truncated"),
                                           task_success), ensure_ascii=False),
                    json.dumps(fb), json.dumps(fa),
                    json.dumps(_compact_objects(obs)),
                    json.dumps(_compact_objects(nobs)),
                    json.dumps(_compact_robot(obs)),
                    json.dumps(_compact_robot(nobs)),
                    offset,
                ))
    else:
        warnings.append("缺少 transitions.jsonl")

    # --- 汇总 ---
    n_turns = len(turn_rows)
    duration_s = None
    official_success = 0
    if turn_rows:
        first, last = turn_rows[0], turn_rows[-1]
        if first[3] and last[4]:
            duration_s = round(last[4] - first[3], 3)
        official_success = int(bool(last[11]))

    accepted_row = accepted_map.get(session_id)
    bundle_str = str(rollout)
    if accepted_row:
        status, accepted, reject_reasons = "accepted", 1, []
    elif bundle_str in rejected_map:
        status, accepted, reject_reasons = "rejected", 0, rejected_map[bundle_str]
    else:
        status, accepted, reject_reasons = "other", 0, []
        warnings.append("不在验收索引也不在上次导出拒绝列表中 (未完成/未提交)")

    if status == "rejected" and official_success:
        warnings.append("官方回执成功但被导出拒绝 — 以验收索引为准, 不算成功数据")

    first_frame = last_frame = None
    if turn_rows:
        fb0 = json.loads(turn_rows[0][14])
        faN = json.loads(turn_rows[-1][15])
        first_frame = fb0.get("base_camera") or next(iter(fb0.values()), None)
        last_frame = faN.get("base_camera") or next(iter(faN.values()), None)

    episode_id = md.get("episode_id") or (accepted_row or {}).get("episode_id")
    con.execute("DELETE FROM turns WHERE session_id=?", (session_id,))
    con.executemany(
        "INSERT INTO turns VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        turn_rows)
    con.execute("INSERT OR REPLACE INTO episodes VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
        session_id, teacher,
        episode_id,
        md.get("env_id") or (accepted_row or {}).get("env_id"),
        md.get("task_slug") or (accepted_row or {}).get("task_slug"),
        md.get("seed") if md.get("seed") is not None else (accepted_row or {}).get("seed"),
        status, accepted, official_success,
        json.dumps(reject_reasons, ensure_ascii=False),
        bundle_str, md.get("batch_id"),
        md.get("teacher_model") or (accepted_row or {}).get("teacher_model"),
        manifest.get("task"), json.dumps(budget),
        manifest.get("created_at_s"), manifest.get("updated_at_s"),
        usage.get("elapsed_s"), duration_s, n_turns,
        usage.get("total_tokens"), usage.get("tool_call_count"),
        last_ep.get("stop_reason") if not isinstance(last_ep.get("stop_reason"), (dict, list)) else json.dumps(last_ep["stop_reason"], ensure_ascii=False),
        (lambda fr: fr if (fr is None or isinstance(fr, str)) else json.dumps(fr, ensure_ascii=False))(last_ep.get("failure_reason")),
        json.dumps(sorted(tools_used), ensure_ascii=False),
        first_frame, last_frame,
        json.dumps(warnings, ensure_ascii=False), fp,
    ))
    con.execute("DELETE FROM episode_derived WHERE session_id=?", (session_id,))
    return True


def index_sft_export(con, root: Path, teacher: str):
    """为 exported/<teacher>/train.v1.jsonl 建 sample_id -> 字节偏移索引。"""
    f = root / "exported" / teacher / "train.v1.jsonl"
    key = f"sft_fp:{teacher}"
    if not f.is_file():
        return 0
    fp = _fp(f)
    if _meta_get(con, key) == fp:
        return 0
    con.execute("DELETE FROM sft_samples WHERE teacher=?", (teacher,))
    n = 0
    with f.open("r", encoding="utf-8") as fh:
        while True:
            offset = fh.tell()
            line = fh.readline()
            if not line:
                break
            if not line.strip():
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            meta = o.get("metadata") or {}
            con.execute(
                "INSERT OR REPLACE INTO sft_samples VALUES(?,?,?,?,?)",
                (o.get("sample_id"), teacher, o.get("episode_id"),
                 meta.get("source_model_call_seq"), offset))
            n += 1
    _meta_set(con, key, fp)
    return n


def build_index(root: Path, db_path: Path, progress=print):
    root = root.resolve()
    t0 = time.time()
    con = connect(db_path)
    with con:
        accepted_map = load_accepted(root)
        rejected_map = load_rejected(root)
        catalog = load_task_catalog(root)
        _meta_set(con, "task_catalog", json.dumps(catalog, ensure_ascii=False))
        progress(f"验收索引 {len(accepted_map)} 条, 导出拒绝 {len(rejected_map)} 个 bundle")

        scanned = indexed = 0
        raw = root / "raw"
        for teacher_dir in sorted(raw.iterdir()) if raw.is_dir() else []:
            teacher = teacher_dir.name
            sessions_dir = teacher_dir / "sessions"
            if not sessions_dir.is_dir():
                continue
            for sess in sorted(sessions_dir.iterdir()):
                rollout = sess / "rollout"
                if not rollout.is_dir():
                    continue
                scanned += 1
                try:
                    if index_session(con, root, teacher, rollout,
                                     accepted_map, rejected_map):
                        indexed += 1
                except Exception as e:  # 单个 bundle 损坏不应拖垮整个索引
                    progress(f"  [warn] {sess.name}: {e}")
        progress(f"扫描 {scanned} 个 session, 重建 {indexed} 个")

        for teacher_dir in sorted((root / "exported").iterdir()) if (root / "exported").is_dir() else []:
            n = index_sft_export(con, root, teacher_dir.name)
            if n:
                progress(f"SFT 样本索引 {teacher_dir.name}: {n} 条")
        _meta_set(con, "built_at_s", str(time.time()))
    con.close()
    progress(f"索引完成, 耗时 {time.time()-t0:.1f}s -> {db_path}")


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="构建 SFT 轨迹查看器索引")
    p.add_argument("--root", default=str(Path(__file__).resolve().parent.parent),
                   help="SFT_data 根目录 (默认: fronted/ 的上一级)")
    p.add_argument("--db", default=None, help="SQLite 路径 (默认 <root>/viewer/index.sqlite)")
    args = p.parse_args(argv)
    root = Path(args.root).resolve()
    db = Path(args.db) if args.db else root / "viewer" / "index.sqlite"
    build_index(root, db)


if __name__ == "__main__":
    main()
