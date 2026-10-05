"""Run one real simulator lifecycle; never invokes a language model."""
import argparse
import faulthandler
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def plain(x):
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    if hasattr(x, "tolist"):
        return x.tolist()
    return x


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--index", type=int, required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    row = json.loads(Path(a.manifest).read_text())["episodes"][a.index]
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    result = {"episode": row, "passed": False, "checks": {}, "python": sys.version,
              "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}}
    env = None
    start = time.monotonic()
    faulthandler.dump_traceback_later(60, repeat=True)
    try:
        import numpy as np
        from PIL import Image
        import gymnasium as gym
        random.seed(row["seed"])
        np.random.seed(row["seed"])
        import sim.env_registry
        print(f"Creating {row['env_id']}", flush=True)
        env = gym.make(row["env_id"], seed=row["seed"], render_mode="rgb_array",
                       image_width=512, image_height=512, disable_env_checker=True)
        obs, info = env.reset(seed=row["seed"])
        print("Reset complete", flush=True)
        result["checks"]["reset"] = True
        result["reset_info"] = plain(info)
        result["observation_keys"] = sorted(obs) if isinstance(obs, dict) else []
        result["initial_robot_state"] = plain(obs.get("proprio", {})) if isinstance(obs, dict) else {}
        result["task_description"] = obs.get("task_description", "") if isinstance(obs, dict) else ""
        result["cameras"] = {}
        for name, camera in obs.get("cameras", {}).items():
            result["cameras"][name] = {}
            for kind in ("rgb", "depth", "intrinsics", "extrinsics"):
                if camera.get(kind) is not None:
                    value = plain(camera[kind])
                    if isinstance(value, dict):
                        result["cameras"][name][kind] = value
                        continue
                    values = np.asarray(value)
                    result["cameras"][name][kind] = {"shape": list(values.shape),
                                                     "finite": bool(np.isfinite(values).all())}
        raw = env.unwrapped._env
        if row["metadata"]["benchmark"] == "robocasa":
            result["renderer"] = raw._env.renderer
            result["episode_metadata"] = plain(raw._env.get_ep_meta())
            result["initial_sim_state"] = plain(raw._env.sim.get_state().flatten())
            result["camera_names"] = list(raw._camera_names)
        if row["metadata"]["benchmark"] == "libero":
            result["initial_sim_state"] = plain(raw._env.get_sim_state())
        elif row["metadata"]["benchmark"] == "metaworld":
            native = raw.unwrapped
            result["initial_sim_state"] = {"qpos": plain(native.data.qpos),
                                            "qvel": plain(native.data.qvel),
                                            "target": plain(native._target_pos)}
        frame = np.asarray(env.render())
        if frame.ndim == 4 and frame.shape[0] == 1:
            frame = frame[0]
        assert frame.ndim == 3 and frame.shape[-1] in (3, 4), frame.shape
        assert np.isfinite(frame).all() and float(frame.std()) > 1, "invalid/blank render"
        frame = frame.astype(np.uint8)
        Image.fromarray(frame).save(out.with_suffix(".png"))
        result["checks"]["render"] = True
        result["frame_shape"] = list(frame.shape)
        result["frame_sha256"] = hashlib.sha256(frame.tobytes()).hexdigest()
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        _, reward, terminated, truncated, step_info = env.step(action)
        result["checks"]["step"] = True
        result.update(reward=float(reward), terminated=bool(terminated), truncated=bool(truncated),
                      action_shape=list(action.shape), step_info=plain(step_info))
        if row["metadata"]["benchmark"] == "libero":
            success = raw._env.check_success()
            checker = "libero.check_success"
        elif "success" in step_info:
            success = step_info["success"]
            checker = "step.info.success"
        elif "task_success" in step_info:
            success = step_info["task_success"]
            checker = "step.info.task_success"
        else:
            raise RuntimeError("Native success field missing; reward is not a success checker")
        result["native_success"] = plain(success)
        result["native_checker"] = checker
        result["checks"]["native_success_read"] = True
        result["passed"] = True
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        faulthandler.cancel_dump_traceback_later()
        if env is not None:
            try:
                env.close()
                result["checks"]["close"] = True
            except Exception:
                result["passed"] = False
                result["close_error"] = traceback.format_exc()
        result["elapsed_s"] = round(time.monotonic() - start, 3)
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"episode_id": row["episode_id"], "passed": result["passed"],
                      "output": str(out), "error": result.get("error", "")[-1200:]}), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
