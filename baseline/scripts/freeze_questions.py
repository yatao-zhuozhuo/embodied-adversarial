"""Freeze only after all real simulator and reproducibility checks pass."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "baseline"
BENCHES = ("libero", "metaworld", "maniskill3", "robocasa")


def reproducibility(bench, index, a, b):
    exact = a.get("frame_sha256") == b.get("frame_sha256")
    state_equal = a.get("initial_sim_state") is not None and a.get("initial_sim_state") == b.get("initial_sim_state")
    if a.get("initial_robot_state") != b.get("initial_robot_state"):
        return {"passed": False, "pixels_exact": exact, "state_exact": state_equal,
                "reason": "robot initial state differs"}
    if bench in {"libero", "robocasa"} and not state_equal:
        return {"passed": False, "pixels_exact": exact, "state_exact": False,
                "reason": "native simulator state missing or different"}
    if exact:
        return {"passed": True, "pixels_exact": True, "state_exact": state_equal}
    if bench == "metaworld" and state_equal:
        import numpy as np
        from PIL import Image
        x = np.asarray(Image.open(ROOT / f"results/environment-preflight/{bench}-{index}.png"), dtype=float)
        y = np.asarray(Image.open(ROOT / f"results/environment-repeat/{bench}-{index}.png"), dtype=float)
        if x.shape != y.shape:
            return {"passed": False, "pixels_exact": False, "state_exact": True}
        delta = np.abs(x - y)
        mae = float(delta.mean())
        # Raster variation is permitted only with bit-identical physical
        # state and goal. One channel value out of 255 is the MAE ceiling.
        return {"passed": mae <= 1.0, "pixels_exact": False, "state_exact": True,
                "pixel_mae_255": mae, "pixel_max_difference": float(delta.max()),
                "pixel_changed_fraction": float(np.mean(delta > 0)), "mae_limit_255": 1.0}
    return {"passed": False, "pixels_exact": exact, "state_exact": state_equal}


def main():
    all_rows, evidence, manifests = [], {}, {}
    for bench in BENCHES:
        path = ROOT / f"manifests/{bench}_10.json"
        manifest = json.loads(path.read_text())
        for i, row in enumerate(manifest["episodes"]):
            a = json.loads((ROOT / f"results/environment-preflight/{bench}-{i}.json").read_text())
            b = json.loads((ROOT / f"results/environment-repeat/{bench}-{i}.json").read_text())
            assert a["passed"] and b["passed"], (bench, i, "lifecycle failed")
            assert a["frame_shape"][:2] == [512, 512], (bench, i, "resolution")
            repeat = reproducibility(bench, i, a, b)
            assert repeat["passed"], (bench, i, "reset is not reproducible", repeat)
            for key in ("env_id", "seed"):
                assert a["episode"][key] == b["episode"][key] == row[key]
            if bench == "robocasa":
                assert a.get("renderer") == b.get("renderer") == "mujoco", "RoboCasa must use the fixed offscreen renderer"
                language = a["episode_metadata"]["lang"]
                assert language and language == b["episode_metadata"]["lang"]
                row["task"] = language
                row["metadata"]["instruction_source"] = "official get_ep_meta().lang at fixed reset seed"
            row["metadata"]["native_success_checker"] = a["native_checker"]
            row["metadata"]["initial_frame_sha256"] = a["frame_sha256"]
            evidence[row["episode_id"]] = {
                "initial_frame_sha256": a["frame_sha256"], "frame_shape": a["frame_shape"],
                "native_checker": a["native_checker"], "initial_sim_state": a.get("initial_sim_state"),
                "episode_metadata": a.get("episode_metadata"), "cameras": a.get("cameras"),
                "checks": a["checks"], "repeat_checks": b["checks"],
                "reproducibility": repeat,
            }
        manifests[bench] = manifest
        all_rows.extend(manifest["episodes"])
    # No writes until all 40 records pass the gate.
    for bench, manifest in manifests.items():
        (ROOT / f"manifests/{bench}_10.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        plan_path = ROOT / f"configs/evaluation/{bench}.json"
        plan = json.loads(plan_path.read_text())
        plan["episodes"] = manifest["episodes"]
        plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    (ROOT / "manifests/all_40.json").write_text(json.dumps({"episodes": all_rows}, ensure_ascii=False, indent=2) + "\n")
    (ROOT / "manifests/reset_evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted((ROOT / "manifests").glob("*.json")) if p.name != "reset_evidence.json"}
    (ROOT / "manifests/SHA256SUMS").write_text("".join(f"{h}  {name}\n" for name, h in hashes.items()))
    print(json.dumps({"frozen": True, "episodes": len(all_rows), "sha256": hashes}, indent=2))


if __name__ == "__main__":
    main()
