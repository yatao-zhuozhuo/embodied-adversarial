"""Download official post-trained Qwen weights with ModelScope into the shared model directory."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=["Qwen3.8-27B", "Qwen3.5-9B", "Qwen3.5-4B"],
                        default=["Qwen3.5-9B"])
    parser.add_argument("--model-root", type=Path, default=Path(__file__).resolve().parents[1] / ".runtime/models")
    parser.add_argument("--revision", default="master")
    args = parser.parse_args()
    from modelscope import snapshot_download

    for name in args.models:
        target = args.model_root / name
        snapshot_download(f"Qwen/{name}", revision=args.revision, local_dir=str(target), max_workers=4)
        index = json.loads((target / "model.safetensors.index.json").read_text())
        shards = set(index["weight_map"].values())
        missing = [s for s in shards if not (target / s).is_file() or (target / s).stat().st_size == 0]
        if missing:
            raise RuntimeError(f"{name}: missing checkpoint shards {missing}")
        print(f"{name}: {len(shards)} checkpoint shards present at {target}")


if __name__ == "__main__":
    main()
