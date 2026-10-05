"""Download pinned official assets through Box's public shared-file endpoint.

The upstream /shared/static endpoint returns 404 on this host. This uses
the same public file, not an asset substitution. Interrupted zips resume.
"""
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "baseline/.runtime/sources/robocasa/robocasa/models/assets"
CACHE = ROOT / "baseline/.runtime/robocasa-downloads"
DESTINATIONS = {"textures": "textures", "generative_textures": "generative_textures",
                "fixtures_lightwheel": "fixtures", "objaverse": "objects/objaverse",
                "aigen_objs": "objects/aigen_objs", "objects_lightwheel": "objects/lightwheel"}


def download(item, *, extract_only=False, download_only=False):
    name, link = item
    if name not in DESTINATIONS:
        return
    marker = CACHE / f"{name}.complete.json"
    if marker.exists():
        return
    archive = CACHE / f"{name}.zip"
    downloaded = CACHE / f"{name}.download.json"
    if not extract_only:
        with urllib.request.urlopen(link, timeout=60) as response:
            html = response.read().decode()
        fid = re.search(r'"itemID":(\d+)', html).group(1)
        size = int(re.search(r'"itemSize":(\d+)', html).group(1))
        shared = link.rsplit("/", 1)[1]
        url = f"https://utexas.app.box.com/index.php?rm=box_download_shared_file&shared_name={shared}&file_id=f_{fid}"
        if not archive.exists() or archive.stat().st_size != size:
            print(f"Downloading {name}, file {fid}", flush=True)
            subprocess.run(["curl", "--fail", "--location", "--retry", "5", "--retry-delay", "3",
                            "--connect-timeout", "30", "--speed-limit", "1024", "--speed-time", "120",
                            "--continue-at", "-", "--silent", "--show-error", "--output", str(archive), url], check=True)
        assert archive.stat().st_size == size
        downloaded.write_text(json.dumps({"shared_link": link, "file_id": fid, "size": size}) + "\n")
        print(f"Downloaded {name}", flush=True)
    if download_only:
        return
    meta = json.loads(downloaded.read_text())
    assert archive.stat().st_size == meta["size"]
    print(f"Extracting {name}", flush=True)
    with zipfile.ZipFile(archive) as z:
        bad = z.testzip()
        if bad:
            raise ValueError(f"Corrupt archive: {bad}")
        destination = (ASSETS / DESTINATIONS[name]).parent
        for entry in z.infolist():
            target = (destination / entry.filename).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError("Unsafe archive path")
        z.extractall(destination)
    digest = hashlib.file_digest(archive.open("rb"), "sha256").hexdigest()
    marker.write_text(json.dumps({**meta,
                                  "sha256": digest, "destination": DESTINATIONS[name]}, indent=2) + "\n")
    print(f"Complete {name}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--download-only", action="store_true")
    modes.add_argument("--extract-only", action="store_true")
    args = parser.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    links = json.loads((ASSETS / "box_links/box_links_assets.json").read_text())
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(lambda item: download(item, extract_only=args.extract_only,
                                           download_only=args.download_only), links.items()))
