#!/usr/bin/env python3
"""Download the published shortlist dumps from Zenodo and check them.

Files already present with the right checksum are left alone, so an interrupted
download can simply be repeated.
"""
import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RECORD = "22894764"                      # reserved DOI 10.5281/zenodo.22894764
BASE = f"https://zenodo.org/records/{RECORD}/files"
MANIFEST = ROOT / "data" / "zenodo_manifest.json"


def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def download(name, dest, expected_size):
    url = f"{BASE}/{name}?download=1"
    tmp = dest.with_suffix(dest.suffix + ".part")
    seen = 0
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as out:
        while True:
            block = r.read(1 << 20)
            if not block:
                break
            out.write(block)
            seen += len(block)
            if expected_size:
                pct = 100 * seen / expected_size
                print(f"\r  {name}  {seen/1e6:8.1f} / {expected_size/1e6:.1f} MB "
                      f"({pct:5.1f}%)", end="", flush=True)
    print()
    tmp.rename(dest)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", type=Path, default=ROOT / "data" / "dumps")
    ap.add_argument("--only", nargs="*", help="file names to fetch (default: all)")
    ap.add_argument("--records-only", action="store_true",
                    help="fetch just the small per-query records")
    args = ap.parse_args()

    if not MANIFEST.exists():
        sys.exit(f"missing {MANIFEST}. It ships with the repository.")
    manifest = json.loads(MANIFEST.read_text())

    wanted = args.only or [n for n in manifest
                           if not args.records_only or n.endswith(".npz")]
    args.dest.mkdir(parents=True, exist_ok=True)

    for name in wanted:
        meta = manifest[name]
        dest = args.dest / name
        if dest.exists() and md5(dest) == meta["md5"]:
            print(f"have  {name}")
            continue
        print(f"fetch {name}  ({meta['size']/1e6:.1f} MB)")
        download(name, dest, meta["size"])
        got = md5(dest)
        if got != meta["md5"]:
            sys.exit(f"checksum mismatch for {name}: {got} != {meta['md5']}")
        print(f"      ok, md5 {got}")

    print(f"\nfiles are in {args.dest}")


if __name__ == "__main__":
    main()
