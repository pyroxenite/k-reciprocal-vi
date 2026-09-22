#!/usr/bin/env python3
"""Create and fill a Zenodo deposition for the released shortlists.

The script never publishes. It creates a draft, sets its metadata, reserves a
DOI and uploads files, leaving the final publish step to a human, which is also
the step that makes a record permanent.

The token is read from the environment only:

    export ZENODO_TOKEN=$(security find-generic-password -s zenodo-token -w)
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import requests

LIVE = "https://zenodo.org/api"
SANDBOX = "https://sandbox.zenodo.org/api"

DESCRIPTION = """<p>Top-K retrieval shortlists and per-query records for
<em>Training-free k-reciprocal re-ranking improves state-of-the-art musical
version identification</em>.</p>

<p>Each dump holds, for every track of a Discogs-VI split used as a query, the
ranked ids of its nearest neighbours, their distances, and whether each
candidate is a true version. The per-query records hold average precision
before and after re-ranking, the rank of the first correct hit, and the clique
size. No audio and no embeddings are included, and none can be reconstructed
from these files.</p>

<p>The files are derived from the Discogs-VI dataset, whose metadata is
licensed CC BY-NC-SA 4.0, and are released under the same licence. Please cite
R. O. Araz, X. Serra and D. Bogdanov, <em>Discogs-VI: A musical version
identification dataset based on public editorial metadata</em>, ISMIR 2024,
alongside this record.</p>

<p>Code: <a href="https://github.com/">k-reciprocal-vi</a>.</p>"""

METADATA = {
    "metadata": {
        "upload_type": "dataset",
        "title": ("Retrieval shortlists for training-free k-reciprocal re-ranking "
                  "in musical version identification"),
        "creators": [
            {"name": "Jardin, Pharoah"},
            {"name": "Peeters, Geoffroy"},
            {"name": "Larrouturou, Gustave"},
        ],
        "description": DESCRIPTION,
        "license": "cc-by-nc-sa-4.0",
        "access_right": "open",
        "keywords": ["music information retrieval", "version identification",
                     "cover song identification", "re-ranking", "Discogs-VI"],
        "prereserve_doi": True,
    }
}


def api(base, token):
    s = requests.Session()
    s.headers.update({"Authorization": f"Bearer {token}"})
    s.base = base
    return s


def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sandbox", action="store_true", help="use sandbox.zenodo.org")
    ap.add_argument("--deposition", type=int, help="continue an existing draft")
    ap.add_argument("--file", action="append", default=[], metavar="LOCAL:REMOTE",
                    help="file to upload, optionally renamed")
    ap.add_argument("--metadata-only", action="store_true")
    args = ap.parse_args()

    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        sys.exit("ZENODO_TOKEN is not set. Try:\n"
                 "  export ZENODO_TOKEN=$(security find-generic-password -s zenodo-token -w)")

    base = SANDBOX if args.sandbox else LIVE
    s = api(base, token)

    if args.deposition:
        r = s.get(f"{base}/deposit/depositions/{args.deposition}")
        r.raise_for_status()
        dep = r.json()
    else:
        r = s.post(f"{base}/deposit/depositions", json={})
        r.raise_for_status()
        dep = r.json()
        print(f"created draft {dep['id']}")

    r = s.put(f"{base}/deposit/depositions/{dep['id']}", json=METADATA)
    r.raise_for_status()
    dep = r.json()
    doi = dep["metadata"].get("prereserve_doi", {}).get("doi", "not reserved")
    print(f"draft   {dep['id']}\nDOI     {doi}\nedit at {dep['links']['html']}")

    if args.metadata_only:
        return

    bucket = dep["links"]["bucket"]
    existing = {f["filename"] for f in dep.get("files", [])}
    for spec in args.file:
        local, _, remote = spec.partition(":")
        remote = remote or Path(local).name
        if remote in existing:
            print(f"skip    {remote} (already uploaded)")
            continue
        size = Path(local).stat().st_size
        print(f"upload  {remote}  ({size/1e9:.2f} GB) ...", flush=True)
        # Zenodo occasionally answers a large PUT with 502 or 504. The upload is
        # idempotent, so a failed attempt can simply be repeated.
        for attempt in range(1, 6):
            with open(local, "rb") as fh:
                r = s.put(f"{bucket}/{remote}", data=fh)
            if r.status_code < 500:
                break
            wait = 5 * attempt
            print(f"        {r.status_code} from Zenodo, retrying in {wait}s "
                  f"(attempt {attempt}/5)", flush=True)
            time.sleep(wait)
        r.raise_for_status()
        print(f"        done, md5 {r.json()['checksum']}", flush=True)

    print("\nNothing was published. Publish deliberately, from the web page above.")


if __name__ == "__main__":
    main()
