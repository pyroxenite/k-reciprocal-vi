#!/usr/bin/env python3
"""Rebuild the paper's tables from the per-query records.

Each record file holds, for every query of the Discogs-VI test split, the
average precision before re-ranking, after re-ranking, and after the variant
that keeps whatever the first stage put at rank 1, plus the rank of the first
correct hit in each case. Every number below follows from those columns, so no
shortlist dump is needed (paper, Sec. 3.3).
"""
import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RECORDS = ROOT / "data" / "records"

# Exact baseline mAP over the full ranking, as reported by each encoder's own
# evaluation. The records store shortlist AP, which is marginally lower because
# a few positives sit beyond the stored depth (paper, Sec. 4.1).
EXACT_BASELINE = {
    "clews_official": 76.91,
    "clews_reproduction": 70.45,
    "vinet_official": 44.26,
    "fish_official": 68.82,
}


def ci95(x):
    return 1.959963985 * x.std(ddof=1) / np.sqrt(len(x))


def first_match_stats(fr_before, fr_after):
    """MR1 and HR@1 over the queries whose shortlist holds a hit.

    A query with no hit inside the shortlist has no first-match rank at all.
    Re-ranking permutes the shortlist, so the same queries are censored either
    way and the comparison stays paired.
    """
    ok = fr_before > 0
    assert np.array_equal(ok, fr_after > 0), "censoring differs between orders"
    a, b = fr_before[ok].astype(float), fr_after[ok].astype(float)
    return dict(n=int(ok.sum()), mr1_before=a.mean(), mr1_after=b.mean(),
                hr1_before=100 * (a == 1).mean(), hr1_after=100 * (b == 1).mean())


def report(name, path):
    d = np.load(path)
    keep = d["npos"] > 0
    base, rr = d["ap_base"][keep], d["ap_rr"][keep]
    delta = 100 * (rr - base)
    exact = EXACT_BASELINE[name]

    print(f"\n{name}   ({keep.sum():,} queries, k1={int(d['k1'])}, "
          f"k2={int(d['k2'])}, lambda={float(d['lam'])}, R={int(d['rerank_depth'])})")
    print(f"  baseline mAP        {exact:6.2f}   (shortlist {100*base.mean():6.2f})")
    print(f"  k-reciprocal        {exact + delta.mean():6.2f}   "
          f"delta {delta.mean():+6.2f} +- {ci95(delta):.2f}   "
          f"{100*(delta > 0).mean():.0f}% of queries improved")

    if "ap_p1" in d:
        d1 = 100 * (d["ap_p1"][keep] - base)
        print(f"  keeping rank 1      {exact + d1.mean():6.2f}   "
              f"delta {d1.mean():+6.2f} +- {ci95(d1):.2f}")

    fm = first_match_stats(d["fr_base"][keep], d["fr_rr"][keep])
    print(f"  MR1  {fm['mr1_before']:6.2f} -> {fm['mr1_after']:6.2f}      "
          f"HR@1 {fm['hr1_before']:6.2f} -> {fm['hr1_after']:6.2f}   "
          f"(over {fm['n']:,} queries with a hit in the shortlist)")
    if "fr_p1" in d:
        fm1 = first_match_stats(d["fr_base"][keep], d["fr_p1"][keep])
        print(f"  keeping rank 1:  MR1 -> {fm1['mr1_after']:6.2f}   "
              f"HR@1 -> {fm1['hr1_after']:6.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", type=Path, default=RECORDS)
    args = ap.parse_args()

    print("Table 1 (Discogs-VI test, official CLEWS) and the encoder rows of Table 3")
    for name in EXACT_BASELINE:
        path = args.records / f"{name}.npz"
        if path.exists():
            report(name, path)
        else:
            print(f"\n{name}: missing {path.name}")

    print("\nNot reproducible from records alone:")
    print("  AQE and alpha-QE rows of Table 1   (re-run from a dump, scripts/rerank_dump.py)")
    print("  Discogs-VI mini and SHS100K-A2A rows of Table 3  (separate galleries)")


if __name__ == "__main__":
    main()
