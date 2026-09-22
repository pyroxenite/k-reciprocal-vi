#!/usr/bin/env python3
"""Gain by clique size, the stratification behind Figure 2 of the paper.

Queries are grouped by how many other versions of the same work the gallery
holds. The point of the figure is the trend: re-ranking pays when a work has
several versions and hurts when it has one (paper, Sec. 4.4).
"""
import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BINS = [(1, 1, "2"), (2, 3, "3-4"), (4, 8, "5-9"),
        (9, 18, "10-19"), (19, 48, "20-49"), (49, 10**9, "50+")]


def ci95(x):
    return 1.959963985 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float("nan")


def strata(path):
    d = np.load(path)
    keep = d["npos"] > 0
    npos, delta = d["npos"][keep], 100 * (d["ap_rr"][keep] - d["ap_base"][keep])
    rows = []
    for lo, hi, label in BINS:
        sel = (npos >= lo) & (npos <= hi)
        if sel.sum():
            rows.append((label, int(sel.sum()), float(delta[sel].mean()),
                         float(ci95(delta[sel]))))
    return rows, delta


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", type=Path,
                    default=ROOT / "data" / "records" / "clews_official.npz")
    ap.add_argument("--plot", type=Path, help="optional path for a PNG")
    args = ap.parse_args()

    rows, delta = strata(args.records)
    print(f"{'clique size':>12}  {'queries':>8}  {'delta mAP':>10}  {'95% CI':>7}")
    for label, n, mean, half in rows:
        print(f"{label:>12}  {n:8,}  {mean:+10.2f}  {half:7.2f}")
    print(f"{'all':>12}  {len(delta):8,}  {delta.mean():+10.2f}  {ci95(delta):7.2f}")

    # Headroom is the obvious alternative explanation: easy queries have less
    # room to improve. Splitting at the median baseline AP rules it out, since
    # the trend survives in both halves (paper, Sec. 4.4).
    d = np.load(args.records)
    keep = d["npos"] > 0
    base = d["ap_base"][keep]
    npos = d["npos"][keep]
    dl = 100 * (d["ap_rr"][keep] - base)
    median = np.median(base)
    for half_name, sel_half in (("below median AP", base <= median),
                                ("above median AP", base > median)):
        small = sel_half & (npos == 1)
        large = sel_half & (npos >= 9) & (npos <= 18)
        print(f"  {half_name}: two-version {dl[small].mean():+.2f}  "
              f"10-19 versions {dl[large].mean():+.2f}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        labels = [r[0] for r in rows]
        means = [r[2] for r in rows]
        errs = [r[3] for r in rows]
        fig, ax = plt.subplots(figsize=(4.2, 2.2))
        ax.axhline(0, color="black", lw=0.6)
        ax.errorbar(labels, means, yerr=errs, fmt="o", ms=3, lw=0.8, capsize=2)
        ax.set_xlabel("query clique size")
        ax.set_ylabel(r"$\Delta$ mAP")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=200)
        print(f"\nwrote {args.plot}")


if __name__ == "__main__":
    main()
