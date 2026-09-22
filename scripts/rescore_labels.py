#!/usr/bin/env python3
"""Score one re-ranking under two label sets (paper, Sec. 4.5).

Re-ranking never reads labels, so the order it produces is the same whatever the
ground truth says. That makes the label-noise check cheap: compute the order
once, then score it twice, under the benchmark's original cliques and under a
corrected set in which false splits have been merged.
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np

from kreciprocal import Shortlists
from kreciprocal.evaluate import average_precision, confidence_interval
from kreciprocal.rerank import Reranker, rerank_row


def merged_cliques(clique_ids, mapping_csv):
    """Apply a map of old clique id -> merged group id."""
    mapping = {}
    with open(mapping_csv) as fh:
        for row in csv.DictReader(fh):
            mapping[row["old_clique_id"]] = row["merged_group_id"]
    return [mapping.get(c, c) for c in clique_ids], set(mapping)


def positives(clique_ids):
    from collections import Counter
    counts = Counter(clique_ids)
    return np.array([counts[c] - 1 for c in clique_ids])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump", type=Path, required=True)
    ap.add_argument("--merges", type=Path, required=True, help="cleaned_clique_map.csv")
    ap.add_argument("--k1", type=int, default=30)
    ap.add_argument("--k2", type=int, default=2)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--depth", type=int, default=200)
    ap.add_argument("--expansion", default="standard")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    s = Shortlists.from_dump(args.dump)
    original = list(s.clique_ids)
    corrected, touched_ids = merged_cliques(original, args.merges)
    orig_arr, corr_arr = np.array(original), np.array(corrected)
    npos_o, npos_c = positives(original), positives(corrected)
    print(f"{len(touched_ids)} cliques merged into {len(set(corr_arr[np.isin(orig_arr, list(touched_ids))]))} groups")

    r = Reranker(s, k1=args.k1, k2=args.k2, expansion=args.expansion)
    rec = {k: [] for k in ("gidx", "touched",
                           "ap_base_o", "ap_rr_o", "ap_base_c", "ap_rr_c")}
    t0 = time.time()
    for i, row in enumerate(range(s.n_queries), 1):
        q = int(s.query_ids[row])
        order = rerank_row(r, row, lam=args.lam, depth=args.depth)
        cand = s.idx[row]
        head = np.concatenate([cand[:args.depth][order], cand[args.depth:]])
        for tag, labels, npos in (("o", orig_arr, npos_o), ("c", corr_arr, npos_c)):
            n = int(npos[q])
            if n <= 0:
                rec[f"ap_base_{tag}"].append(np.nan)
                rec[f"ap_rr_{tag}"].append(np.nan)
                continue
            base = labels[cand] == labels[q]
            after = labels[head] == labels[q]
            rec[f"ap_base_{tag}"].append(average_precision(base.tolist(), n))
            rec[f"ap_rr_{tag}"].append(average_precision(after.tolist(), n))
        rec["gidx"].append(q)
        rec["touched"].append(original[q] in touched_ids)
        if i % 5000 == 0:
            rate = i / (time.time() - t0)
            print(f"  {i:,}/{s.n_queries:,}  {rate:.0f} q/s  "
                  f"eta {(s.n_queries-i)/rate/60:.1f} min", flush=True)

    out = {k: np.array(v) for k, v in rec.items()}
    ok_o = ~np.isnan(out["ap_base_o"])
    ok_c = ~np.isnan(out["ap_base_c"])
    d_o = 100 * (out["ap_rr_o"] - out["ap_base_o"])[ok_o]
    d_c = 100 * (out["ap_rr_c"] - out["ap_base_c"])[ok_c]
    print(f"\noriginal labels   {d_o.mean():+.2f} +- {confidence_interval(d_o):.2f}  (n={len(d_o):,})")
    print(f"corrected labels  {d_c.mean():+.2f} +- {confidence_interval(d_c):.2f}  (n={len(d_c):,})")
    t = out["touched"]
    print(f"\nmerge-touched cliques only:")
    print(f"  original   {100*(out['ap_rr_o']-out['ap_base_o'])[t & ok_o].mean():+.2f}")
    print(f"  corrected  {100*(out['ap_rr_c']-out['ap_base_c'])[t & ok_c].mean():+.2f}  (n={int((t & ok_c).sum()):,})")
    if args.out:
        np.savez(args.out, **out)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
