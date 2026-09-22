#!/usr/bin/env python3
"""Every number behind Figure 1, untruncated.

The figure shows the first six entries of each shortlist. This prints the whole
thing: the ranking before and after re-ranking, which candidates rank the query
back, the expanded set, the Jaccard distance of every candidate, and the average
precision on both sides.
"""
import argparse
from pathlib import Path

import numpy as np

from kreciprocal import Shortlists
from kreciprocal.evaluate import average_precision
from kreciprocal.rerank import Reranker, jaccard, rerank_row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump", type=Path, required=True)
    ap.add_argument("--query", type=int, required=True, help="gallery track id")
    ap.add_argument("--k1", type=int, default=30)
    ap.add_argument("--k2", type=int, default=2)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--depth", type=int, default=200)
    ap.add_argument("--expansion", default="standard")
    ap.add_argument("--show", type=int, default=12, help="rows to print")
    ap.add_argument("--csv", type=Path, help="write the full candidate table here")
    args = ap.parse_args()

    s = Shortlists.from_dump(args.dump)
    row = s.row(args.query)
    if row is None:
        raise SystemExit(f"track {args.query} has no shortlist of its own")
    r = Reranker(s, k1=args.k1, k2=args.k2, expansion=args.expansion)

    npos = int(s.positives_per_query()[args.query])
    base = s.is_match[row]
    order = rerank_row(r, row, lam=args.lam, depth=args.depth)
    after = np.concatenate([base[:args.depth][order], base[args.depth:]])

    recip = r.reciprocal_set(args.query)
    expanded = r.expand(args.query)
    print(f"query {args.query}   positives in gallery: {npos}")
    print(f"reciprocal set R(q,k1): {len(recip)} of the top {args.k1} answer back")
    print(f"expanded set R*(q):     {len(expanded)} members")
    print(f"AP {100*average_precision(base.tolist(), npos):.1f} -> "
          f"{100*average_precision(after.tolist(), npos):.1f}")

    fq = r.smooth(args.query)
    cands = s.idx[row, :args.depth]
    dj = np.array([jaccard(fq, r.smooth(int(c))) for c in cands])
    dstar = (1 - args.lam) * dj + args.lam * (s.dist[row, :args.depth] / s.max_distance)
    new_rank = np.empty(len(order), dtype=int)
    new_rank[order] = np.arange(1, len(order) + 1)

    print(f"\n{'rank':>5} {'new':>4} {'track':>7} {'match':>6} {'d':>8} {'d_J':>7} {'d*':>7}")
    for i in range(min(args.show, args.depth)):
        print(f"{i+1:5d} {new_rank[i]:4d} {int(cands[i]):7d} "
              f"{'yes' if base[i] else '':>6} {s.dist[row, i]:8.4f} "
              f"{dj[i]:7.4f} {dstar[i]:7.4f}")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["rank_before", "rank_after", "track", "is_match", "d", "d_J", "d_star",
                        "in_reciprocal_set", "in_expanded_set"])
            rs, es = set(recip), set(expanded)
            for i in range(args.depth):
                t = int(cands[i])
                w.writerow([i + 1, int(new_rank[i]), t, bool(base[i]),
                            f"{s.dist[row, i]:.6f}", f"{dj[i]:.6f}", f"{dstar[i]:.6f}",
                            t in rs, t in es])
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
