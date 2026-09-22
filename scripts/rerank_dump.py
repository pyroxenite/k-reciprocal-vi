#!/usr/bin/env python3
"""Re-rank a shortlist dump and report the gain (paper, Tables 1 and 3).

Reads a dump, re-ranks every query (or a random subset), and prints the paired
change in mAP with its interval. Optionally writes the per-query records that
scripts/reproduce_tables.py consumes.
"""
import argparse
import time
from pathlib import Path

import numpy as np

from kreciprocal import Shortlists
from kreciprocal.evaluate import average_precision, confidence_interval
from kreciprocal.rerank import Reranker, rerank_row


def first_match_rank(mask):
    hit = np.flatnonzero(mask)
    return int(hit[0]) + 1 if len(hit) else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump", type=Path, required=True)
    ap.add_argument("--k1", type=int, default=30)
    ap.add_argument("--k2", type=int, default=2)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--depth", type=int, default=200)
    ap.add_argument("--expansion", choices=("standard", "loose", "zhong", "published"),
                    default="standard",
                    help="'standard' is Zhong's rule and the paper's; 'loose' is the wider variant")
    ap.add_argument("--queries", type=int, default=0,
                    help="evaluate a random subset of this size (0 = all)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--records", type=Path, help="write per-query records here (.npz)")
    args = ap.parse_args()

    t0 = time.time()
    s = Shortlists.from_dump(args.dump)
    npos = s.positives_per_query()
    rows = np.arange(s.n_queries)
    if args.queries and args.queries < len(rows):
        rows = np.sort(np.random.default_rng(args.seed).choice(rows, args.queries, replace=False))
    print(f"{s.n_queries:,} queries x {s.depth} neighbours, evaluating {len(rows):,} "
          f"({time.time()-t0:.0f}s to load)", flush=True)

    r = Reranker(s, k1=args.k1, k2=args.k2, expansion=args.expansion)
    out = {k: [] for k in ("gidx", "npos", "ap_base", "ap_rr", "ap_p1",
                           "fr_base", "fr_rr", "fr_p1", "n_recip", "n_expanded")}
    t0 = time.time()
    for done, row in enumerate(rows, 1):
        q = int(s.query_ids[row])
        n = int(npos[q])
        if n <= 0:
            continue
        order = rerank_row(r, row, lam=args.lam, depth=args.depth)
        base = s.is_match[row]
        after = np.concatenate([base[:len(order)][order], base[len(order):]])
        out["gidx"].append(q)
        out["npos"].append(n)
        out["ap_base"].append(average_precision(base.tolist(), n))
        out["ap_rr"].append(average_precision(after.tolist(), n))
        out["fr_base"].append(first_match_rank(base))
        out["fr_rr"].append(first_match_rank(after))

        # The variant that keeps whatever the first stage put at rank 1 and
        # re-sorts only the rest (paper, Table 1, last row).
        keep = np.concatenate([[0], order[order != 0]])
        p1 = np.concatenate([base[:len(order)][keep], base[len(order):]])
        out["ap_p1"].append(average_precision(p1.tolist(), n))
        out["fr_p1"].append(first_match_rank(p1))
        out["n_recip"].append(len(r.reciprocal_set(q)))
        out["n_expanded"].append(len(r.expand(q)))
        if done % 2000 == 0:
            rate = done / (time.time() - t0)
            print(f"  {done:,}/{len(rows):,}  {rate:.0f} q/s  "
                  f"eta {(len(rows)-done)/rate/60:.1f} min", flush=True)

    delta = 100 * (np.array(out["ap_rr"]) - np.array(out["ap_base"]))
    print(f"\n{len(delta):,} queries scored in {(time.time()-t0)/60:.1f} min")
    print(f"shortlist mAP {100*np.mean(out['ap_base']):.2f} -> "
          f"{100*np.mean(out['ap_rr']):.2f}")
    print(f"paired delta  {delta.mean():+.2f} +- {confidence_interval(delta):.2f} "
          f"({100*(delta > 0).mean():.0f}% improved)")

    if args.records:
        np.savez(args.records, k1=args.k1, k2=args.k2, lam=args.lam,
                 rerank_depth=args.depth,
                 **{k: np.array(v) for k, v in out.items()})
        print(f"wrote {args.records}")


if __name__ == "__main__":
    main()
