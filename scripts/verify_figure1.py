#!/usr/bin/env python3
"""Check every number printed in Figure 1 against the dump it came from.

The figure's layout is hand-maintained, which is fine for layout and not for
data: a stale distance in the paper's flagship figure would be invisible and
wrong. This reads the values back out of the .tex and recomputes each one.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np

from kreciprocal import Shortlists
from kreciprocal.evaluate import average_precision
from kreciprocal.rerank import Reranker, jaccard, rerank_row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump", type=Path, required=True)
    ap.add_argument("--tex", type=Path, required=True)
    ap.add_argument("--query", type=int, required=True)
    ap.add_argument("--version", type=int, required=True, help="the true version a+")
    ap.add_argument("--distractor", type=int, required=True, help="a-")
    ap.add_argument("--k1", type=int, default=30)
    ap.add_argument("--k2", type=int, default=2)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--depth", type=int, default=200)
    ap.add_argument("--expansion", default="standard")
    args = ap.parse_args()

    tex = args.tex.read_text()
    s = Shortlists.from_dump(args.dump)
    row = s.row(args.query)
    r = Reranker(s, k1=args.k1, k2=args.k2, expansion=args.expansion)

    npos = int(s.positives_per_query()[args.query])
    base = s.is_match[row]
    order = rerank_row(r, row, lam=args.lam, depth=args.depth)
    after = np.concatenate([base[:args.depth][order], base[args.depth:]])
    fq = r.smooth(args.query)

    checks, fails = [], []

    def check(name, got, want, tol=5e-3):
        ok = abs(got - want) <= tol
        checks.append((name, got, want, ok))
        if not ok:
            fails.append(name)

    # the two average precisions printed in the badges
    ap_before = 100 * average_precision(base.tolist(), npos)
    ap_after = 100 * average_precision(after.tolist(), npos)
    printed = [float(m) for m in re.findall(r"AP \$=([\d.]+)\$", tex)]
    check("AP before", ap_before, min(printed), tol=0.05)
    check("AP after", ap_after, max(printed), tol=0.05)

    # the two Jaccard distances under the middle panels
    for who, track, pat in (("a+", args.version, r"d_J\(q,a\^\+\) = ([\d.]+)"),
                            ("a-", args.distractor, r"d_J\(q,a\^-\) = ([\d.]+)")):
        want = float(re.search(pat, tex).group(1))
        check(f"d_J {who}", jaccard(fq, r.smooth(track)), want, tol=5e-3)

    # where the two tracks end up
    cands = list(s.idx[row, :args.depth])
    new_rank = np.empty(len(order), dtype=int)
    new_rank[order] = np.arange(1, len(order) + 1)
    check("a+ new rank", float(new_rank[cands.index(args.version)]), 1.0, tol=0)
    check("a- new rank", float(new_rank[cands.index(args.distractor)]), 9.0, tol=0)

    width = max(len(n) for n, *_ in checks)
    for name, got, want, ok in checks:
        print(f"  {'ok ' if ok else 'BAD'} {name:<{width}}  got {got:8.3f}  printed {want:8.3f}")
    print(f"\n{len(checks) - len(fails)}/{len(checks)} values match")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
