"""Query-expansion baselines (paper, Sec. 4.3).

Both baselines pool the query with its top-e neighbours and re-score every
candidate with a weighted vote over the pooled tracks' own shortlists. AQE
weights the pool uniformly; alpha-QE discounts distant members. Since the
similarity used here is exp(-d), the weight sim^alpha of Radenovic et al. is
exp(-alpha * d).
"""
import numpy as np


def similarity_row(shortlists, track):
    """s_x: the track's stored shortlist as a sparse similarity map (Sec. 4.3)."""
    r = shortlists.row(track)
    if r is None:
        return {}
    return {int(c): float(np.exp(-d))
            for c, d in zip(shortlists.idx[r], shortlists.dist[r])}


def query_expansion_row(shortlists, row, e=5, alpha=3.0, depth=200, cache=None):
    """Permutation of one query's top-`depth` candidates under query expansion.

    alpha = 0 gives AQE (uniform weights).
    """
    depth = min(depth, shortlists.depth)
    query = int(shortlists.query_ids[row])
    pool = [query] + [int(j) for j in shortlists.idx[row, :e]]
    if alpha == 0.0:
        weights = np.ones(len(pool))
    else:
        d = np.concatenate([[0.0], shortlists.dist[row, :e].astype(float)])
        weights = np.exp(-alpha * d)
    weights /= weights.sum()

    cache = {} if cache is None else cache
    rows = []
    for m in pool:
        if m not in cache:
            cache[m] = similarity_row(shortlists, m)
        rows.append(cache[m])

    candidates = shortlists.idx[row, :depth]
    scores = np.array([sum(w * r.get(int(c), 0.0) for w, r in zip(weights, rows))
                       for c in candidates])
    return np.argsort(-scores, kind="stable")


def query_expansion(shortlists, e=5, alpha=3.0, depth=200, rows=None):
    """Re-rank every query with query expansion and return the new track ids."""
    rows = range(shortlists.n_queries) if rows is None else rows
    cache = {}
    out = []
    for row in rows:
        order = query_expansion_row(shortlists, row, e=e, alpha=alpha,
                                    depth=depth, cache=cache)
        head = shortlists.idx[row, :depth][order]
        out.append(np.concatenate([head, shortlists.idx[row, depth:]]))
    return np.stack(out)
