"""Scoring, and the exact paired estimator of the gain (paper, Sec. 3.3).

Re-ranking only permutes the first R positions, so the tail of a ranking
contributes the same average precision before and after. The change in AP can
therefore be read off the shortlist alone, exactly, and only its mean over
queries carries a confidence interval.
"""
import numpy as np


def average_precision(matches, n_positives):
    """AP of one ranking, given a boolean hit mask and the true positive count."""
    if n_positives <= 0:
        return None
    hits = 0
    total = 0.0
    for rank, is_hit in enumerate(matches, start=1):
        if is_hit:
            hits += 1
            total += hits / rank
    return total / n_positives


def confidence_interval(values, level=0.95):
    """Half-width of the normal interval on the mean of paired differences."""
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return float("nan")
    z = 1.959963985 if level == 0.95 else float(level)
    return z * values.std(ddof=1) / np.sqrt(len(values))


def paired_delta_ap(shortlists, reranked_rows, rows=None, n_positives=None):
    """Per-query change in shortlist AP between the stored and re-ranked orders.

    reranked_rows[i] is the permutation of row `rows[i]`'s candidates, as
    returned by rerank_row. Returns the per-query deltas, skipping queries with
    no positives at all.
    """
    if shortlists.is_match is None:
        raise ValueError("is_match is required to score a ranking")
    if n_positives is None:
        n_positives = shortlists.positives_per_query()
    rows = range(shortlists.n_queries) if rows is None else rows

    deltas = []
    for row, order in zip(rows, reranked_rows):
        query = int(shortlists.query_ids[row])
        npos = int(n_positives[query])
        if npos <= 0:
            continue
        base = shortlists.is_match[row]
        depth = len(order)
        after = np.concatenate([base[:depth][order], base[depth:]])
        deltas.append(average_precision(after.tolist(), npos)
                      - average_precision(base.tolist(), npos))
    return np.asarray(deltas)


def summarise(deltas, baseline_map=None):
    """Mean gain with its interval, and the estimated re-ranked mAP (Eq. 7)."""
    mean = float(np.mean(deltas))
    half = confidence_interval(deltas)
    out = {"n": len(deltas), "delta": mean, "ci95": half,
           "improved": float(np.mean(deltas > 0))}
    if baseline_map is not None:
        out["baseline_map"] = baseline_map
        out["reranked_map"] = baseline_map + mean
    return out
