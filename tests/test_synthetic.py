"""A synthetic collection where the answer is known by construction.

Tracks are built in cliques of a few versions each. Within a clique distances
are small; across cliques they are large but not always larger, so the first
stage makes mistakes that reciprocity can repair.
"""
import numpy as np
import pytest

from kreciprocal import Shortlists, rerank, paired_delta_ap, summarise, qe


def synthetic(n_cliques=40, per_clique=4, k=40, seed=0):
    rng = np.random.default_rng(seed)
    clique_ids = [c for c in range(n_cliques) for _ in range(per_clique)]
    n = len(clique_ids)
    base = rng.normal(size=(n_cliques, 8))
    points = np.stack([base[c] + 0.45 * rng.normal(size=8) for c in clique_ids])
    d = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    idx = np.argsort(d, axis=1)[:, :k]
    dist = np.take_along_axis(d, idx, axis=1)
    is_match = np.array([[clique_ids[i] == clique_ids[j] for j in row]
                         for i, row in enumerate(idx)])
    return Shortlists(idx=idx, dist=dist, query_ids=np.arange(n),
                      is_match=is_match, clique_ids=clique_ids)


def test_shortlists_validate():
    s = synthetic()
    assert s.n_queries == s.idx.shape[0]
    assert s.row(0) == 0
    with pytest.raises(ValueError):
        Shortlists(idx=np.zeros((3, 2)), dist=np.zeros((3, 3)),
                   query_ids=np.arange(3))
    with pytest.raises(ValueError):          # distances must ascend
        Shortlists(idx=np.array([[1, 2]]), dist=np.array([[0.9, 0.1]]),
                   query_ids=np.array([0]))


def test_reranking_helps_on_synthetic_data():
    s = synthetic()
    rows = list(range(s.n_queries))
    orders = [np.argsort(np.arange(20)) for _ in rows]   # identity, as a control
    assert np.allclose(paired_delta_ap(s, orders, rows=rows, ), 0.0)

    from kreciprocal.rerank import Reranker, rerank_row
    r = Reranker(s, k1=10, k2=2)
    orders = [rerank_row(r, row, lam=0.5, depth=20) for row in rows]
    stats = summarise(paired_delta_ap(s, orders, rows=rows))
    assert stats["delta"] > 0, stats


def test_expansion_rules_agree_in_shape():
    s = synthetic()
    from kreciprocal.rerank import Reranker
    published = Reranker(s, k1=10, k2=2, expansion="published")
    zhong = Reranker(s, k1=10, k2=2, expansion="zhong")
    for t in range(0, s.n_queries, 7):
        a, b = published.expand(t), zhong.expand(t)
        assert set(published.reciprocal_set(t)) <= set(a)
        assert set(zhong.reciprocal_set(t)) <= set(b)


def test_query_expansion_runs():
    s = synthetic()
    rows = list(range(20))
    orders = [qe.query_expansion_row(s, row, e=3, alpha=3.0, depth=20) for row in rows]
    deltas = paired_delta_ap(s, orders, rows=rows)
    assert len(deltas) == len(rows)
