"""The input format: one ranked shortlist per query.

The method never sees embeddings. It reads the ranked neighbour lists a
retrieval system already produces, which is what makes it encoder-agnostic
(paper, Sec. 3.1).
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Shortlists:
    """Ranked neighbours for every query, plus optional labels for evaluation.

    idx        (Q, K) int64   gallery track id at each rank
    dist       (Q, K) float32 distance to the query, ascending along each row
    query_ids  (Q,)   int64   the gallery track each row belongs to
    is_match   (Q, K) bool    optional, true where the candidate is a real
                              version. Used only to score, never to re-rank.
    clique_ids (N,)           optional, the work each gallery track belongs to.

    Track ids are gallery row numbers. A query never contains itself. Rows may
    cover part of the gallery only, but a candidate without a row of its own
    cannot be tested for reciprocity, so coverage weakens the method.
    """

    idx: np.ndarray
    dist: np.ndarray
    query_ids: np.ndarray
    is_match: np.ndarray = None
    clique_ids: list = field(default=None)

    def __post_init__(self):
        self.idx = np.asarray(self.idx, dtype=np.int64)
        self.dist = np.asarray(self.dist, dtype=np.float32)
        self.query_ids = np.asarray(self.query_ids, dtype=np.int64)
        if self.idx.shape != self.dist.shape:
            raise ValueError(f"idx {self.idx.shape} and dist {self.dist.shape} differ")
        if self.idx.ndim != 2:
            raise ValueError("idx must be (queries, neighbours)")
        if len(self.query_ids) != len(self.idx):
            raise ValueError("one query id per row is required")
        if self.is_match is not None:
            self.is_match = np.asarray(self.is_match, dtype=bool)
            if self.is_match.shape != self.idx.shape:
                raise ValueError("is_match must match idx in shape")
        steps = np.diff(self.dist, axis=1)
        if steps.size and steps.min() < -1e-4:
            raise ValueError("distances must ascend along each row")

        # row_of[t] is the row holding track t's own shortlist, or -1.
        n = max(int(self.query_ids.max()), int(self.idx.max())) + 1
        self.row_of = np.full(n, -1, dtype=np.int64)
        self.row_of[self.query_ids] = np.arange(len(self.query_ids))

    @property
    def n_queries(self):
        return len(self.idx)

    @property
    def depth(self):
        return self.idx.shape[1]

    @property
    def max_distance(self):
        """Largest distance stored anywhere, used to normalise d in Eq. 5."""
        return float(self.dist.max())

    def row(self, track):
        """The row index of a track's own shortlist, or None if it has none."""
        r = int(self.row_of[track]) if track < len(self.row_of) else -1
        return None if r < 0 else r

    def positives_per_query(self):
        """Number of other versions each gallery track has, from clique_ids."""
        if self.clique_ids is None:
            raise ValueError("clique_ids are required to count positives")
        from collections import Counter
        counts = Counter(self.clique_ids)
        return np.array([counts[c] - 1 for c in self.clique_ids])

    @classmethod
    def from_dump(cls, path):
        """Read a published .pt dump (requires torch)."""
        import torch
        d = torch.load(path, map_location="cpu", weights_only=False)
        return cls(
            idx=d["topk_indices"].long().numpy(),
            dist=d["topk_distances"].float().numpy(),
            query_ids=np.asarray(d["query_indices"]).astype(np.int64),
            is_match=d["topk_is_match"].bool().numpy(),
            clique_ids=list(d["clique_ids"]),
        )
