"""k-reciprocal re-ranking over shortlists (paper, Sec. 3.2).

The steps, in the order the paper introduces them:

    reciprocal_set   the mutual neighbours R(x, k1)
    expand           the expanded set R*(x), Eq. 1
    feature          the sparse k-reciprocal feature f_x, Eq. 2
    smooth           local query expansion over k2 neighbours, Eq. 3
    jaccard          the weighted Jaccard distance d_J, Eq. 4
    rerank           the mix with the original distance and the re-sort, Eq. 5
"""
import numpy as np

# Distance assigned to a pair that appears in no stored shortlist. Large enough
# that the pair carries almost no weight in a feature.
ABSENT = 5.0


class Reranker:
    """Holds the caches that make re-ranking cheap across many queries.

    expansion selects how R* is built:
      "standard"  the rule of Zhong et al. (CVPR 2017), which is Eq. 1 of the
                  paper and what every reported number uses. Both the forward
                  and backward windows are ceil(k1 / 2), and only j is the
                  anchor.
      "loose"     a slightly wider variant: a candidate also joins when it ranks
                  back the feature's owner rather than only j, and the backward
                  test uses the full k1 window. See the README for the measured
                  difference.
    """

    def __init__(self, shortlists, k1=30, k2=2, expansion="standard", absent=ABSENT):
        expansion = {"zhong": "standard", "published": "loose"}.get(expansion, expansion)
        if expansion not in ("standard", "loose"):
            raise ValueError("expansion must be 'standard' or 'loose'")
        self.s = shortlists
        self.k1 = k1
        self.k2 = k2
        self.expansion = expansion
        self.absent = absent
        self.khalf = max(1, int(round(k1 / 2)))

        # Neighbour sets used for the reciprocity tests. The published rule tests
        # membership in the full k1 window; Zhong's uses the half window for the
        # candidate step, so both are cached.
        wide = max(k1, self.khalf + 1)
        self._wide = {int(t): set(self.s.idx[r, :wide].tolist())
                      for r, t in enumerate(self.s.query_ids)}
        self._half = {int(t): set(self.s.idx[r, :self.khalf].tolist())
                      for r, t in enumerate(self.s.query_ids)}
        self._features = {}

    def distance(self, track, other):
        """Stored distance between two tracks, or the absent constant."""
        if track == other:
            return 0.0
        r = self.s.row(track)
        if r is None:
            return self.absent
        hit = np.where(self.s.idx[r] == other)[0]
        return float(self.s.dist[r, hit[0]]) if len(hit) else self.absent

    def reciprocal_set(self, track):
        """R(x, k1): the neighbours of x that rank x back (Sec. 3.2)."""
        r = self.s.row(track)
        if r is None:
            return []
        return [int(j) for j in self.s.idx[r, :self.k1]
                if track in self._wide.get(int(j), ())]

    def expand(self, track):
        """R*(x): the reciprocal set plus the neighbourhoods it vouches for (Eq. 1)."""
        recip = self.reciprocal_set(track)
        if not recip:
            return [track]
        out = set(recip)
        for j in recip:
            jr = self.s.row(j)
            if jr is None:
                continue
            candidates = self.s.idx[jr, :self.khalf]
            if self.expansion == "loose":
                members = [int(m) for m in candidates
                           if track in self._wide.get(int(m), ())
                           or j in self._wide.get(int(m), ())]
            else:
                members = [int(m) for m in candidates
                           if j in self._half.get(int(m), ())]
            if members and len(set(members) & out) > (2 / 3) * len(members):
                out |= set(members)
        return sorted(out)

    def feature(self, track):
        """f_x: softmax over negated distances on R*(x), zero elsewhere (Eq. 2)."""
        if track in self._features:
            return self._features[track]
        members = self.expand(track)
        weights = np.exp(-np.array([self.distance(track, m) for m in members]))
        weights /= weights.sum()
        out = (np.asarray(members, dtype=np.int64), weights)
        self._features[track] = out
        return out

    def smooth(self, track):
        """Average a feature with those of its k2 nearest neighbours (Eq. 3)."""
        if self.k2 <= 0:
            return self.feature(track)
        r = self.s.row(track)
        members = [track] + ([int(j) for j in self.s.idx[r, :self.k2]] if r is not None else [])
        pooled = {}
        for m in members:
            ids, weights = self.feature(m)
            for i, w in zip(ids.tolist(), weights.tolist()):
                pooled[i] = pooled.get(i, 0.0) + w / len(members)
        ids = np.array(sorted(pooled), dtype=np.int64)
        return ids, np.array([pooled[i] for i in ids.tolist()])


def jaccard(a, b):
    """Weighted Jaccard distance between two sparse features (Eq. 4)."""
    ia, wa = a
    ib, wb = b
    i = j = 0
    lo = hi = 0.0
    while i < len(ia) and j < len(ib):
        if ia[i] == ib[j]:
            x, y = wa[i], wb[j]
            lo += min(x, y)
            hi += max(x, y)
            i += 1
            j += 1
        elif ia[i] < ib[j]:
            hi += wa[i]
            i += 1
        else:
            hi += wb[j]
            j += 1
    hi += wa[i:].sum() + wb[j:].sum()
    return 1.0 - lo / hi if hi > 0 else 1.0


def rerank_row(reranker, row, lam=0.5, depth=200, max_distance=None):
    """Re-ranked order of one query's top-`depth` candidates (Eq. 5).

    Returns the permutation applied to those candidates, so that
    idx[row, :depth][order] is the new ranking.
    """
    s = reranker.s
    depth = min(depth, s.depth)
    query = int(s.query_ids[row])
    dmax = s.max_distance if max_distance is None else max_distance
    fq = reranker.smooth(query)
    candidates = s.idx[row, :depth]
    dj = np.array([jaccard(fq, reranker.smooth(int(c))) for c in candidates])
    mixed = (1 - lam) * dj + lam * (s.dist[row, :depth] / dmax)
    return np.argsort(mixed, kind="stable")


def rerank(shortlists, k1=30, k2=2, lam=0.5, depth=200, expansion="standard", rows=None):
    """Re-rank every query and return the new track ids per row.

    Ranks beyond `depth` keep their original position.
    """
    r = Reranker(shortlists, k1=k1, k2=k2, expansion=expansion)
    rows = range(shortlists.n_queries) if rows is None else rows
    out = []
    for row in rows:
        order = rerank_row(r, row, lam=lam, depth=depth)
        head = shortlists.idx[row, :depth][order]
        out.append(np.concatenate([head, shortlists.idx[row, depth:]]))
    return np.stack(out)
