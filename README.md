# k-reciprocal-vi

Training-free re-ranking for musical version identification. It takes the
shortlists a retrieval system already produces, checks which candidates rank the
query back, and re-sorts on that evidence. No training, no labels, no audio, and
no access to the encoder itself.

On the full Discogs-VI test set (110,082 tracks, 109,753 queries) it improves
every encoder we tried:

| first stage | baseline mAP | after re-ranking |
|---|---|---|
| CLEWS (official checkpoint) | 76.91 | **78.73** (+1.82 ± 0.04) |
| CLEWS, keeping rank 1 fixed | 76.91 | **78.84** (+1.93 ± 0.04) |
| Fish (official checkpoint) | 68.82 | 73.00 (+4.18 ± 0.05) |
| Discogs-VINet (official checkpoint) | 44.26 | 49.86 (+5.60 ± 0.05) |
| CLEWS (our reproduction) | 70.45 | 73.25 (+2.80 ± 0.04) |

It costs 15 to 35 CPU-minutes for the whole benchmark, single threaded, against
roughly 17 GPU-hours to encode the same collection.

Paper: *Training-free k-reciprocal re-ranking improves state-of-the-art musical
version identification* (submitted). See `CITATION.cff`.

## Install

```bash
pip install -e .              # numpy only
pip install -e ".[dumps]"     # + torch, to read the published .pt shortlists
```

Python 3.9 or later. No GPU at any point.

## Use it on your own encoder

The method needs one thing: for every track in your collection, the ranked list
of its nearest neighbours and their distances. Nothing else, and in particular
not the embeddings.

### The shortlist format

`Shortlists` holds one row per query. With `Q` queries over a gallery of `N`
tracks and `K` stored neighbours per query:

| field | shape | dtype | meaning |
|---|---|---|---|
| `idx` | `(Q, K)` | int64 | `idx[i, r]` is the gallery track id ranked `r` for query `i` |
| `dist` | `(Q, K)` | float32 | `dist[i, r]` its distance to the query, ascending along `r` |
| `query_ids` | `(Q,)` | int64 | the gallery track id that row `i` is the shortlist of |
| `is_match` | `(Q, K)` | bool | optional, true where the candidate is a real version. Evaluation only, never used to re-rank |
| `clique_ids` | `(N,)` | any hashable | optional, the work each gallery track belongs to. Evaluation only |

Conventions, all of which the constructor checks:

- **Track ids are gallery row numbers**, `0 .. N-1`, not names. Keep your own
  id-to-title mapping outside.
- **Distances ascend** along each row: `dist[i, 0]` is the nearest. Smaller means
  closer. The scale is yours; nothing assumes a range.
- **A query never contains itself.** Rank 0 is the closest *other* track.
- **Rows may cover only part of the gallery**, but reciprocity can only be
  tested for tracks that have a row of their own. A candidate without one is
  treated as far away, which weakens the method rather than breaking it. For the
  published results every gallery track has a row, so `Q == N`.
- **A pair absent from a shortlist** has no stored distance. It is given a large
  constant, so it contributes almost nothing to the features.

A three-track toy example, where track 0's nearest neighbour is track 2 at
distance 0.11:

```python
idx  = np.array([[2, 1], [0, 2], [0, 1]], dtype=np.int64)
dist = np.array([[0.11, 0.84], [0.42, 0.55], [0.11, 0.55]], dtype=np.float32)
shortlists = Shortlists(idx=idx, dist=dist, query_ids=np.arange(3))
```

Re-ranking needs `idx`, `dist` and `query_ids`. The other two fields only matter
if you want the paired estimator to report a gain.

The published dumps are PyTorch files with the same content under the names
`topk_indices`, `topk_distances`, `topk_is_match`, `query_indices`, `clique_ids`
and `aps`; `Shortlists.from_dump(path)` reads them.

```python
import numpy as np
from kreciprocal import Shortlists, rerank

# idx[i, r] is the track id at rank r for query i; dist[i, r] its distance.
shortlists = Shortlists(idx=idx, dist=dist, query_ids=np.arange(len(idx)))

order = rerank(shortlists, k1=30, k2=2, lam=0.5, depth=200)
```

`order[i]` is the re-ranked list of track ids for query `i`. Ranks past `depth`
keep their original position.

Two requirements worth stating plainly:

- **Every gallery track must appear as a query.** Reciprocity asks whether a
  candidate ranks the query back, which needs that candidate's own list. Running
  your evaluation with all tracks as queries usually costs nothing extra.
- **Shortlists must be deep enough.** We store 1000 neighbours per track and
  re-rank the top 200. Depth matters for the reciprocity tests, not just for the
  re-ranked window.

The defaults (`k1=30, k2=2, lam=0.5, depth=200`) were tuned once on the
Discogs-VI validation split and then left alone for every encoder and dataset in
the paper, so they are a reasonable starting point rather than a per-dataset
choice.

Query expansion baselines are available for comparison:

```python
from kreciprocal.qe import query_expansion
order = query_expansion(shortlists, e=5, alpha=3.0, depth=200)   # alpha=0 gives AQE
```

## Reproduce the paper

Two tiers, depending on how much you want to recompute.

### Tier 1: the numbers and figures, from per-query records (no download)

The per-query records are small enough to ship here (a few MB each). They hold,
for every query, the average precision before and after re-ranking, the rank of
the first correct hit, and the clique size.

```bash
python scripts/reproduce_tables.py     # Tables 1 and 3
python scripts/reproduce_figure2.py    # gain by clique size
```

This recomputes every reported gain, confidence interval and first-match
statistic from the stored per-query values.

### Tier 2: re-run the method, from the shortlist dumps (1.9 GB download)

```bash
python scripts/fetch_dumps.py          # from Zenodo, with checksums
python scripts/rerank_dump.py --dump data/dumps/discogsvi_test.pt \
    --k1 30 --k2 2 --lam 0.5 --depth 200
```

This runs the full method and should reproduce Tier 1's records exactly. Expect
15 to 35 CPU-minutes and about 6 GB of RAM.

### Figure 1

The figure in the paper shows the first six entries of each shortlist. The full
tables behind it, both rankings, the complete reciprocal sets with the
two-thirds test for each candidate, and the point cloud coordinates, are
produced by:

```bash
python scripts/figure1_full.py --query "Watch What Happens"
```

## How it works

Each function names the part of the paper it implements.

| code | paper |
|---|---|
| `reciprocal_set` | Sec. 3.2, the mutual-neighbour set R(x, k1) |
| `expand` | Sec. 3.2, Eq. 1, the expanded set R*(x) |
| `feature` | Sec. 3.2, Eq. 2, the sparse k-reciprocal feature |
| `smooth` | Sec. 3.2, Eq. 3, local query expansion over k2 neighbours |
| `jaccard` | Sec. 3.2, Eq. 4, weighted Jaccard distance |
| `final_distance` | Sec. 3.2, Eq. 5, the mix with the original distance |
| `paired_delta_ap` | Sec. 3.3, Eqs. 6 and 7, the exact paired estimator |
| `qe.query_expansion` | Sec. 4.3, AQE and alpha-QE |

## What this does not do

- It does not help when a work has only one other version in the collection. On
  two-version cliques the gain is negative (-1.91 mAP points). The gain grows
  with clique size, and that is the single best predictor of whether it is worth
  running.
- It is transductive. It needs the shortlists of a fixed collection, which suits
  offline deduplication rather than streaming queries.
- It slightly degrades first-match statistics on a strong first stage. Keeping
  rank 1 fixed removes that cost and gains more mAP, at the price of never
  correcting a wrong top hit.

## What is not in this repository

No audio, no encoder weights, no training code. The encoders are published
separately: [CLEWS](https://github.com/sony/clews),
[Discogs-VINet](https://github.com/raraz15/Discogs-VINet), and Fish. The dataset
is [Discogs-VI](https://mtg.github.io/discogs-vi-dataset/).

## Licence

Code is MIT. The released dumps and records derive from Discogs-VI metadata and
are CC BY-NC-SA 4.0, so they are attribution, non-commercial and share-alike.
See `LICENSE-DATA.md`.
