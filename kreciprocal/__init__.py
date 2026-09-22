"""Training-free k-reciprocal re-ranking for musical version identification."""
from .shortlists import Shortlists
from .rerank import Reranker, rerank, rerank_row, jaccard
from .evaluate import paired_delta_ap, summarise, average_precision
from . import qe

__all__ = ["Shortlists", "Reranker", "rerank", "rerank_row", "jaccard",
           "paired_delta_ap", "summarise", "average_precision", "qe"]
__version__ = "0.1.0"
