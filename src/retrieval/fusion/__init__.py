"""
Fusion algorithms: linear convex combination, RRF, and adaptive hybrid.
"""

from src.retrieval.fusion.linear import linear_fusion, min_max_normalize
from src.retrieval.fusion.rrf import reciprocal_rank_fusion
from src.retrieval.fusion.adaptive import adaptive_hybrid_fusion, compute_adaptive_alpha, NL_INDICATORS

__all__ = [
    "linear_fusion",
    "min_max_normalize",
    "reciprocal_rank_fusion",
    "adaptive_hybrid_fusion",
    "compute_adaptive_alpha",
    "NL_INDICATORS",
]
