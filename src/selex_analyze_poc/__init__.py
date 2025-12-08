"""
Usage:
    python -m selex_analyze_poc.compare_embeddings \
        --data-dir ./data/ \
        --seeds-file ./data/seeds.txt \
        --output-dir ./results/
"""

from selex_analyze_poc.base import (
    CandidateRanker,
    EmbeddingResult,
    FeatureExtractor,
    RankingResult,
)
from selex_analyze_poc.evaluation import (
    EvaluationResult,
    compare_methods,
    evaluate_seed_recovery,
    load_seeds,
    load_selex_data,
    plot_embedding_enrichment_analysis,
)
from selex_analyze_poc.extractors import (
    BPPMGATExtractor,
    CombinedExtractor,
    KmerExtractor,
    OneHotExtractor,
    RNAFMExtractor,
    ThermoExtractor,
)
from selex_analyze_poc.ranker import ClusterEnrichmentRanker, CountBasedRanker

__all__ = [
    "EmbeddingResult",
    "RankingResult",
    "FeatureExtractor",
    "CandidateRanker",
    "OneHotExtractor",
    "KmerExtractor",
    "ThermoExtractor",
    "CombinedExtractor",
    "RNAFMExtractor",
    "BPPMGATExtractor",
    "ClusterEnrichmentRanker",
    "CountBasedRanker",
    "EvaluationResult",
    "evaluate_seed_recovery",
    "compare_methods",
    "plot_embedding_enrichment_analysis",
    "load_seeds",
    "load_selex_data",
]
