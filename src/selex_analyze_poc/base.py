"""Abstract base classes for SELEX PoC."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass
class EmbeddingResult:
    """Embedding result with metadata.

    Attributes:
        sequences: List of RNA sequences
        embeddings: Embedding matrix (N, D)
        metadata: Additional info (round, count, etc.)
    """

    sequences: list[str]
    embeddings: np.ndarray  # (N, D)
    metadata: pl.DataFrame | None = None

    def __post_init__(self) -> None:
        if len(self.sequences) != self.embeddings.shape[0]:
            raise ValueError(
                f"Mismatch: {len(self.sequences)} sequences, "
                f"{self.embeddings.shape[0]} embeddings"
            )


@dataclass
class RankingResult:
    """Ranking result from any ranker.

    Attributes:
        sequences: List of sequences in original order
        scores: Score array (N,) - higher = better candidate
        method: Name of the ranking method
    """

    sequences: list[str]
    scores: np.ndarray  # (N,)
    method: str

    def __post_init__(self) -> None:
        if len(self.sequences) != len(self.scores):
            raise ValueError(
                f"Mismatch: {len(self.sequences)} sequences, {len(self.scores)} scores"
            )

    def top_k(self, k: int) -> list[str]:
        """Return top-K sequences by score."""
        sorted_indices = np.argsort(self.scores)[::-1]
        return [self.sequences[i] for i in sorted_indices[:k]]

    def to_dataframe(self) -> pl.DataFrame:
        """Convert to Polars DataFrame."""
        return pl.DataFrame(
            {
                "sequence": self.sequences,
                "score": self.scores,
                "method": [self.method] * len(self.sequences),
            }
        ).sort("score", descending=True)


class FeatureExtractor(ABC):
    """Abstract base class for feature/embedding extraction.

    Implementations:
        - Stage 1: RNAFMExtractor (pretrained, no training)
        - Stage 2: BPPMGATExtractor (structure-aware)
    """

    def __init__(self) -> None:
        self._last_extracted_sequences: list[str] | None = None

    @property
    def last_extracted_sequences(self) -> list[str]:
        if self._last_extracted_sequences is None:
            raise ValueError("No sequences have been extracted yet.")
        return self._last_extracted_sequences

    @abstractmethod
    def extract(self, sequences: list[str]) -> np.ndarray:
        """Extract embeddings from sequences.

        Args:
            sequences: List of RNA sequences

        Returns:
            Embedding matrix (N, D)
        """
        pass

    @property
    @abstractmethod
    def dim(self) -> int:
        """Output embedding dimension."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Extractor name for logging."""
        pass


class CandidateRanker(ABC):
    """Abstract base class for candidate ranking.

    Implementations:
        - ClusterEnrichmentRanker: Cluster-based scoring
        - AptaTraceRanker: K-mer context shifting (wrapper)
    """

    @abstractmethod
    def rank(self, data: pl.DataFrame) -> RankingResult:
        """Rank candidates from SELEX data.

        Args:
            data: DataFrame with columns [sequence, round, count, ...]

        Returns:
            RankingResult with scores for each sequence
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Ranker name for logging."""
        pass
