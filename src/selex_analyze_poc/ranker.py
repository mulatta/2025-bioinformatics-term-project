"""Candidate rankers for SELEX PoC.

Implementations:
    - ClusterEnrichmentRanker: Embedding + clustering + enrichment scoring
    - AptaTraceRanker: Wrapper for AptaTRACE (TODO)
"""

import numpy as np
import polars as pl
import umap
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from selex_analyze_poc.base import CandidateRanker, FeatureExtractor, RankingResult


class ClusterEnrichmentRanker(CandidateRanker):
    """Cluster-based enrichment ranking.
    ...
    Args:
        ...
        device: "cpu" or "cuda".
    """

    def __init__(
        self,
        extractor: FeatureExtractor,
        n_clusters: int = 20,
        umap_dim: int = 10,
        use_hdbscan: bool = False,
        min_cluster_size: int = 50,
        device: str = "cpu",
    ):
        self.extractor = extractor
        self.n_clusters = n_clusters
        self.umap_dim = umap_dim
        self.use_hdbscan = use_hdbscan
        self.min_cluster_size = min_cluster_size
        self.device = device

        if self.device == "auto":
            try:
                import cupy

                self.device = "cuda"
            except ImportError:
                self.device = "cpu"

        if self.device == "cuda":
            try:
                import cuml
                import cupy as xp
                from cuml.preprocessing import StandardScaler as GpuStandardScaler

                self.xp = xp
                self.cuml = cuml
                self.GpuStandardScaler = GpuStandardScaler
            except ImportError:
                print(
                    "Warning: device='cuda' but cuml/cupy not found. Falling back to cpu."
                )
                self.device = "cpu"
                self.xp = np
        else:
            self.xp = np

        self.embeddings_ = None
        self.umap_embeddings_ = None
        self.cluster_labels_ = None
        self.cluster_scores_ = None

    def _reduce_dimensionality(self, embeddings: np.ndarray) -> np.ndarray:
        """Reduce embedding dimensionality with UMAP."""
        if self.device == "cuda":
            reducer = self.cuml.UMAP(
                n_components=self.umap_dim,
                n_neighbors=15,
                min_dist=0.1,
                metric="cosine",
                random_state=42,
            )
            return reducer.fit_transform(embeddings)

        reducer = umap.UMAP(
            n_components=self.umap_dim,
            n_neighbors=15,
            min_dist=0.1,
            metric="cosine",
            random_state=42,
        )
        return reducer.fit_transform(embeddings)

    def _cluster(self, embeddings: np.ndarray) -> np.ndarray:
        """Cluster embeddings."""
        if self.device == "cuda":
            scaler = self.GpuStandardScaler()
            scaled = scaler.fit_transform(embeddings)

            if self.use_hdbscan:
                clusterer = self.cuml.DBSCAN(
                    min_samples=10,
                    eps=1.0,  # This might need tuning
                )
                labels = clusterer.fit_predict(scaled)
            else:
                clusterer = self.cuml.KMeans(
                    n_clusters=self.n_clusters,
                    random_state=42,
                    n_init=10,
                )
                labels = clusterer.fit_predict(scaled)
            return self.xp.asnumpy(labels)

        scaler = StandardScaler()
        scaled = scaler.fit_transform(embeddings)

        if self.use_hdbscan:
            import hdbscan

            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=10,
                metric="euclidean",
            )
            labels = clusterer.fit_predict(scaled)
        else:
            clusterer = KMeans(
                n_clusters=self.n_clusters,
                random_state=42,
                n_init=10,
            )
            labels = clusterer.fit_predict(scaled)

        return labels

    def _compute_cluster_enrichment(
        self,
        data: pl.DataFrame,
        cluster_labels: np.ndarray,
    ) -> dict[int, float]:
        """Compute enrichment score for each cluster.

        Enrichment = (fraction in final round) / (fraction in round 0)

        Returns:
            Dict mapping cluster_id -> enrichment_score
        """
        df = data.with_columns(pl.Series("cluster", cluster_labels))

        rounds = df["round"].unique().sort().to_list()
        if len(rounds) < 2:
            # No enrichment without multiple rounds
            unique_clusters = np.unique(cluster_labels[cluster_labels >= 0])
            return dict.fromkeys(unique_clusters, 1.0)

        initial_round = rounds[0]
        final_round = rounds[-1]

        cluster_scores = {}

        for cluster_id in np.unique(cluster_labels):
            if cluster_id < 0:  # Skip noise cluster from HDBSCAN
                continue

            cluster_mask = df["cluster"] == cluster_id

            initial_count = df.filter(cluster_mask & (df["round"] == initial_round))[
                "count"
            ].sum()
            initial_total = df.filter(df["round"] == initial_round)["count"].sum()
            initial_frac = initial_count / initial_total if initial_total > 0 else 0

            final_count = df.filter(cluster_mask & (df["round"] == final_round))[
                "count"
            ].sum()
            final_total = df.filter(df["round"] == final_round)["count"].sum()
            final_frac = final_count / final_total if final_total > 0 else 0

            enrichment = (final_frac + 1e-6) / (initial_frac + 1e-6)
            cluster_scores[cluster_id] = enrichment

        return cluster_scores

    def rank(self, data: pl.DataFrame, seeds: set[str]) -> RankingResult:
        """Rank candidates by cluster enrichment.

        Args:
            data: DataFrame with columns [sequence, round, count]
            seeds: Set of known seed sequences (ground truth)

        Returns:
            RankingResult with enrichment-based scores
        """
        self.seeds = seeds
        sequences = data["sequence"].unique().to_list()

        print(f"Extracting embeddings for {len(sequences)} sequences...")
        self.embeddings_ = self.extractor.extract(sequences)

        print(f"Reducing to {self.umap_dim} dimensions...")
        self.umap_embeddings_ = self._reduce_dimensionality(self.embeddings_)

        print("Clustering...")
        self.cluster_labels_ = self._cluster(self.umap_embeddings_)

        seq_to_idx = {seq: i for i, seq in enumerate(sequences)}
        seq_cluster = data["sequence"].map_elements(
            lambda s: self.cluster_labels_[seq_to_idx.get(s, -1)], return_dtype=pl.Int64
        )
        data_with_clusters = data.with_columns(seq_cluster.alias("cluster"))

        self.cluster_scores_ = self._compute_cluster_enrichment(
            data_with_clusters, data_with_clusters["cluster"].to_numpy()
        )

        scores = np.array(
            [
                self.cluster_scores_.get(self.cluster_labels_[i], 0.0)
                for i in range(len(sequences))
            ]
        )

        return RankingResult(
            sequences=sequences,
            scores=scores,
            method=f"ClusterEnrichment({self.extractor.name})",
        )

    def get_visualization_data(self) -> pl.DataFrame:
        """Get data for UMAP visualization."""
        if self.umap_embeddings_ is None:
            raise ValueError("Must call rank() first")

        if self.umap_embeddings_.shape[1] > 2:
            if self.device == "cuda":
                reducer = self.cuml.UMAP(
                    n_components=2,
                    n_neighbors=15,
                    min_dist=0.1,
                    random_state=42,
                )
                # Need to convert umap_embeddings_ from numpy back to cupy for this step
                umap_embeddings_cp = self.xp.asarray(self.umap_embeddings_)
                coords = reducer.fit_transform(umap_embeddings_cp)
                coords = self.xp.asnumpy(coords)
            else:
                reducer = umap.UMAP(
                    n_components=2,
                    n_neighbors=15,
                    min_dist=0.1,
                    random_state=42,
                )
                coords = reducer.fit_transform(self.umap_embeddings_)
        else:
            coords = self.umap_embeddings_

        if hasattr(coords, "get"):
            coords = coords.get()

        sequences = self.extractor.last_extracted_sequences

        return pl.DataFrame(
            {
                "umap_x": coords[:, 0],
                "umap_y": coords[:, 1],
                "cluster": self.cluster_labels_,
                "cluster_score": [
                    self.cluster_scores_.get(c, 0.0) for c in self.cluster_labels_
                ],
                "is_seed": [s in self.seeds for s in sequences],
            }
        )

    @property
    def name(self) -> str:
        cluster_method = (
            "HDBSCAN" if self.use_hdbscan else f"KMeans(k={self.n_clusters})"
        )
        return f"ClusterEnrichment({self.extractor.name}, {cluster_method})"


class CountBasedRanker(CandidateRanker):
    """Simple count-based ranking (baseline).

    Ranks sequences by their count fold-change from round 0 to final round.
    """

    def rank(self, data: pl.DataFrame) -> RankingResult:
        """Rank by count fold-change."""
        rounds = data["round"].unique().sort().to_list()
        if len(rounds) < 2:
            raise ValueError("Need at least 2 rounds for fold-change")

        initial_round = rounds[0]
        final_round = rounds[-1]

        rpm_data = (
            data.group_by(["sequence", "round"])
            .agg(pl.col("count").sum())
            .with_columns(
                (pl.col("count") / pl.col("count").sum().over("round") * 1e6).alias(
                    "rpm"
                )
            )
        )

        initial_rpm = rpm_data.filter(pl.col("round") == initial_round).select(
            ["sequence", pl.col("rpm").alias("rpm_initial")]
        )
        final_rpm = rpm_data.filter(pl.col("round") == final_round).select(
            ["sequence", pl.col("rpm").alias("rpm_final")]
        )

        merged = (
            initial_rpm.join(final_rpm, on="sequence", how="outer")
            .fill_null(0)
            .with_columns(
                ((pl.col("rpm_final") + 1) / (pl.col("rpm_initial") + 1)).alias(
                    "fold_change"
                )
            )
        )

        return RankingResult(
            sequences=merged["sequence"].to_list(),
            scores=merged["fold_change"].to_numpy(),
            method="CountFoldChange",
        )

    @property
    def name(self) -> str:
        return "CountFoldChange"
