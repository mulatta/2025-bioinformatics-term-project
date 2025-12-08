"""Evaluation metrics for SELEX PoC.

Primary metric: Seed Recovery
- Precision@K: What fraction of top-K are seeds?
- Recall@K: What fraction of seeds are in top-K?
- AUC-ROC: Overall ranking quality
"""

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from selex_analyze_poc.base import RankingResult


@dataclass
class EvaluationResult:
    """Evaluation result for a single method.

    Attributes:
        method: Name of the ranking method
        precision_at_k: Dict mapping k -> precision
        recall_at_k: Dict mapping k -> recall
        auc_roc: Area under ROC curve
        n_sequences: Total number of sequences
        n_seeds: Number of seed sequences
    """

    method: str
    precision_at_k: dict[int, float] = field(default_factory=dict)
    recall_at_k: dict[int, float] = field(default_factory=dict)
    auc_roc: float = 0.0
    n_sequences: int = 0
    n_seeds: int = 0

    def to_dict(self) -> dict:
        """Convert to flat dict for comparison."""
        result = {
            "method": self.method,
            "auc_roc": self.auc_roc,
            "n_sequences": self.n_sequences,
            "n_seeds": self.n_seeds,
        }
        for k, v in self.precision_at_k.items():
            result[f"precision@{k}"] = v
        for k, v in self.recall_at_k.items():
            result[f"recall@{k}"] = v
        return result


def evaluate_seed_recovery(
    ranking: RankingResult,
    seeds: set[str],
    k_values: list[int] | None = None,
) -> EvaluationResult:
    """Evaluate ranking by seed recovery.

    Args:
        ranking: RankingResult from any ranker
        seeds: Set of known seed sequences (ground truth)
        k_values: Top-K values to evaluate (default: [10, 50, 100, 500])

    Returns:
        EvaluationResult with precision@K, recall@K, and AUC-ROC
    """
    if k_values is None:
        k_values = [10, 50, 100, 500]

    sorted_indices = np.argsort(ranking.scores)[::-1]
    sorted_seqs = [ranking.sequences[i] for i in sorted_indices]

    labels = np.array([1 if seq in seeds else 0 for seq in ranking.sequences])

    try:
        auc = roc_auc_score(labels, ranking.scores)
    except ValueError:
        auc = 0.5

    precision_at_k = {}
    recall_at_k = {}

    for k in k_values:
        if k > len(sorted_seqs):
            k = len(sorted_seqs)

        top_k = set(sorted_seqs[:k])
        hits = len(top_k & seeds)

        precision_at_k[k] = hits / k if k > 0 else 0
        recall_at_k[k] = hits / len(seeds) if len(seeds) > 0 else 0

    return EvaluationResult(
        method=ranking.method,
        precision_at_k=precision_at_k,
        recall_at_k=recall_at_k,
        auc_roc=auc,
        n_sequences=len(ranking.sequences),
        n_seeds=len(seeds),
    )


def compare_methods(
    results: list[EvaluationResult],
    output_dir: Path | None = None,
    umap_data: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Compare multiple evaluation results.

    Args:
        results: List of EvaluationResult from different methods
        output_dir: Optional directory to save outputs
        umap_data: Optional DataFrame with UMAP coordinates for visualization

    Returns:
        Comparison DataFrame
    """
    rows = [r.to_dict() for r in results]
    df = pl.DataFrame(rows)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        df.write_csv(output_dir / "comparison_table.csv")

        _plot_precision_recall(results, output_dir)
        _plot_auc_comparison(results, output_dir)

        if umap_data is not None:
            _plot_umap_from_data(umap_data, output_dir)
        else:
            _plot_umap(output_dir)

        print(f"Results saved to {output_dir}")

    return df


def _plot_precision_recall(results: list[EvaluationResult], output_dir: Path) -> None:
    """Plot precision and recall at K for each method."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for result in results:
        k_values = sorted(result.precision_at_k.keys())
        precisions = [result.precision_at_k[k] for k in k_values]
        recalls = [result.recall_at_k[k] for k in k_values]

        ax1.plot(k_values, precisions, marker="o", label=result.method)
        ax2.plot(k_values, recalls, marker="o", label=result.method)

    ax1.set_xlabel("K")
    ax1.set_ylabel("Precision@K")
    ax1.set_title("Precision at Top-K")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("K")
    ax2.set_ylabel("Recall@K")
    ax2.set_title("Recall at Top-K")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "precision_recall_curve.png", dpi=150)
    plt.close()


def _plot_auc_comparison(results: list[EvaluationResult], output_dir: Path) -> None:
    """Plot AUC-ROC comparison bar chart."""
    fig, ax = plt.subplots(figsize=(8, 5))

    methods = [r.method for r in results]
    aucs = [r.auc_roc for r in results]

    colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))
    bars = ax.bar(methods, aucs, color=colors)

    ax.set_ylabel("AUC-ROC")
    ax.set_title("Seed Recovery: AUC-ROC Comparison")
    ax.set_ylim(0, 1)

    for bar, auc in zip(bars, aucs, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{auc:.3f}",
            ha="center",
            fontsize=10,
        )

    ax.axhline(y=0.5, color="gray", linestyle="--", label="Random", alpha=0.7)
    ax.legend()

    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "auc_comparison.png", dpi=150)
    plt.close()


def _plot_umap(output_dir: Path) -> None:
    """Plot UMAP visualization from saved parquet file."""
    umap_data_path = output_dir / "umap_clusters.parquet"
    if not umap_data_path.exists():
        print(f"Warning: UMAP data not found at {umap_data_path}. Skipping UMAP plot.")
        return

    df_umap = pl.read_parquet(umap_data_path)
    _plot_umap_from_data(df_umap, output_dir)


def _plot_umap_from_data(df_umap: pl.DataFrame, output_dir: Path) -> None:
    """Plot UMAP visualization of clusters, highlighting seeds.

    Creates two plots:
    1. Clusters colored by cluster ID with seeds highlighted
    2. Clusters colored by enrichment score with seeds highlighted
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    ax1 = axes[0]
    non_seeds = df_umap.filter(~pl.col("is_seed"))
    ax1.scatter(
        non_seeds["umap_x"],
        non_seeds["umap_y"],
        c=non_seeds["cluster"],
        cmap="tab20",
        s=10,
        alpha=0.5,
        edgecolor="none",
    )

    seeds = df_umap.filter(pl.col("is_seed"))
    if len(seeds) > 0:
        ax1.scatter(
            seeds["umap_x"],
            seeds["umap_y"],
            c="red",
            s=80,
            alpha=0.9,
            label=f"Seeds (n={len(seeds)})",
            marker="*",
            edgecolor="black",
            linewidth=0.5,
        )

    ax1.set_title("UMAP: Clusters (Seeds in Red)")
    ax1.set_xlabel("UMAP 1")
    ax1.set_ylabel("UMAP 2")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    if "cluster_score" in df_umap.columns:
        non_seeds = df_umap.filter(~pl.col("is_seed"))
        scatter2 = ax2.scatter(
            non_seeds["umap_x"],
            non_seeds["umap_y"],
            c=non_seeds["cluster_score"],
            cmap="plasma",
            s=10,
            alpha=0.6,
            edgecolor="none",
        )
        plt.colorbar(scatter2, ax=ax2, label="Cluster Enrichment Score")

        if len(seeds) > 0:
            ax2.scatter(
                seeds["umap_x"],
                seeds["umap_y"],
                c="lime",
                s=80,
                alpha=0.9,
                label=f"Seeds (n={len(seeds)})",
                marker="*",
                edgecolor="black",
                linewidth=0.5,
            )

        ax2.set_title("UMAP: Enrichment Score (Seeds in Green)")
        ax2.set_xlabel("UMAP 1")
        ax2.set_ylabel("UMAP 2")
        ax2.legend(loc="upper right")
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(
            0.5,
            0.5,
            "No cluster_score column",
            ha="center",
            va="center",
            transform=ax2.transAxes,
        )
        ax2.set_title("Enrichment Score (N/A)")

    plt.tight_layout()
    plt.savefig(output_dir / "umap_visualization.png", dpi=150)
    plt.close()


def plot_embedding_enrichment_analysis(
    df_umap: pl.DataFrame,
    output_dir: Path,
    title_prefix: str = "",
) -> None:
    """Plot analysis of embedding space vs enrichment for PoC validation.

    This visualization helps answer: "Do foundation model embeddings
    naturally separate enriched sequences from non-enriched ones?"

    Args:
        df_umap: DataFrame with umap_x, umap_y, is_seed, cluster_score columns
        output_dir: Directory to save plots
        title_prefix: Optional prefix for plot titles (e.g., "RNA-FM" or "Physics-based")
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    ax1 = axes[0, 0]
    non_seeds = df_umap.filter(~pl.col("is_seed"))
    seeds = df_umap.filter(pl.col("is_seed"))

    ax1.scatter(
        non_seeds["umap_x"],
        non_seeds["umap_y"],
        c="lightgray",
        s=5,
        alpha=0.3,
        label=f"Non-seeds (n={len(non_seeds)})",
    )
    if len(seeds) > 0:
        ax1.scatter(
            seeds["umap_x"],
            seeds["umap_y"],
            c="red",
            s=30,
            alpha=0.8,
            label=f"Seeds (n={len(seeds)})",
        )
    ax1.set_title(f"{title_prefix}Seed Distribution in Embedding Space")
    ax1.set_xlabel("UMAP 1")
    ax1.set_ylabel("UMAP 2")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    if len(seeds) > 0:
        h, xedges, yedges = np.histogram2d(
            seeds["umap_x"].to_numpy(),
            seeds["umap_y"].to_numpy(),
            bins=20,
        )
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        im = ax2.imshow(
            h.T,
            origin="lower",
            extent=extent,
            cmap="Reds",
            aspect="auto",
            interpolation="gaussian",
        )
        plt.colorbar(im, ax=ax2, label="Seed Count")
        ax2.set_title(f"{title_prefix}Seed Density Heatmap")
    else:
        ax2.text(
            0.5, 0.5, "No seeds", ha="center", va="center", transform=ax2.transAxes
        )
    ax2.set_xlabel("UMAP 1")
    ax2.set_ylabel("UMAP 2")

    ax3 = axes[1, 0]
    if "cluster_score" in df_umap.columns:
        seed_scores = seeds["cluster_score"].to_numpy() if len(seeds) > 0 else []
        non_seed_scores = non_seeds["cluster_score"].to_numpy()

        ax3.hist(
            non_seed_scores,
            bins=30,
            alpha=0.5,
            label="Non-seeds",
            color="gray",
            density=True,
        )
        if len(seed_scores) > 0:
            ax3.hist(
                seed_scores,
                bins=30,
                alpha=0.7,
                label="Seeds",
                color="red",
                density=True,
            )
            ax3.axvline(
                np.median(seed_scores),
                color="red",
                linestyle="--",
                label=f"Seed median: {np.median(seed_scores):.2f}",
            )
        ax3.axvline(
            np.median(non_seed_scores),
            color="gray",
            linestyle="--",
            label=f"Non-seed median: {np.median(non_seed_scores):.2f}",
        )
        ax3.set_title(f"{title_prefix}Cluster Score Distribution")
        ax3.set_xlabel("Cluster Enrichment Score")
        ax3.set_ylabel("Density")
        ax3.legend()
    else:
        ax3.text(
            0.5,
            0.5,
            "No cluster_score column",
            ha="center",
            va="center",
            transform=ax3.transAxes,
        )

    ax4 = axes[1, 1]
    if "cluster" in df_umap.columns:
        cluster_stats = (
            df_umap.group_by("cluster")
            .agg(
                [
                    pl.len().alias("total"),
                    pl.col("is_seed").sum().alias("seed_count"),
                ]
            )
            .with_columns(
                (pl.col("seed_count") / pl.col("total") * 100).alias("seed_pct")
            )
            .sort("seed_pct", descending=True)
        )

        clusters = cluster_stats["cluster"].to_numpy()
        seed_pcts = cluster_stats["seed_pct"].to_numpy()

        colors = ["red" if pct > 0 else "gray" for pct in seed_pcts]
        ax4.bar(range(len(clusters)), seed_pcts, color=colors, alpha=0.7)
        ax4.set_title(f"{title_prefix}Seed Percentage by Cluster")
        ax4.set_xlabel("Cluster (sorted by seed %)")
        ax4.set_ylabel("Seed Percentage (%)")
        ax4.axhline(
            y=len(seeds) / len(df_umap) * 100 if len(df_umap) > 0 else 0,
            color="blue",
            linestyle="--",
            label="Expected (random)",
        )
        ax4.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "embedding_enrichment_analysis.png", dpi=150)
    plt.close()

    print(f"Embedding-enrichment analysis saved to {output_dir}")


def load_seeds(seeds_file: Path) -> set[str]:
    """Load seed sequences from file.

    Supports:
    - Plain text (one sequence per line)
    - CSV with 'sequence' column
    - Parquet with 'sequence' column
    - Tab-separated 'seeds.txt' from AptaSuite simulation

    Args:
        seeds_file: Path to seeds file

    Returns:
        Set of seed sequences
    """
    suffix = seeds_file.suffix.lower()

    if suffix == ".parquet":
        df = pl.read_parquet(seeds_file)
        return set(df["sequence"].to_list())

    elif suffix == ".csv":
        df = pl.read_csv(seeds_file)
        return set(df["sequence"].to_list())

    elif seeds_file.name == "seeds.txt" or suffix == ".txt":
        df = pl.read_csv(
            seeds_file,
            separator="\t",
            has_header=True,
        )
        if "Sequence" in df.columns:
            return set(df["Sequence"].to_list())
        return set(df.select(df.columns[1]).to_series().to_list())

    else:
        with open(seeds_file) as f:
            return {line.strip() for line in f if line.strip()}


def load_selex_data(data_dir: Path) -> pl.DataFrame:
    """Load SELEX data from directory.

    Expects files named like:
    - Round0.fastq.gz, Round1.fastq.gz, ...
    - Or: round_0.parquet, round_1.parquet, ...

    It checks for FASTQ files in both the root of the data_dir and in an
    'export' subdirectory.

    Args:
        data_dir: Directory containing round files

    Returns:
        DataFrame with columns [sequence, round, count]
    """
    data_dir = Path(data_dir)
    all_data = []

    parquet_files = sorted(data_dir.glob("*.parquet"))
    if parquet_files:
        for pq in parquet_files:
            df = pl.read_parquet(pq)
            import re

            match = re.search(r"(\d+)", pq.stem)
            if match:
                round_num = int(match.group(1))
                df = df.with_columns(pl.lit(round_num).alias("round"))
                all_data.append(df)
    else:
        fastq_files = sorted(data_dir.glob("*.fastq*"))

        for subdir in ["export", "rounds"]:
            if (data_dir / subdir).exists():
                fastq_files.extend(sorted((data_dir / subdir).glob("*.fastq*")))

        fastq_files = sorted(list(set(fastq_files)))

        for fq in fastq_files:
            sequences = _parse_fastq(fq)
            from collections import Counter

            counts = Counter(sequences)

            import re

            match = re.search(r"[._-]?(\d+)", fq.stem)
            round_num = int(match.group(1)) if match else 0

            df = pl.DataFrame(
                {
                    "sequence": list(counts.keys()),
                    "count": list(counts.values()),
                    "round": [round_num] * len(counts),
                }
            )
            all_data.append(df)

    if not all_data:
        raise ValueError(f"No data files found in {data_dir} or {data_dir / 'export'}")

    return pl.concat(all_data)


def _parse_fastq(path: Path) -> list[str]:
    """Parse FASTQ file and return sequences."""
    import gzip

    sequences = []
    opener = gzip.open if str(path).endswith(".gz") else open

    with opener(path, "rt") as f:
        for i, line in enumerate(f):
            if i % 4 == 1:
                sequences.append(line.strip())

    return sequences
