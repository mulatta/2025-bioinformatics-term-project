"""
Usage:
    python -m selex_analyze_poc.compare_embeddings \
        --data-dir ./data/aptasuite/ \
        --seeds-file ./data/aptasuite/seeds.txt \
        --output-dir ./results/poc_comparison/
"""

from dataclasses import dataclass
from pathlib import Path

import click
import matplotlib.pyplot as plt
import polars as pl

from selex_analyze_poc.evaluation import (
    EvaluationResult,
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


@dataclass
class MethodResult:
    """Result from a single method."""

    name: str
    eval_result: EvaluationResult
    viz_data: pl.DataFrame | None
    ranking_data: pl.DataFrame


def run_count_baseline(
    df: pl.DataFrame,
    seeds: set[str],
) -> MethodResult:
    """Run count-based fold-change baseline."""
    print("\n" + "=" * 50)
    print("Running: Count-Based Baseline (Fold Change)")
    print("=" * 50)

    ranker = CountBasedRanker()
    ranking = ranker.rank(df)
    eval_result = evaluate_seed_recovery(ranking, seeds)

    print(f"  AUC-ROC: {eval_result.auc_roc:.4f}")

    return MethodResult(
        name="CountFoldChange",
        eval_result=eval_result,
        viz_data=None,
        ranking_data=ranking.to_dataframe(),
    )


def run_embedding_method(
    df: pl.DataFrame,
    seeds: set[str],
    extractor,
    n_clusters: int,
    device: str,
) -> MethodResult:
    """Run an embedding-based method."""
    name = extractor.name
    print("\n" + "=" * 50)
    print(f"Running: {name}")
    print("=" * 50)

    ranker = ClusterEnrichmentRanker(
        extractor=extractor,
        n_clusters=n_clusters,
        device=device,
    )

    ranking = ranker.rank(df, seeds)
    eval_result = evaluate_seed_recovery(ranking, seeds)
    viz_data = ranker.get_visualization_data()

    print(f"  AUC-ROC: {eval_result.auc_roc:.4f}")

    return MethodResult(
        name=name,
        eval_result=eval_result,
        viz_data=viz_data,
        ranking_data=ranking.to_dataframe(),
    )


def create_comparison_table(results: list[MethodResult]) -> pl.DataFrame:
    """Create a comparison table from all results."""
    rows = []
    for r in results:
        row = r.eval_result.to_dict()
        rows.append(row)

    return pl.DataFrame(rows).sort("auc_roc", descending=True)


def plot_comparison_summary(
    results: list[MethodResult],
    output_dir: Path,
) -> None:
    """Create summary comparison plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    ax1 = axes[0, 0]
    methods = [r.name for r in results]
    aucs = [r.eval_result.auc_roc for r in results]

    colors = [
        "#e74c3c" if "RNA-FM" in m else "#3498db" if "Count" in m else "#2ecc71"
        for m in methods
    ]
    bars = ax1.barh(methods, aucs, color=colors)
    ax1.axvline(x=0.5, color="gray", linestyle="--", label="Random", alpha=0.7)
    ax1.set_xlabel("AUC-ROC")
    ax1.set_title("Seed Recovery: AUC-ROC Comparison")
    ax1.set_xlim(0, 1)

    for bar, auc in zip(bars, aucs, strict=True):
        ax1.text(
            auc + 0.02, bar.get_y() + bar.get_height() / 2, f"{auc:.3f}", va="center"
        )

    ax2 = axes[0, 1]
    k_values = [10, 50, 100, 500]
    for r in results:
        precisions = [r.eval_result.precision_at_k.get(k, 0) for k in k_values]
        ax2.plot(k_values, precisions, marker="o", label=r.name)

    ax2.set_xlabel("K")
    ax2.set_ylabel("Precision@K")
    ax2.set_title("Precision at Top-K")
    ax2.legend(loc="best", fontsize=8)
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    for r in results:
        recalls = [r.eval_result.recall_at_k.get(k, 0) for k in k_values]
        ax3.plot(k_values, recalls, marker="o", label=r.name)

    ax3.set_xlabel("K")
    ax3.set_ylabel("Recall@K")
    ax3.set_title("Recall at Top-K")
    ax3.legend(loc="best", fontsize=8)
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    ax4.axis("off")

    best_by_auc = max(results, key=lambda r: r.eval_result.auc_roc)

    summary_text = f"""
    PoC Results Summary
    {"=" * 40}

    Total Methods Compared: {len(results)}
    Best AUC-ROC: {best_by_auc.name} ({best_by_auc.eval_result.auc_roc:.4f})

    Hypothesis Test:
    ----------------
    Foundation Model (RNA-FM) AUC: {next((r.eval_result.auc_roc for r in results if "RNA-FM" in r.name), "N/A")}
    Random Baseline AUC: 0.5000

    Interpretation:
    - AUC ≈ 0.5: Method is no better than random
    - AUC < 0.6: Minimal predictive value
    - AUC > 0.7: Meaningful predictive value

    Conclusion:
    {"Foundation model shows limited ability to separate enriched sequences" if any(r.eval_result.auc_roc < 0.6 for r in results if "RNA-FM" in r.name) else "Foundation model shows some predictive ability"}
    """

    ax4.text(
        0.1,
        0.9,
        summary_text,
        transform=ax4.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
    )

    plt.tight_layout()
    plt.savefig(output_dir / "poc_comparison_summary.png", dpi=150)
    plt.close()


def plot_umap_comparison(
    results: list[MethodResult],
    output_dir: Path,
) -> None:
    """Create side-by-side UMAP comparison for embedding methods."""
    embedding_results = [r for r in results if r.viz_data is not None]

    if not embedding_results:
        print("No embedding results to visualize")
        return

    n_methods = len(embedding_results)
    fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 6))

    if n_methods == 1:
        axes = [axes]

    for ax, result in zip(axes, embedding_results, strict=True):
        viz = result.viz_data

        non_seeds = viz.filter(~pl.col("is_seed"))
        seeds = viz.filter(pl.col("is_seed"))

        ax.scatter(
            non_seeds["umap_x"],
            non_seeds["umap_y"],
            c="lightgray",
            s=5,
            alpha=0.3,
        )

        if len(seeds) > 0:
            ax.scatter(
                seeds["umap_x"],
                seeds["umap_y"],
                c="red",
                s=30,
                alpha=0.8,
                label=f"Seeds (n={len(seeds)})",
            )

        ax.set_title(f"{result.name}\nAUC={result.eval_result.auc_roc:.3f}")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "umap_comparison.png", dpi=150)
    plt.close()


@click.command()
@click.option(
    "--data-dir",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Directory containing SELEX round data",
)
@click.option(
    "--seeds-file",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="File containing seed sequences (ground truth)",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("./poc_results"),
    help="Output directory for results",
)
@click.option(
    "--n-clusters",
    type=int,
    default=20,
    help="Number of clusters for K-Means",
)
@click.option(
    "--device",
    type=click.Choice(["cpu", "cuda", "auto"]),
    default="auto",
    help="Device for computation",
)
@click.option(
    "--max-len",
    type=int,
    default=100,
    help="Max sequence length",
)
@click.option(
    "--max-sequences",
    type=int,
    default=None,
    help="Limit sequences for testing",
)
@click.option(
    "--skip-rnafm",
    is_flag=True,
    help="Skip RNA-FM (slow, needs GPU)",
)
@click.option(
    "--skip-gat",
    is_flag=True,
    help="Skip BPPM-GAT",
)
def main(
    data_dir: Path,
    seeds_file: Path,
    output_dir: Path,
    n_clusters: int,
    device: str,
    max_len: int,
    max_sequences: int | None,
    skip_rnafm: bool,
    skip_gat: bool,
) -> None:
    """Run PoC comparison: Foundation Model vs Physics-Based Embeddings.

    This script tests the hypothesis that foundation model embeddings
    alone cannot distinguish enriched sequences in SELEX data.
    """
    print("=" * 60)
    print("PoC: Foundation Model vs Physics-Based Embeddings")
    print("=" * 60)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[1/6] Loading data from {data_dir}...")
    df = load_selex_data(data_dir)
    print(f"  Loaded {len(df)} records, {df['sequence'].n_unique()} unique sequences")
    print(f"  Rounds: {sorted(df['round'].unique().to_list())}")

    if max_sequences is not None:
        unique_seqs = (
            df["sequence"]
            .unique()
            .sample(min(max_sequences, df["sequence"].n_unique()))
        )
        df = df.filter(pl.col("sequence").is_in(unique_seqs))
        print(f"  Sampled to {df['sequence'].n_unique()} sequences")

    print(f"\n[2/6] Loading seeds from {seeds_file}...")
    seeds = load_seeds(seeds_file)
    print(f"  Loaded {len(seeds)} seed sequences")

    seq_set = set(df["sequence"].unique().to_list())
    overlap = len(seeds & seq_set)
    print(
        f"  Seeds in dataset: {overlap}/{len(seeds)} ({100 * overlap / len(seeds):.1f}%)"
    )

    results: list[MethodResult] = []

    print("\n[3/6] Running baselines...")
    results.append(run_count_baseline(df, seeds))

    print("\n[4/6] Running physics-based methods...")
    physics_extractor = CombinedExtractor(
        [
            OneHotExtractor(max_length=max_len),
            KmerExtractor(k=3),
            ThermoExtractor(max_length=max_len),
        ],
        device=device,
    )
    results.append(
        run_embedding_method(df, seeds, physics_extractor, n_clusters, device)
    )

    if not skip_rnafm:
        print("\n[5/6] Running foundation model (RNA-FM)...")
        rnafm_extractor = RNAFMExtractor(device=device)
        results.append(
            run_embedding_method(df, seeds, rnafm_extractor, n_clusters, device)
        )
    else:
        print("\n[5/6] Skipping RNA-FM (--skip-rnafm)")

    if not skip_gat:
        print("\n[6/6] Running BPPM-GAT...")
        gat_extractor = BPPMGATExtractor(device=device, max_length=max_len)
        results.append(
            run_embedding_method(df, seeds, gat_extractor, n_clusters, device)
        )
    else:
        print("\n[6/6] Skipping BPPM-GAT (--skip-gat)")

    print("\n" + "=" * 60)
    print("Generating comparison outputs...")
    print("=" * 60)

    comparison_df = create_comparison_table(results)
    comparison_df.write_csv(output_dir / "comparison_table.csv")
    print("\nComparison Table:")
    print(comparison_df)

    for r in results:
        safe_name = (
            r.name.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "_")
        )
        r.ranking_data.write_parquet(output_dir / f"ranking_{safe_name}.parquet")

        if r.viz_data is not None:
            r.viz_data.write_parquet(output_dir / f"umap_{safe_name}.parquet")

            method_dir = output_dir / safe_name
            method_dir.mkdir(exist_ok=True)
            plot_embedding_enrichment_analysis(
                r.viz_data, method_dir, title_prefix=f"{r.name}: "
            )

    plot_comparison_summary(results, output_dir)
    plot_umap_comparison(results, output_dir)

    print(f"\nResults saved to {output_dir}")
    print("\nKey files:")
    print(f"  - {output_dir}/comparison_table.csv")
    print(f"  - {output_dir}/poc_comparison_summary.png")
    print(f"  - {output_dir}/umap_comparison.png")


if __name__ == "__main__":
    main()
