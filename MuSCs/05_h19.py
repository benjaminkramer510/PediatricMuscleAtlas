"""
H19 in MuSCs: expression per cluster, on the UMAP, and against donor age.

Figures: 07_h19_dotplot_by_cluster.png, 08_h19_umap.png, 09_h19_vs_age.png, 10_h19_vs_age_by_cluster.png
"""

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import seaborn as sns
from scipy.stats import linregress

from utils import AGE_KEY, BATCH_KEY, LEIDEN_KEY, SAMPLE_KEY, SEED, configure_plotting, read_musc, savefig

GENE = "H19"
MIN_CELLS = 20  # a sample needs this many cells in a cluster to get a point in that cluster's panel
MIN_SAMPLES_FOR_FIT = 5  # fewer points than this give a meaningless line and R^2
COHORTS = {"B": ("BCH", "#2166AC"), "C": ("Cornell", "#B2182B")}
SCATTER_KWS = {"s": 28, "alpha": 0.8, "edgecolor": "white", "linewidths": 0.5}


def pseudobulk(adata, group_cols):
    counts = adata.layers["counts"].tocsr()
    df = adata.obs[group_cols + [AGE_KEY, BATCH_KEY]].copy()
    df["gene"] = counts[:, adata.var_names.get_loc(GENE)].toarray().ravel()
    df["lib"] = np.asarray(counts.sum(axis=1)).ravel()

    pb = df.groupby(group_cols, observed=True).agg(
        n_cells=("gene", "size"), gene=("gene", "sum"), lib=("lib", "sum"), age=(AGE_KEY, "first"),
        cohort=(BATCH_KEY, "first"),
    ).reset_index()
    pb["log2CPM"] = np.log2(pb["gene"] / pb["lib"] * 1e6 + 1)
    return pb


def scatter_with_fit(ax, d):
    """One color per cohort, each with its own least-squares line, slope and R^2."""

    for i, (code, (name, color)) in enumerate(COHORTS.items()):
        dc = d[d["cohort"] == code]
        ax.scatter(dc["age"], dc["log2CPM"], color=color, label=name, **SCATTER_KWS)
        if len(dc) < MIN_SAMPLES_FOR_FIT:
            continue
        # seed fixes the bootstrapped 95% confidence band, which otherwise shifts slightly on every run
        sns.regplot(data=dc, x="age", y="log2CPM", ax=ax, color=color, scatter=False, line_kws={"lw": 1.8},
                    seed=SEED)
        fit = linregress(dc["age"], dc["log2CPM"])
        ax.text(0.97, 0.97 - 0.1 * i, f"{name}: slope = {fit.slope:.3f}, $R^2$ = {fit.rvalue ** 2:.3f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8, color=color)
    ax.set(xlabel="Donor age (years)", ylabel=f"log2 {GENE} expression")
    ax.grid(False)


def main():
    configure_plotting()
    adata = read_musc()

    dp = sc.pl.dotplot(adata, [GENE], groupby=LEIDEN_KEY, layer="lognorm", cmap="Reds",
                       title=f"{GENE} expression per MuSC cluster", show=False, return_fig=True)
    dp.make_figure()
    savefig("07_h19_dotplot_by_cluster.png", dp.fig)

    pct = (adata[:, GENE].layers["lognorm"] > 0).mean() * 100
    fig, ax = plt.subplots(figsize=(5.8, 5))
    sc.pl.umap(adata, color=GENE, layer="lognorm", ax=ax, show=False, cmap="viridis", sort_order=True,
               vmin=0, vmax="p99.9", size=20, frameon=False, title=f"{GENE} ({pct:.1f}% of MuSCs expressing)")
    fig.tight_layout()
    savefig("08_h19_umap.png", fig)

    pb = pseudobulk(adata, [SAMPLE_KEY])
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    scatter_with_fit(ax, pb)
    ax.set_title(f"{GENE} vs age (n = {len(pb)} samples)")
    ax.legend(loc="lower left", frameon=False, fontsize=8)
    fig.tight_layout()
    savefig("09_h19_vs_age.png", fig)

    pbc = pseudobulk(adata, [SAMPLE_KEY, LEIDEN_KEY])
    pbc = pbc[pbc["n_cells"] >= MIN_CELLS]
    clusters = adata.obs[LEIDEN_KEY].cat.categories
    nrows = -(-len(clusters) // 4)
    fig, axes = plt.subplots(nrows, 4, figsize=(15, 3.5 * nrows), sharex=True, squeeze=False)
    for ax, cluster in zip(axes.flat, clusters):
        d = pbc[pbc[LEIDEN_KEY] == cluster]
        scatter_with_fit(ax, d)
        ax.set_title(f"Cluster {cluster} (n={len(d)} samples)")
        ax.tick_params(labelbottom=True)
    for ax in axes.flat[len(clusters):]:
        ax.set_visible(False)
    fig.suptitle(f"{GENE} per age vs cluster")
    fig.tight_layout()
    savefig("10_h19_vs_age_by_cluster.png", fig)


if __name__ == "__main__":
    main()
