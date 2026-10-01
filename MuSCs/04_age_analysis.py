"""
How MuSC clusters change with donor age.

Figures: results_age/*.pdf
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.stats import false_discovery_control, spearmanr
from scvi.external import MRVI

from utils import (
    AGE_DIR,
    AGE_GROUP_COLORS,
    AGE_GROUP_KEY,
    AGE_GROUP_LABELS,
    AGE_KEY,
    AGE_MODEL_DIR,
    ANNOTATION_KEY,
    BATCH_KEY,
    LATENT_KEY,
    LEIDEN_KEY,
    MRVI_BATCH_SIZE,
    MRVI_MAX_EPOCHS,
    MRVI_N_LATENT,
    SAMPLE_KEY,
    configure_plotting,
    present_genes,
    read_musc,
    savefig,
)

CANDIDATE_AGE_GENES = [
    "ABCA8", "ABCA10", "IGF1", "CCN5", "CRLF1", "VIT", "PTGIS", "ABLIM3",
    "GREB1L", "PRKG1", "AJAP1", "FKBP5", "GPHN", "LRRC7", "COL15A1", "LAMA2",
]
AGE_PALETTE = dict(zip(AGE_GROUP_LABELS, AGE_GROUP_COLORS))


def bh(pvals):
    """Benjamini-Hochberg q-values that leave NaN p-values as NaN."""

    pvals = np.asarray(pvals, dtype=float)
    q = np.full_like(pvals, np.nan)
    ok = ~np.isnan(pvals)
    if ok.any():
        q[ok] = false_discovery_control(pvals[ok])
    return q


def spearman_with_age(ages, values):
    if pd.Series(ages).nunique() < 3 or pd.Series(values).nunique() < 3:
        return np.nan, np.nan
    return spearmanr(ages, values)


def plot_heatmap(df, name, cmap, vmin=None, vmax=None, cbar_label=None):
    """Clusters x columns heatmap. A missing vmin/vmax defaults to -/+ the 95th percentile of |values|."""

    data = df.to_numpy(dtype=float)
    lim = np.nanpercentile(np.abs(data), 95)
    vmin = -lim if vmin is None else vmin
    vmax = lim if vmax is None else vmax

    fig, ax = plt.subplots(figsize=(max(4.5, 0.28 * df.shape[1] + 2.0), max(2.6, 0.32 * df.shape[0] + 1.2)))
    im = ax.imshow(np.ma.masked_invalid(data), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(df.shape[1]), df.columns, rotation=45, ha="right")
    ax.set_yticks(range(df.shape[0]), df.index)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label=cbar_label)
    savefig(name, fig, folder=AGE_DIR)


def panel_grid(n):
    nrows = math.ceil(n / 3)
    fig, axes = plt.subplots(nrows, 3, figsize=(11, 3.4 * nrows))
    axes = axes.ravel()
    for ax in axes[n:]:
        ax.set_visible(False)
    return fig, axes


def cluster_fractions(adata, clusters):
    """Fraction of each sample's MuSCs in each cluster (0 when a sample has none), with sample age."""

    counts = adata.obs.groupby([SAMPLE_KEY, ANNOTATION_KEY], observed=True).size().rename("n")
    grid = pd.MultiIndex.from_product([adata.obs[SAMPLE_KEY].unique(), clusters], names=[SAMPLE_KEY, ANNOTATION_KEY])
    counts = counts.reindex(grid, fill_value=0).reset_index()
    counts["fraction"] = counts["n"] / counts.groupby(SAMPLE_KEY, observed=True)["n"].transform("sum")
    meta = adata.obs.groupby(SAMPLE_KEY, observed=True)[[AGE_KEY, AGE_GROUP_KEY]].first()
    return counts.join(meta, on=SAMPLE_KEY).rename(columns={ANNOTATION_KEY: "cluster"})


def plot_age_umap(adata, clusters):
    cluster_palette = {c: plt.get_cmap("tab10")(i % 10) for i, c in enumerate(clusters)}
    sc.pl.umap(adata, color=[ANNOTATION_KEY, AGE_KEY, AGE_GROUP_KEY], palette={**cluster_palette, **AGE_PALETTE},
               frameon=False, title=["", "", ""], wspace=0.8, legend_fontsize=8, size=9, show=False)
    savefig("age_umap.pdf", folder=AGE_DIR)


def plot_fraction_by_age_group(fractions, clusters):
    fig, axes = panel_grid(len(clusters))
    for i, (ax, cluster) in enumerate(zip(axes, clusters)):
        sub = fractions[fractions["cluster"] == cluster]
        data = [sub.loc[sub[AGE_GROUP_KEY] == g, "fraction"].values for g in AGE_GROUP_LABELS]
        box = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False)
        for patch, group in zip(box["boxes"], AGE_GROUP_LABELS):
            patch.set_facecolor(AGE_PALETTE[group])
            patch.set_alpha(0.6)
        for x, vals in enumerate(data, 1):
            jitter = np.random.default_rng(0).normal(x, 0.04, size=len(vals))
            ax.scatter(jitter, vals, s=15, color="#3f3f3f", alpha=0.7, linewidth=0)
        ax.set_title(f"Cluster {i}", fontsize=10)
        ax.set_xticks(range(1, len(AGE_GROUP_LABELS) + 1), AGE_GROUP_LABELS, rotation=45, ha="right")
        ax.set_ylabel("Fraction")
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(h_pad=3.0, w_pad=2.0)
    savefig("number_age_groups.pdf", fig, folder=AGE_DIR)


def plot_fraction_vs_age(fractions, clusters):
    stats = pd.DataFrame(
        [(c, *spearman_with_age(sub[AGE_KEY], sub["fraction"])) for c, sub in fractions.groupby("cluster", observed=True)],
        columns=["cluster", "rho", "pval"],
    ).set_index("cluster")
    stats["qval"] = bh(stats["pval"])

    fig, axes = panel_grid(len(clusters))
    for ax, cluster in zip(axes, clusters):
        sub = fractions[fractions["cluster"] == cluster]
        for group, color in AGE_PALETTE.items():
            grp = sub[sub[AGE_GROUP_KEY] == group]
            ax.scatter(grp[AGE_KEY], grp["fraction"], s=22, color=color, alpha=0.85, linewidth=0,
                       label=group if ax is axes[0] else None)
        slope, intercept = np.polyfit(sub[AGE_KEY], sub["fraction"], deg=1)
        xs = np.linspace(sub[AGE_KEY].min(), sub[AGE_KEY].max(), 100)
        ax.plot(xs, slope * xs + intercept, color="black", lw=1)
        ax.text(0.03, 0.95, f"rho={stats.loc[cluster, 'rho']:.2f}, q={stats.loc[cluster, 'qval']:.3g}",
                transform=ax.transAxes, va="top", ha="left", fontsize=8)
        ax.set_title(cluster, fontsize=10)
        ax.set(xlabel="Age", ylabel="Fraction")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(loc="lower right", fontsize=7, frameon=False)
    fig.tight_layout(h_pad=3.0, w_pad=2.0)
    savefig("number_continuous_age.pdf", fig, folder=AGE_DIR)

    plot_heatmap(stats.loc[clusters, ["rho"]].rename(columns={"rho": "Spearman rho"}),
                 "differential_abundance_continuous_age.pdf", cmap="RdBu_r", vmin=-1, vmax=1, cbar_label="rho")


def load_age_model(adata):
    """MrVI model for the age analyses, with each sample's age and age group attached as covariates."""

    if AGE_MODEL_DIR.exists():
        model = MRVI.load(str(AGE_MODEL_DIR), adata=adata)
    else:
        MRVI.setup_anndata(adata, layer="counts", sample_key=SAMPLE_KEY, batch_key=BATCH_KEY)
        model = MRVI(adata, n_latent=MRVI_N_LATENT)
        model.train(max_epochs=MRVI_MAX_EPOCHS, batch_size=MRVI_BATCH_SIZE, early_stopping=True)
        model.save(str(AGE_MODEL_DIR), overwrite=True)

    meta = adata.obs.groupby(SAMPLE_KEY, observed=True)[[AGE_KEY, AGE_GROUP_KEY]].first()
    samples = model.sample_info[SAMPLE_KEY].astype(str)
    model.sample_info[AGE_KEY] = samples.map(meta[AGE_KEY]).values
    model.sample_info[AGE_GROUP_KEY] = samples.map(meta[AGE_GROUP_KEY]).values
    return model


def plot_abundance_by_age_group(model, adata, clusters):
    """MrVI differential abundance: mean log enrichment of each age group's samples, per cluster."""

    ds = model.differential_abundance(adata=adata, sample_cov_keys=[AGE_GROUP_KEY], compute_log_enrichment=True,
                                      omit_original_sample=True, batch_size=128)
    enrich = ds[f"{AGE_GROUP_KEY}_log_enrichs"].to_pandas()
    mat = enrich.groupby(adata.obs.loc[enrich.index, ANNOTATION_KEY].values, observed=True).mean()
    mat = mat.reindex(clusters)[AGE_GROUP_LABELS]
    plot_heatmap(mat, "differential_abundance_age_group_enrichment.pdf", cmap="RdBu_r",
                 vmin=np.nanmin(mat.values), vmax=np.nanmax(mat.values), cbar_label="MRVI log enrichment")


def plot_expression_age_effect(model, adata, clusters):
    """MrVI differential expression over continuous age, on up to 1,000 cells per cluster."""

    rng = np.random.default_rng(0)
    labels = adata.obs[ANNOTATION_KEY].values
    keep = []
    for cluster in clusters:
        idx = np.flatnonzero(labels == cluster)
        keep.extend(rng.choice(idx, size=1000, replace=False) if len(idx) > 1000 else idx)
    sub = adata[np.asarray(keep)].copy()

    de = model.differential_expression(adata=sub, sample_cov_keys=[AGE_KEY], batch_size=64, use_vmap="auto",
                                       mc_samples=25, store_lfc=False, filter_inadmissible_samples=False)
    per_cell = pd.DataFrame({
        "effect_size": de["effect_size"].sel(covariate=AGE_KEY).to_pandas(),
        "padj": de["padj"].sel(covariate=AGE_KEY).to_pandas(),
    })
    per_cell["cluster"] = sub.obs.loc[per_cell.index, ANNOTATION_KEY].values
    summary = per_cell.groupby("cluster", observed=True).agg(
        mean_effect_size=("effect_size", "mean"),
        median_effect_size=("effect_size", "median"),
        significant_cell_fraction=("padj", lambda p: np.mean(p < 0.05)),
    ).reindex(clusters)
    plot_heatmap(summary, "differential_expression_age_effect.pdf", cmap="Reds", vmin=0, cbar_label="MRVI age effect")


def plot_candidate_gene_age_correlation(adata, clusters):
    """Spearman correlation of each candidate gene with donor age, per cluster.

    Correlations are across cells, not samples. Black outlines mark BH q < 0.05 within a cluster.
    (MrVI's per-gene fold changes, store_lfc=True, fail on this model, so they are not used.)
    """

    genes = present_genes(adata, CANDIDATE_AGE_GENES)
    expr = adata[:, genes].layers["lognorm"].toarray()
    labels = adata.obs[ANNOTATION_KEY].values
    ages = adata.obs[AGE_KEY].values

    rows = []
    for cluster in clusters:
        idx = labels == cluster
        for j, gene in enumerate(genes):
            rows.append((cluster, gene, *spearman_with_age(ages[idx], expr[idx, j])))
    table = pd.DataFrame(rows, columns=["cluster", "gene", "rho", "pval"])
    table["qval"] = table.groupby("cluster")["pval"].transform(bh)

    fig, ax = plt.subplots(figsize=(max(7, 0.32 * len(genes) + 2), 3.5))
    for _, row in table.iterrows():
        size = 25 + 120 * min(abs(row["rho"]) if np.isfinite(row["rho"]) else 0, 1)
        ax.scatter(genes.index(row["gene"]), clusters.index(row["cluster"]), s=size, c=[row["rho"]], cmap="RdBu_r",
                   vmin=-1, vmax=1, edgecolor="black" if row["qval"] < 0.05 else "none", linewidth=0.5)
    ax.set_xticks(range(len(genes)), genes, rotation=45, ha="right")
    ax.set_yticks(range(len(clusters)), clusters)
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    fig.colorbar(plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(-1, 1)), ax=ax, fraction=0.035, pad=0.02,
                 label="Spearman rho")
    savefig("gene_level_age_lfc_dotplot.pdf", fig, folder=AGE_DIR)


def plot_sample_distance_vs_age_gap(adata):
    """Do samples further apart in age sit further apart in the MrVI latent space? One point per sample pair."""

    centroids = pd.DataFrame(adata.obsm[LATENT_KEY], index=adata.obs_names).groupby(adata.obs[SAMPLE_KEY].values).mean()
    ages = adata.obs.groupby(SAMPLE_KEY, observed=True)[AGE_KEY].first().loc[centroids.index]

    pairs = np.triu_indices(len(centroids), k=1)
    age_gap = squareform(pdist(ages.values.reshape(-1, 1)))[pairs]
    distance = squareform(pdist(centroids.values))[pairs]
    rho, pval = spearmanr(age_gap, distance)

    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    ax.scatter(age_gap, distance, s=20, color="#4C78A8", alpha=0.75, linewidth=0)
    slope, intercept = np.polyfit(age_gap, distance, 1)
    xs = np.linspace(age_gap.min(), age_gap.max(), 100)
    ax.plot(xs, slope * xs + intercept, color="black", lw=1)
    ax.text(0.03, 0.95, f"rho={rho:.2f}, p={pval:.3g}", transform=ax.transAxes, va="top", ha="left", fontsize=8)
    ax.set(xlabel="Age gap", ylabel="MRVI latent distance")
    ax.spines[["top", "right"]].set_visible(False)
    savefig("local_sample_distance_age.pdf", fig, folder=AGE_DIR)


def plot_admissibility(adata, clusters):
    """How much further each cell is from its own cluster's latent centroid than from the nearest one (0 = nearest)."""

    latent = adata.obsm[LATENT_KEY]
    labels = adata.obs[ANNOTATION_KEY].values
    dists = np.column_stack([np.linalg.norm(latent - latent[labels == c].mean(axis=0), axis=1) for c in clusters])
    own = dists[np.arange(len(labels)), pd.Categorical(labels, categories=clusters).codes]
    adata.obs["admissibility_proxy"] = dists.min(axis=1) - own

    sc.pl.umap(adata, color="admissibility_proxy", cmap="viridis", frameon=False, title="", show=False)
    savefig("outlier_admissibility_umap.pdf", folder=AGE_DIR)

    mat = adata.obs.groupby([ANNOTATION_KEY, AGE_GROUP_KEY], observed=True)["admissibility_proxy"].mean()
    plot_heatmap(mat.unstack(AGE_GROUP_KEY).reindex(clusters), "outlier_admissibility_age_heatmaps.pdf",
                 cmap="viridis", cbar_label="mean admissibility")


def main():
    configure_plotting()
    adata = read_musc()
    clusters = list(adata.obs[ANNOTATION_KEY].cat.categories)

    fractions = cluster_fractions(adata, clusters)
    plot_age_umap(adata, clusters)
    plot_fraction_by_age_group(fractions, clusters)
    plot_fraction_vs_age(fractions, clusters)

    model = load_age_model(adata)
    plot_abundance_by_age_group(model, adata, clusters)
    plot_expression_age_effect(model, adata, clusters)

    plot_candidate_gene_age_correlation(adata, clusters)
    plot_sample_distance_vs_age_gap(adata)
    plot_admissibility(adata, clusters)


if __name__ == "__main__":
    main()
