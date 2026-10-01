"""
Full-atlas UMAP, then subsets the atlas to MuSCs, embeds them with MrVI and clusters them with Leiden.

Figures: 01_full_atlas_umap.png, 02_umap_subclusters.png
"""

import pandas as pd
import scanpy as sc
import torch
from scvi.external import MRVI

from utils import (
    BATCH_KEY,
    FULL_ATLAS_H5AD,
    LATENT_KEY,
    LEIDEN_KEY,
    MRVI_BATCH_SIZE,
    MRVI_MAX_EPOCHS,
    MRVI_MODEL_DIR,
    MRVI_N_LATENT,
    MUSC_H5AD,
    SAMPLE_KEY,
    SEED,
    configure_plotting,
    savefig,
)

LEIDEN_RESOLUTION = 0.3
N_NEIGHBORS = 15


def mrvi_latent(adata):
    """Sample-corrected MrVI latent (u), from the saved model if there is one."""

    if MRVI_MODEL_DIR.exists():
        # The saved model must be given exactly the genes it was trained on
        trained_genes = torch.load(MRVI_MODEL_DIR / "model.pt", map_location="cpu", weights_only=False)["var_names"]
        model = MRVI.load(str(MRVI_MODEL_DIR), adata=adata[:, pd.Index(trained_genes)].copy())
    else:
        MRVI.setup_anndata(adata, layer="counts", sample_key=SAMPLE_KEY, batch_key=BATCH_KEY)
        model = MRVI(adata, n_latent=MRVI_N_LATENT)
        model.train(max_epochs=MRVI_MAX_EPOCHS, batch_size=MRVI_BATCH_SIZE, early_stopping=True)
        model.save(str(MRVI_MODEL_DIR), overwrite=True)
    return model.get_latent_representation(give_z=False)


def main():
    configure_plotting()
    sc.settings.n_jobs = 4

    # Only obs/obsm are needed for the full-atlas plot and the MuSC subset, so open it backed
    atlas = sc.read_h5ad(FULL_ATLAS_H5AD, backed="r")
    sc.pl.embedding(atlas, basis="X_umap", color=["celltype", BATCH_KEY], frameon=False,
                    title=["celltype", "batch (B=BCH, C=Cornell)"], show=False)
    savefig("01_full_atlas_umap.png")

    adata = atlas[atlas.obs["celltype"].astype(str) == "MuSC"].to_memory()
    adata.layers["counts"] = adata.X.copy()
    adata.obsm[LATENT_KEY] = mrvi_latent(adata)

    sc.pp.filter_genes(adata, min_cells=5)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.layers["lognorm"] = adata.X.copy()

    sc.pp.neighbors(adata, use_rep=LATENT_KEY, n_neighbors=N_NEIGHBORS, random_state=SEED)
    sc.tl.umap(adata, random_state=SEED)
    sc.tl.leiden(adata, resolution=LEIDEN_RESOLUTION, key_added=LEIDEN_KEY, flavor="igraph",
                 n_iterations=2, directed=False, random_state=SEED)
    adata.obs[LEIDEN_KEY] = adata.obs[LEIDEN_KEY].astype(str)

    MUSC_H5AD.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(MUSC_H5AD)

    sc.pl.umap(adata, color=LEIDEN_KEY, legend_loc="right margin", frameon=False,
               title=f"MrVI Leiden clusters, resolution {LEIDEN_RESOLUTION}", show=False)
    savefig("02_umap_subclusters.png")


if __name__ == "__main__":
    main()
