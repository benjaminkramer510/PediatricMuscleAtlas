"""
CellRank 2 PseudotimeKernel trajectory.

Figures: 05_pseudotimekernel_umap.png, 06_pseudotimekernel_streamlines.png
"""

import numpy as np
import scanpy as sc
from cellrank.kernels import ConnectivityKernel, PseudotimeKernel

from utils import LEIDEN_KEY, SEED, configure_plotting, read_musc, savefig


def find_root(adata):
    pax7 = np.asarray(adata[:, "PAX7"].layers["lognorm"].todense()).ravel()
    total_counts = np.asarray(adata.layers["counts"].sum(axis=1)).ravel()
    well_captured = np.flatnonzero(total_counts >= np.median(total_counts))
    return int(well_captured[np.argmax(pax7[well_captured])])


def main():
    configure_plotting()
    np.random.seed(SEED)
    adata = read_musc()

    adata.uns["iroot"] = find_root(adata)
    sc.tl.diffmap(adata)
    sc.tl.dpt(adata)
    # Cells unreachable from the root get infinite pseudotime; cap them at the finite maximum
    pt = adata.obs["dpt_pseudotime"]
    adata.obs["dpt_pseudotime"] = pt.replace(np.inf, pt[np.isfinite(pt)].max())

    sc.pl.umap(adata, color="dpt_pseudotime", cmap="plasma", show=False)
    savefig("05_pseudotimekernel_umap.png")

    pk = PseudotimeKernel(adata, time_key="dpt_pseudotime").compute_transition_matrix(threshold_scheme="soft")
    ck = ConnectivityKernel(adata).compute_transition_matrix()
    (0.8 * pk + 0.2 * ck).plot_projection(basis="umap", color=LEIDEN_KEY, legend_loc="right", recompute=True,
                                          show=False)
    savefig("06_pseudotimekernel_streamlines.png")


if __name__ == "__main__":
    main()
