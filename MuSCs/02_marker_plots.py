"""
Myogenic marker expression on the MuSC UMAP, and the Cornell atlas MuSC marker panel per cluster.

Figures: 03_featureplot_myomarkers.png, 04_cornell_marker_dotplot.png
"""

import scanpy as sc
from matplotlib.colors import LinearSegmentedColormap

from utils import ANNOTATION_KEY, configure_plotting, present_genes, read_musc, savefig

MYO_MARKERS = ["PAX7", "PAX3", "MYF5", "MYOD1", "MYOG", "MKI67"]  # MKI67 marks cycling (proliferating) cells

CORNELL_GENES = [
    "MYF6", "MEF2D", "MYH7B", "MYH1", "MEF2C", "MYMK", "MYMX", "MYOD1", "MYOG",
    "CDKN1C", "CDK4", "CDK1", "MKI67", "CXCR4", "ITGA7", "CEROX1", "MEGF10",
    "MYF5", "MEF2A", "SPRY1", "CALCR", "PAX7", "MEG3",
]
CORNELL_CMAP = LinearSegmentedColormap.from_list("cornell_pink_blue", ["#e0368f", "#3a53c4"])


def main():
    configure_plotting()
    adata = read_musc()

    sc.pl.umap(adata, color=present_genes(adata, MYO_MARKERS), ncols=3, cmap="viridis", size=9, show=False)
    savefig("03_featureplot_myomarkers.png")

    sc.pl.dotplot(adata, var_names=present_genes(adata, CORNELL_GENES), groupby=ANNOTATION_KEY,
                  standard_scale="var", cmap=CORNELL_CMAP, swap_axes=True, dendrogram=False, show=False)
    savefig("04_cornell_marker_dotplot.png")


if __name__ == "__main__":
    main()
