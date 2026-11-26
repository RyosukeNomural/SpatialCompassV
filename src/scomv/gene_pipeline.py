from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scanpy as sc
import skbio
from skbio.stats.ordination import pcoa
import plotly.express as px

from .preparation.choose_roi import extract_roi, contour_regions
from .preparation.scomv_calc_vector import (
    compute_min_vectors_polar, flatten_min_vectors, polar_hist_for_expression,
)
from .polar_plot import plot_polar_heatmap


# =========================================================
# SCOMV pipeline as a class
# =========================================================

@dataclass
class SCOMVPipeline:
    adata: Any
    grid: Any

    # ---- defaults (can be overridden per run) ----
    bin_size: int = 10
    region_col: str = "region_10"

    contour_bounds: Tuple[float, float, float, float] = (0, 10000, 0, 10000)
    invert_y: bool = True
    make_inside_negative: bool = True

    # ---- polar bins ----
    radius_bins: Optional[np.ndarray] = None
    angle_bins_deg: Optional[np.ndarray] = None

    # ---- Moran / sampling options ----
    require_moran_nonneg: bool = False
    subsample_fraction: float = 1.0

    # ---- cached / last run results ----
    last_roi: Optional[Tuple[float, float, float, float]] = None
    subset_grid: Any = None
    filtered_shortest: Any = None
    xy_list: Any = None
    outline_points: Optional[List[Tuple[float, float]]] = None
    inside_points: Optional[List[Tuple[float, float]]] = None
    min_vector_df: Optional[pd.DataFrame] = None
    selected_genes: Optional[List[str]] = None
    polar_counts_list: Optional[List[np.ndarray]] = None
    dist_df: Optional[pd.DataFrame] = None
    pcoa_res: Any = None
    coords: Optional[pd.DataFrame] = None
    explained: Optional[pd.Series] = None

    def _ensure_bins(self):
        if self.radius_bins is None:
            self.radius_bins = np.arange(-150, 310, 10)
        if self.angle_bins_deg is None:
            self.angle_bins_deg = np.arange(-180, 181, 30)

    # ----------------------------
    # Step 0: ROI subset for adata (for Moran's I)
    # ----------------------------
    @staticmethod
    def subset_by_roi(adata, roi: Tuple[float, float, float, float]):
        min_x, max_x, min_y, max_y = roi
        return adata[
            (adata.obs["imagecol"] >= min_x) & (adata.obs["imagecol"] <= max_x) &
            (adata.obs["imagerow"] >= min_y) & (adata.obs["imagerow"] <= max_y)
        ].copy()

    # ----------------------------
    # Step 1: compute Moran's I
    # ----------------------------
    def compute_moran_df(
        self,
        roi: Tuple[float, float, float, float],
        subsample_fraction: Optional[float] = None,
        delaunay: bool = True,
        n_perms: int = 100,
        n_jobs: int = 1,
        plot_cdf: bool = False,
    ) -> pd.DataFrame:
        frac = self.subsample_fraction if subsample_fraction is None else subsample_fraction
        ad = self.subset_by_roi(self.adata, roi)
        ad_sub = sc.pp.subsample(ad, fraction=frac, copy=True)

        sq.gr.spatial_neighbors(ad_sub, coord_type="generic", delaunay=delaunay)
        sq.gr.spatial_autocorr(ad_sub, mode="moran", n_perms=n_perms, n_jobs=n_jobs)

        moran_df = ad_sub.uns["moranI"].copy()

        if plot_cdf:
            I_values = moran_df["I"].to_numpy()
            sorted_vals = np.sort(I_values)
            cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)

            plt.figure(figsize=(6, 4))
            plt.plot(sorted_vals, cdf, linewidth=2)
            plt.xlabel("Moran's I")
            plt.ylabel("Cumulative Frequency")
            plt.title("CDF of Moran's I")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        return moran_df

    # ----------------------------
    # Step 2: gene total counts
    # ----------------------------
    @staticmethod
    def gene_total_counts_int(adata) -> np.ndarray:
        X = adata.X
        return np.array(X.sum(axis=0)).ravel().astype(int)

    # ----------------------------
    # Step 3: build polar distributions for genes
    # ----------------------------
    def build_polar_distributions_for_genes(
        self,
        adata_grid,
        min_vector_df: pd.DataFrame,
        moran_df: Optional[pd.DataFrame] = None,
        min_gene_count: int = 0,
        require_moran_nonneg: Optional[bool] = None,
        bin_size_um: Optional[int] = None,
        make_plots: bool = False,
        clim_ratio: float = 0.15,
    ) -> Tuple[List[np.ndarray], List[str]]:
        self._ensure_bins()
        req_moran = self.require_moran_nonneg if require_moran_nonneg is None else require_moran_nonneg
        bin_um = self.bin_size if bin_size_um is None else bin_size_um

        moran_I = moran_df["I"] if moran_df is not None else None

        X = adata_grid.X
        n_genes = adata_grid.n_vars
        n_cells = adata_grid.n_obs

        polar_counts_list: List[np.ndarray] = []
        selected_genes: List[str] = []
        flat = flatten_min_vectors(min_vector_df.iloc[:n_cells])

        for n in range(n_genes):
            gene_name = adata_grid.var.index[n]
            expr = X[:, n]
            if hasattr(expr, "toarray"):
                expr = expr.toarray().ravel()
            else:
                expr = np.asarray(expr).ravel()

            total_n = int(expr.sum())
            if total_n <= min_gene_count:
                continue

            if moran_I is not None and req_moran:
                if gene_name in moran_I.index:
                    if float(moran_I.loc[gene_name]) < 0:
                        continue
                else:
                    continue

            # count cast with int(), split over tied vectors, normalized (same numbers as plt.hist2d)
            counts = polar_hist_for_expression(expr, flat, self.radius_bins, self.angle_bins_deg, bin_um)
            if counts is None:
                continue

            if make_plots:
                max_val = np.max(counts) if counts.size else 0
                fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"})
                plot_polar_heatmap(
                    counts, self.radius_bins, self.angle_bins_deg, ax=ax, cmap="viridis", vmin=0,
                    vmax=max_val * float(clim_ratio) if max_val > 0 else None,
                    cbar_label="Density", title=f"{gene_name} (n={total_n})",
                )
                plt.tight_layout()
                plt.show()
                plt.close(fig)

            selected_genes.append(gene_name)
            polar_counts_list.append(counts)

        return polar_counts_list, selected_genes

    # ----------------------------
    # Step 4: MINAS similarity -> distance df
    # ----------------------------
    @staticmethod
    def minas_similarity_matrix(
        polar_counts_list: List[np.ndarray],
        labels: List[str],
        make_distance: bool = True,
        fill_diagonal_zero: bool = True,
    ) -> pd.DataFrame:
        """Histogram intersection sum(min(A_i, A_j)) for every pair of genes (1 - that if ``make_distance``).

        Computed with a parallel numba kernel over the gene pairs (all-zero polar cells are skipped, they add
        nothing); same numbers as ``float(np.minimum(A_i, A_j).sum())`` up to floating-point summation order
        (< 1e-14). 17,878 genes: ~7 s instead of ~4 min for the pairwise Python loop.
        """
        n = len(labels)
        X = np.stack([np.asarray(a, dtype=np.float64).ravel() for a in polar_counts_list]) if n else np.zeros((0, 0))
        if n and np.isnan(X).any():
            # NaN propagates through np.minimum; keep the reference loop for that (never produced by the pipeline)
            table = np.zeros((n, n), dtype=float)
            for i in range(n):
                for j in range(i, n):
                    table[i, j] = table[j, i] = float(np.minimum(X[i], X[j]).sum())
        else:
            X = np.ascontiguousarray(X[:, X.any(axis=0)]) if n else X
            table = _histogram_intersection(X)
        if make_distance:
            np.subtract(1.0, table, out=table)
        if fill_diagonal_zero:
            np.fill_diagonal(table, 0.0)
        return pd.DataFrame(table, index=labels, columns=labels, copy=False)

    # ----------------------------
    # Step 5: PCoA
    # ----------------------------
    @staticmethod
    def run_pcoa_from_distance_df(dist_df: pd.DataFrame, dimensions: Optional[int] = None,
                                  full_max_n: int = 3000):
        """Classical PCoA of a gene x gene distance matrix.

        Up to ``full_max_n`` genes (and ``dimensions=None``): ``skbio.stats.ordination.pcoa`` (full
        eigendecomposition, all axes), as before. Above that, or when ``dimensions`` is given, only the leading
        ``dimensions`` axes (default 10) are computed with ARPACK (``scipy.sparse.linalg.eigsh``): the same
        eigenvalues and coordinates as the full decomposition (each axis up to its sign; the sign is fixed so that
        the largest-magnitude coordinate is positive), in seconds instead of ~7 min for 17,878 genes.
        ``proportion_explained`` is then relative to the trace of the centred matrix (as skbio does for
        ``method="fsvd"``); the full decomposition divides by the sum of the positive eigenvalues only.

        Returns (skbio OrdinationResults, coordinates DataFrame ``PC1..``, proportion_explained Series).
        """
        n = len(dist_df)
        if dimensions is None and n <= full_max_n:
            dm = skbio.DistanceMatrix(dist_df.values, ids=list(dist_df.index))
            res = pcoa(dm)
        else:
            res = _pcoa_top(dist_df, 10 if dimensions is None else int(dimensions))
        return res, res.samples, res.proportion_explained

    # ----------------------------
    # Plot: PCoA Plotly
    # ----------------------------
    @staticmethod
    def plot_pcoa_plotly(coords: pd.DataFrame, labels: List[str], width: int = 720, height: int = 540, margin: float = 0.05):
        df_plot = pd.DataFrame({
            "PCoA1": coords.iloc[:, 0].values,
            "PCoA2": coords.iloc[:, 1].values,
            "gene": labels
        })
        fig = px.scatter(df_plot, x="PCoA1", y="PCoA2", hover_name="gene")
        fig.update_traces(marker=dict(size=12))
        fig.update_yaxes(scaleanchor="x", scaleratio=1)

        xmin, xmax = df_plot["PCoA1"].min(), df_plot["PCoA1"].max()
        ymin, ymax = df_plot["PCoA2"].min(), df_plot["PCoA2"].max()
        fig.update_xaxes(range=[xmin - margin, xmax + margin])
        fig.update_yaxes(range=[ymin - margin, ymax + margin])

        fig.update_layout(
            width=width, height=height,
            title="PCoA Plot",
            xaxis_title="PCoA1", yaxis_title="PCoA2",
            font=dict(size=18),
        )
        fig.show()
        #return fig

    # =========================================================
    # Public API: run full pipeline (single ROI)
    # =========================================================
    def run(
        self,
        roi: Tuple[float, float, float, float],
        *,
        bin_size: Optional[int] = None,
        region_col: Optional[str] = None,
        contour_bounds: Optional[Tuple[float, float, float, float]] = None,
        invert_y: Optional[bool] = None,
        make_inside_negative: Optional[bool] = None,
        make_plots: bool = False,
    ) -> Dict[str, Any]:

        # override defaults if provided
        if bin_size is not None: self.bin_size = bin_size
        if region_col is not None: self.region_col = region_col
        if contour_bounds is not None: self.contour_bounds = contour_bounds
        if invert_y is not None: self.invert_y = invert_y
        if make_inside_negative is not None: self.make_inside_negative = make_inside_negative

        self._ensure_bins()
        self.last_roi = roi

        # ------------------
        # 1) ROI extract (grid)
        # ------------------
        self.subset_grid, self.filtered_shortest, self.xy_list = extract_roi(
            grid=self.grid,
            roi=roi,
            bin_size=self.bin_size,
            region_col=self.region_col,
        )

        # ------------------
        # 2) contour extraction
        # ------------------
        min_x, max_x, min_y, max_y = self.contour_bounds
        g_x_cont, g_y_cont, g_x_inside, g_y_inside = contour_regions(
            self.filtered_shortest,
            self.adata,
            min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y,
            out_dir_base=None,
            show=False,
        )
        self.outline_points = list(zip(g_x_cont, g_y_cont))
        self.inside_points  = list(zip(g_x_inside, g_y_inside))

        # ------------------
        # 3) min vectors
        # ------------------
        self.min_vector_df = compute_min_vectors_polar(
            xy_list=self.xy_list,
            outline_points=self.outline_points,
            inside_points=self.inside_points,
            invert_y=self.invert_y,
            make_inside_negative=self.make_inside_negative,
        )

        # ------------------
        # 4) polar distributions
        # ------------------
        self.polar_counts_list, self.selected_genes = self.build_polar_distributions_for_genes(
            adata_grid=self.subset_grid,
            min_vector_df=self.min_vector_df,
            moran_df=None,
            min_gene_count=0,
            make_plots=make_plots,
        )

        # ------------------
        # 7) distance
        # ------------------
        self.dist_df = self.minas_similarity_matrix(
            polar_counts_list=self.polar_counts_list,
            labels=self.selected_genes,
            make_distance=True,
            fill_diagonal_zero=True,
        )

        # ------------------
        # 8) PCoA
        # ------------------
        self.pcoa_res, self.coords, self.explained = self.run_pcoa_from_distance_df(self.dist_df)

        return {
            "subset_grid": self.subset_grid,
            "min_vector_df": self.min_vector_df,
            "selected_genes": self.selected_genes,
            #"polar_counts_list": self.polar_counts_list,
            "dist_df": self.dist_df,
            "pcoa_res": self.pcoa_res,
            "coords": self.coords,
            "explained": self.explained,
        }


# =========================================================
# Usage
# =========================================================
# pipe = SCOMVPipeline(adata=adata, grid=grid, bin_size=10, region_col="region_10")
# out = pipe.run(roi=(min_x, max_x, min_y, max_y))
# pipe.plot_pcoa_plotly(pipe.coords, pipe.selected_genes)


# =========================================================
# fast gene x gene steps (many genes, e.g. whole-transcriptome panels)
# =========================================================
_HI_KERNEL = None


def _histogram_intersection(X: np.ndarray) -> np.ndarray:
    """S[i, j] = sum_k min(X[i, k], X[j, k]) for all row pairs (numba, parallel over rows)."""
    global _HI_KERNEL
    if _HI_KERNEL is None:
        import numba

        @numba.njit(parallel=True, cache=True)
        def _kernel(X):
            n, m = X.shape
            S = np.empty((n, n))
            for i in numba.prange(n):
                for j in range(i, n):
                    s = 0.0
                    for k in range(m):
                        a = X[i, k]
                        b = X[j, k]
                        s += a if a < b else b
                    S[i, j] = s
                    S[j, i] = s
            return S

        _HI_KERNEL = _kernel
    return _HI_KERNEL(np.ascontiguousarray(X, dtype=np.float64))


def _pcoa_top(dist_df: pd.DataFrame, k: int):
    """Leading-``k`` classical PCoA (Gower centring of -D^2/2, ARPACK) as a skbio OrdinationResults."""
    from scipy.sparse.linalg import eigsh
    from skbio import OrdinationResults

    n = len(dist_df)
    k = max(1, min(int(k), n - 2))
    B = np.square(np.asarray(dist_df.values, dtype=np.float64))
    B *= -0.5
    B -= B.mean(axis=0, keepdims=True)
    B -= B.mean(axis=1, keepdims=True)
    total = float(np.trace(B))
    v0 = np.random.default_rng(0).standard_normal(n)
    vals, vecs = eigsh(B, k=k, which="LA", v0=v0)
    del B
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    vals[np.isclose(vals, 0)] = 0.0
    vals = np.clip(vals, 0, None)
    flip = np.sign(vecs[np.abs(vecs).argmax(axis=0), np.arange(vecs.shape[1])])
    vecs = vecs * np.where(flip == 0, 1, flip)
    axes = [f"PC{i}" for i in range(1, k + 1)]
    coords = pd.DataFrame(vecs * np.sqrt(vals), index=list(dist_df.index), columns=axes)
    return OrdinationResults(
        short_method_name="PCoA",
        long_method_name=f"Principal Coordinate Analysis (leading {k} axes, ARPACK)",
        eigvals=pd.Series(vals, index=axes),
        samples=coords,
        proportion_explained=pd.Series(vals / total, index=axes),
    )
