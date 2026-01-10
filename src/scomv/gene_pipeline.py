from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scanpy as sc
import squidpy as sq
import skbio
from skbio.stats.ordination import pcoa
import plotly.express as px

from .preparation.choose_roi import extract_roi, contour_regions
from .preparation.scomv_calc_vector import compute_min_vectors_polar


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
    subsample_fraction: float = 0.5
    min_gene_percentile: float = 30
    max_gene_percentile: float = 95
    require_moran_nonneg: bool = True

    contour_bounds: Tuple[float, float, float, float] = (0, 10000, 0, 10000)
    invert_y: bool = True
    make_inside_negative: bool = True

    # ---- polar bins ----
    radius_bins: Optional[np.ndarray] = None
    angle_bins_deg: Optional[np.ndarray] = None

    # ---- cached / last run results ----
    last_roi: Optional[Tuple[float, float, float, float]] = None
    subset_grid: Any = None
    filtered_shortest: Any = None
    xy_list: Any = None
    outline_points: Optional[List[Tuple[float, float]]] = None
    inside_points: Optional[List[Tuple[float, float]]] = None
    min_vector_df: Optional[pd.DataFrame] = None
    moran_df: Optional[pd.DataFrame] = None
    gene_lens: Optional[np.ndarray] = None
    p_low: Optional[int] = None
    p_high: Optional[int] = None
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
    # Step 0: ROI subset for adata (Moran用)
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

            gene_angles = []
            gene_radiis = []
            gene_weights = []

            for i in range(n_cells):
                # ここは元コード互換（int化）。正規化値を使うなら int を外して float にしてOK
                e = int(expr[i])
                if e == 0:
                    continue

                angles = min_vector_df["angle"].iloc[i]
                radiis = min_vector_df["radii"].iloc[i]
                n_vecs = len(angles)
                if n_vecs == 0:
                    continue

                w = e / n_vecs
                for ang, rad in zip(angles, radiis):
                    gene_angles.append(ang)
                    gene_radiis.append(rad * bin_um)
                    gene_weights.append(w)

            if len(gene_angles) == 0:
                continue

            deg = np.degrees(gene_angles)
            w = np.asarray(gene_weights, dtype=float)
            w = w / w.sum()

            fig = plt.figure(figsize=(6, 6)) if make_plots else plt.figure(figsize=(1, 1))
            counts, _, _, img = plt.hist2d(
                gene_radiis, deg,
                bins=[self.radius_bins, self.angle_bins_deg],
                weights=w,
                cmap="viridis"
            )

            if make_plots:
                max_val = np.max(counts) if counts.size else 0
                if max_val > 0:
                    img.set_clim(0, max_val * float(clim_ratio))
                plt.title(f"{gene_name} (n={total_n})")
                plt.tight_layout()
                plt.show()

            plt.clf()
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
        n = len(labels)
        table = np.zeros((n, n), dtype=float)

        for i in range(n):
            A = np.asarray(polar_counts_list[i])
            for j in range(i, n):
                B = np.asarray(polar_counts_list[j])
                sim = float(np.minimum(A, B).sum())
                table[i, j] = sim
                table[j, i] = sim

        df = pd.DataFrame(table, index=labels, columns=labels)

        if make_distance:
            df = 1.0 - df
        if fill_diagonal_zero:
            np.fill_diagonal(df.values, 0.0)

        return df

    # ----------------------------
    # Step 5: PCoA
    # ----------------------------
    @staticmethod
    def run_pcoa_from_distance_df(dist_df: pd.DataFrame):
        dm = skbio.DistanceMatrix(dist_df.values, ids=list(dist_df.index))
        res = pcoa(dm)
        coords = res.samples
        explained = res.proportion_explained
        return res, coords, explained

    # ----------------------------
    # Plot: PCoA Plotly
    # ----------------------------
    @staticmethod
    def plot_pcoa_plotly(coords: pd.DataFrame, labels: List[str], width: int = 1200, height: int = 900, margin: float = 0.05):
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
        # return fig

    # =========================================================
    # Public API: run full pipeline (single ROI)
    # =========================================================
    def run(
        self,
        roi: Tuple[float, float, float, float],
        *,
        bin_size: Optional[int] = None,
        region_col: Optional[str] = None,
        subsample_fraction: Optional[float] = None,
        min_gene_percentile: Optional[float] = None,
        max_gene_percentile: Optional[float] = None,
        require_moran_nonneg: Optional[bool] = None,
        contour_bounds: Optional[Tuple[float, float, float, float]] = None,
        invert_y: Optional[bool] = None,
        make_inside_negative: Optional[bool] = None,
        make_plots: bool = False,
    ) -> Dict[str, Any]:

        # override defaults if provided
        if bin_size is not None: self.bin_size = bin_size
        if region_col is not None: self.region_col = region_col
        if subsample_fraction is not None: self.subsample_fraction = subsample_fraction
        if min_gene_percentile is not None: self.min_gene_percentile = min_gene_percentile
        if max_gene_percentile is not None: self.max_gene_percentile = max_gene_percentile
        if require_moran_nonneg is not None: self.require_moran_nonneg = require_moran_nonneg
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
        # 4) Moran
        # ------------------
        self.moran_df = self.compute_moran_df(
            roi=roi,
            subsample_fraction=self.subsample_fraction,
            plot_cdf=False,
        ).sort_index()

        # ------------------
        # 5) gene lens + thresholds
        # ------------------
        self.gene_lens = self.gene_total_counts_int(self.subset_grid)
        self.p_low = int(np.percentile(self.gene_lens, self.min_gene_percentile))
        self.p_high = int(np.percentile(self.gene_lens, self.max_gene_percentile))

        # ------------------
        # 6) polar distributions
        # ------------------
        self.polar_counts_list, self.selected_genes = self.build_polar_distributions_for_genes(
            adata_grid=self.subset_grid,
            min_vector_df=self.min_vector_df,
            moran_df=self.moran_df,
            min_gene_count=self.p_low,
            require_moran_nonneg=self.require_moran_nonneg,
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
            "moran_df": self.moran_df,
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
