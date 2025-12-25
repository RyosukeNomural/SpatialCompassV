from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional, Sequence, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import skbio
from skbio.stats.ordination import pcoa


@dataclass
class CellPolarPipeline:
    """
    A pipeline that takes a cell-level table (x, y, Cluster) and a min_vector_df
    (indexed by grid coordinates), then performs:
      - cluster-wise polar distributions
      - inter-cluster distance computation
      - PCoA / heatmap visualization
    """

    cell_df: pd.DataFrame
    min_vector_df: pd.DataFrame

    # ---- column settings ----
    cluster_col: str = "Cluster"
    x_col: int = 0
    y_col: int = 1

    # ---- grid settings ----
    bin_size: int = 10
    coord_col: str = "coord_tuple"
    unlabeled_name: str = "Unlabeled"
    min_cells: int = 30

    # ---- polar hist settings ----
    bin_size_um: int = 10
    radius_bins: Optional[np.ndarray] = None
    angle_bins_deg: Optional[np.ndarray] = None

    # ---- cached results ----
    last_roi: Optional[Tuple[float, float, float, float]] = None
    cell_df_filtered: Optional[pd.DataFrame] = None
    selected_clusters: Optional[List[str]] = None
    polar_counts_list: Optional[List[np.ndarray]] = None
    counts_by_cluster: Optional[Dict[str, np.ndarray]] = None
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
    # Step 1: ROI filter + coord_tuple
    # ----------------------------
    def annotate_cells(
        self,
        roi: Tuple[float, float, float, float],
        *,
        bin_size: Optional[int] = None,
        coord_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Filter cells within the ROI and add coord_tuple as
        (floor(x/bin), floor(y/bin)).
        """
        if bin_size is None:
            bin_size = self.bin_size
        if coord_col is None:
            coord_col = self.coord_col

        min_x, max_x, min_y, max_y = roi
        df = self.cell_df

        filtered_df = df[
            (df.iloc[:, self.x_col] > min_x) & (df.iloc[:, self.x_col] < max_x) &
            (df.iloc[:, self.y_col] > min_y) & (df.iloc[:, self.y_col] < max_y)
        ].copy()

        filtered_df[coord_col] = list(zip(
            np.floor(filtered_df.iloc[:, self.x_col] / bin_size).astype(int),
            np.floor(filtered_df.iloc[:, self.y_col] / bin_size).astype(int),
        ))

        # Stabilize coord_tuple types (e.g., list -> tuple)
        filtered_df[coord_col] = filtered_df[coord_col].apply(lambda x: tuple(x))

        self.last_roi = roi
        self.cell_df_filtered = filtered_df
        return filtered_df

    # ----------------------------
    # Step 2: cluster polar hist2d
    # ----------------------------
    def compute_cluster_polar_distributions(
        self,
        *,
        cell_df_filtered: Optional[pd.DataFrame] = None,
        plot: bool = True,
        clim_ratio: float = 0.5,
        cmap: str = "viridis",
        xlim: Tuple[float, float] = (-150, 300),
        ylim: Tuple[float, float] = (-180, 180),
    ) -> Tuple[List[np.ndarray], List[str], Dict[str, np.ndarray]]:

        self._ensure_bins()

        if cell_df_filtered is None:
            if self.cell_df_filtered is None:
                raise ValueError(
                    "cell_df_filtered is None. Call annotate_cells(roi) first or pass cell_df_filtered."
                )
            cell_df_filtered = self.cell_df_filtered

        if self.coord_col not in cell_df_filtered.columns:
            raise KeyError(
                f"'{self.coord_col}' not found. Run annotate_cells() first or set coord_col correctly."
            )

        polar_counts_list: List[np.ndarray] = []
        selected_clusters: List[str] = []
        counts_by_cluster: Dict[str, np.ndarray] = {}

        for target_cluster in cell_df_filtered[self.cluster_col].unique():
            if target_cluster == self.unlabeled_name:
                continue

            sub = cell_df_filtered[cell_df_filtered[self.cluster_col].eq(target_cluster)].copy()
            total_n = len(sub)
            if total_n < self.min_cells:
                continue

            cluster_angles = []
            cluster_radiis = []
            cluster_weights = []

            # 1 cell -> 1 grid key -> vector list
            for _, row in sub.iterrows():
                x, y = map(int, row[self.coord_col])
                key = (x, y)

                if key not in self.min_vector_df.index:
                    continue

                angles = self.min_vector_df.at[key, "angle"]
                radiis = self.min_vector_df.at[key, "radii"]

                n_vecs = len(angles)
                if n_vecs == 0:
                    continue

                w = 1.0 / n_vecs
                for ang, rad in zip(angles, radiis):
                    cluster_angles.append(ang)
                    cluster_radiis.append(rad * self.bin_size_um)
                    cluster_weights.append(w)

            if len(cluster_angles) == 0:
                continue

            selected_clusters.append(target_cluster)

            degree_list = np.degrees(cluster_angles)
            w = np.asarray(cluster_weights, dtype=float)
            w = w / w.sum()

            if plot:
                plt.figure(figsize=(6, 6))

            A_counts, _, _, img = plt.hist2d(
                cluster_radiis, degree_list,
                bins=[self.radius_bins, self.angle_bins_deg],
                weights=w,
                cmap=cmap
            )

            if clim_ratio is not None:
                max_val = np.max(A_counts) if A_counts.size else 0
                if max_val > 0:
                    img.set_clim(0, max_val * float(clim_ratio))

            if plot:
                cbar = plt.colorbar(img)
                cbar.set_label("Density", fontsize=15)
                cbar.ax.tick_params(labelsize=12)

                plt.xlabel("Radius", fontsize=15)
                plt.ylabel("Angle", fontsize=15)
                plt.ylim(*ylim)
                plt.yticks(np.arange(-180, 181, 30), fontsize=13)
                plt.xlim(*xlim)
                plt.xticks(fontsize=13)
                plt.grid(True)
                plt.title(f"{target_cluster} (n = {total_n})", fontsize=20)
                plt.tight_layout()
                plt.show()
            else:
                plt.close("all")

            polar_counts_list.append(A_counts)
            counts_by_cluster[target_cluster] = A_counts

        self.polar_counts_list = polar_counts_list
        self.selected_clusters = selected_clusters
        self.counts_by_cluster = counts_by_cluster
        return polar_counts_list, selected_clusters, counts_by_cluster

    # ----------------------------
    # Step 3: similarity/distance matrix
    # ----------------------------
    @staticmethod
    def cell_similarity_matrix(
        polar_counts_list: List[np.ndarray],
        labels: Sequence[str],
        make_distance: bool = True,
        fill_diagonal_zero: bool = True,
    ) -> pd.DataFrame:
        n = len(polar_counts_list)
        if n != len(labels):
            raise ValueError(f"len(polar_counts_list)={n} != len(labels)={len(labels)}")

        mats = [np.asarray(m) for m in polar_counts_list]
        shape0 = mats[0].shape
        if any(m.shape != shape0 for m in mats):
            raise ValueError(
                f"All A_counts must have same shape. First={shape0}, got={[m.shape for m in mats]}"
            )

        table = np.zeros((n, n), dtype=float)
        for i in range(n):
            A = mats[i]
            for j in range(i, n):
                B = mats[j]
                sim = float(np.minimum(A, B).sum())
                table[i, j] = sim
                table[j, i] = sim

        df = pd.DataFrame(table, index=labels, columns=labels).fillna(0)

        if make_distance:
            df = 1.0 - df
        if fill_diagonal_zero:
            np.fill_diagonal(df.values, 0.0)

        return df

    def build_distance(
        self,
        *,
        make_distance: bool = True,
        fill_diagonal_zero: bool = True,
    ) -> pd.DataFrame:
        if self.polar_counts_list is None or self.selected_clusters is None:
            raise ValueError("Run compute_cluster_polar_distributions() first.")

        self.dist_df = self.cell_similarity_matrix(
            polar_counts_list=self.polar_counts_list,
            labels=self.selected_clusters,
            make_distance=make_distance,
            fill_diagonal_zero=fill_diagonal_zero,
        )
        return self.dist_df

    # ----------------------------
    # Step 4: PCoA
    # ----------------------------
    def run_pcoa(self, dist_df: Optional[pd.DataFrame] = None):
        if dist_df is None:
            if self.dist_df is None:
                raise ValueError("dist_df is None. Call build_distance() first or pass dist_df.")
            dist_df = self.dist_df

        dm = skbio.DistanceMatrix(dist_df.values, ids=list(dist_df.index))
        self.pcoa_res = pcoa(dm)
        self.coords = self.pcoa_res.samples
        self.explained = self.pcoa_res.proportion_explained
        return self.pcoa_res, self.coords, self.explained

    def plot_explained_variance(
        self,
        n_components: int = 10,
        figsize: Tuple[int, int] = (8, 4),
        title: str = "PCoA Explained Variance",
        ylabel: str = "Proportion Explained",
        xlabel: str = "PCoA Axis",
        show_values: bool = True,
    ):
        if self.explained is None:
            raise ValueError("Run run_pcoa() first.")

        evr = np.asarray(self.explained)[:n_components]
        x = np.arange(1, len(evr) + 1)

        plt.figure(figsize=figsize)
        bars = plt.bar(x, evr)

        if show_values:
            for b, v in zip(bars, evr):
                plt.text(
                    b.get_x() + b.get_width() / 2,
                    b.get_height(),
                    f"{v:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=10
                )

        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(title, fontsize=14)
        plt.xticks(x, [f"PCoA{i}" for i in x])
        plt.ylim(0, max(evr) * 1.15 if len(evr) else 1)
        plt.tight_layout()
        plt.show()

    # ----------------------------
    # Step 5: Heatmap
    # ----------------------------
    def heatmap(
        self,
        dist_df: Optional[pd.DataFrame] = None,
        method: str = "ward",
        metric: str = "euclidean",
        cmap: str = "viridis",
        figsize: Tuple[int, int] = (10, 10),
        font_size: int = 18,
        rotation: int = 90,
        show_similarity: bool = True,  # True: display 1 - dist_df
    ):
        if dist_df is None:
            if self.dist_df is None:
                raise ValueError("dist_df is None. Call build_distance() first or pass dist_df.")
            dist_df = self.dist_df

        sns.set(context="notebook")

        mat = (1 - dist_df) if show_similarity else dist_df

        g = sns.clustermap(
            mat,
            method=method,
            metric=metric,
            cmap=cmap,
            figsize=figsize
        )

        g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), fontsize=font_size, rotation=rotation)
        g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=font_size)

        plt.show()
        return g

    # ----------------------------
    # One-shot runner
    # ----------------------------
    def run(
        self,
        roi: Tuple[float, float, float, float],
        *,
        plot_hist: bool = False,
        clim_ratio: float = 0.5,
    ) -> Dict[str, Any]:
        self.annotate_cells(roi)
        self.compute_cluster_polar_distributions(plot=plot_hist, clim_ratio=clim_ratio)
        self.build_distance()
        self.run_pcoa()
        return {
            "roi": roi,
            "cell_df_filtered": self.cell_df_filtered,
            "selected_clusters": self.selected_clusters,
            "polar_counts_list": self.polar_counts_list,
            "dist_df": self.dist_df,
            "coords": self.coords,
            "explained": self.explained,
        }


"""
# =========================================================
# Usage
# =========================================================

cell_pipe = CellPolarPipeline(cell_df=cell_df, min_vector_df=min_vector_df)
cell_out = cell_pipe.run(roi=(2400, 3200, 2400, 3800), plot_hist=False)

# Distance matrix
dist = cell_pipe.dist_df

# PCoA explained variance
cell_pipe.plot_explained_variance(n_components=8)

# Heatmap (visualize 1 - dist)
cell_pipe.heatmap(font_size=14, figsize=(8, 8))
"""
