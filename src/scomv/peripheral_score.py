from __future__ import annotations

from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Representative immune / blood-cell marker genes
# ---------------------------------------------------------------------------
IMMUNE_BLOOD_MARKERS: Dict[str, List[str]] = {
    "T cell":        ["CD3D", "CD3E", "CD3G", "CD8A", "CD8B", "CD4"],
    "Treg":          ["FOXP3", "IL2RA"],
    "B cell":        ["CD19", "MS4A1", "CD79A", "CD79B"],
    "Plasma cell":   ["MZB1", "IGHG1", "IGKC"],
    "NK cell":       ["NCAM1", "KLRD1", "GNLY"],
    "Macrophage":    ["CD68", "CD163", "CD14", "CSF1R"],
    "Dendritic":     ["CLEC9A", "IRF8", "ITGAX"],
    "Neutrophil":    ["S100A8", "S100A9", "MPO", "ELANE"],
    "Mast cell":     ["KIT", "TPSAB1"],
    "Erythrocyte":   ["HBB", "HBA1", "HBA2", "GYPA"],
}


def _default_bins(radius_bins, angle_bins_deg):
    if radius_bins is None:
        radius_bins = np.arange(-150, 310, 10)
    if angle_bins_deg is None:
        angle_bins_deg = np.arange(-180, 181, 30)
    return radius_bins, angle_bins_deg


def _circular_moment(p: np.ndarray, angle_centers_deg: np.ndarray, k: int) -> float:
    """k-th trigonometric moment: rho_k = |sum p * exp(i*k*theta)|"""
    rad = np.deg2rad(angle_centers_deg)
    return float(np.sqrt(np.sum(p * np.cos(k * rad)) ** 2 + np.sum(p * np.sin(k * rad)) ** 2))


def compute_peripheral_score(
    A_counts: np.ndarray,
    d_lo: float = 10.0,
    d_hi: float = 50.0,
    radius_bins: Optional[np.ndarray] = None,
    angle_bins_deg: Optional[np.ndarray] = None,
) -> dict:
    """Compute peripheral score for a single gene.

    Returns
    -------
    dict with keys:
        peripheral_score : coverage * uniformity
        coverage         : fraction of total expression within [d_lo, d_hi]
        uniformity       : normalized Shannon entropy of angle distribution (0=concentrated, 1=uniform)
        rho1             : 1st circular moment (high = unimodal concentration)
        rho2             : 2nd circular moment (high = bimodal; distinguishes adjacent vs opposite)
        profile          : np.ndarray (n_angle,), density summed over [d_lo, d_hi]
    """
    radius_bins, angle_bins_deg = _default_bins(radius_bins, angle_bins_deg)
    dist_centers = (radius_bins[:-1] + radius_bins[1:]) / 2
    angle_centers = (angle_bins_deg[:-1] + angle_bins_deg[1:]) / 2

    mask = (dist_centers >= d_lo) & (dist_centers <= d_hi)
    profile = A_counts[mask, :].sum(axis=0)  # (n_angle,)

    total_all = float(A_counts.sum())
    total_in_range = float(profile.sum())

    empty = {
        "peripheral_score": 0.0, "coverage": 0.0,
        "uniformity": 0.0, "rho1": 0.0, "rho2": 0.0, "profile": profile,
    }
    if total_all == 0 or total_in_range == 0:
        return empty

    coverage = total_in_range / total_all

    p = profile / total_in_range

    # Shannon entropy normalized to [0, 1]
    n_bins = len(p)
    eps = 1e-10
    uniformity = float(-np.sum(p * np.log(p + eps)) / np.log(n_bins))

    rho1 = _circular_moment(p, angle_centers, k=1)
    rho2 = _circular_moment(p, angle_centers, k=2)

    return {
        "peripheral_score": coverage * uniformity,
        "coverage": coverage,
        "uniformity": uniformity,
        "rho1": rho1,
        "rho2": rho2,
        "profile": profile,
    }


def rank_genes_by_peripheral_score(
    polar_counts_list: List[np.ndarray],
    selected_genes: List[str],
    d_lo: float = 10.0,
    d_hi: float = 50.0,
    radius_bins: Optional[np.ndarray] = None,
    angle_bins_deg: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Score all genes and return a DataFrame sorted by peripheral_score (desc)."""
    radius_bins, angle_bins_deg = _default_bins(radius_bins, angle_bins_deg)
    records = []
    for gene_name, A_counts in zip(selected_genes, polar_counts_list):
        res = compute_peripheral_score(A_counts, d_lo, d_hi, radius_bins, angle_bins_deg)
        records.append({
            "gene": gene_name,
            "peripheral_score": res["peripheral_score"],
            "coverage": res["coverage"],
            "uniformity": res["uniformity"],
            "rho1": res["rho1"],
            "rho2": res["rho2"],
        })
    return (
        pd.DataFrame(records)
        .sort_values("peripheral_score", ascending=False)
        .reset_index(drop=True)
    )


def add_pattern_label(
    df: pd.DataFrame,
    peri_quantile: float = 0.67,
    coverage_threshold: float = 0.3,
    uniformity_threshold: float = 0.5,
) -> pd.DataFrame:
    """Add a 'pattern' column for easy interpretation by non-experts.

    Labels
    ------
    Peripheral          : top peripheral_score genes (uniformly near boundary)
    Partially peripheral: high coverage but low uniformity (directional bias)
    Other               : low coverage or ambiguous

    Parameters
    ----------
    peri_quantile        : top fraction classified as Peripheral (default top 33%)
    coverage_threshold   : minimum coverage to be considered near boundary
    uniformity_threshold : below this -> directionally biased
    """
    df = df.copy()
    peri_cutoff = df["peripheral_score"].quantile(peri_quantile)

    def _label(row):
        if row["peripheral_score"] >= peri_cutoff:
            return "Peripheral"
        if row["coverage"] >= coverage_threshold and row["uniformity"] < uniformity_threshold:
            return "Partially peripheral"
        return "Other"

    df["pattern"] = df.apply(_label, axis=1)
    return df


def plot_angle_profile(
    A_counts: np.ndarray,
    gene_name: str = "gene",
    d_lo: float = 10.0,
    d_hi: float = 50.0,
    radius_bins: Optional[np.ndarray] = None,
    angle_bins_deg: Optional[np.ndarray] = None,
    figsize=(8, 4),
) -> None:
    """Bar chart: x=angle, y=density summed over [d_lo, d_hi].

    The dashed red line shows the uniform baseline (profile.mean()).
    Flat bars == high uniformity (entropy) == high peripheral_score.
    A single peak == high rho1 (unimodal, partially peripheral).
    Two opposite peaks == high rho2 + low rho1 (bimodal).
    """
    radius_bins, angle_bins_deg = _default_bins(radius_bins, angle_bins_deg)
    angle_centers = (angle_bins_deg[:-1] + angle_bins_deg[1:]) / 2

    res = compute_peripheral_score(A_counts, d_lo, d_hi, radius_bins, angle_bins_deg)
    profile = res["profile"]

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(
        angle_centers, profile,
        width=25, color="steelblue", alpha=0.78, edgecolor="white", linewidth=0.6,
    )
    if profile.sum() > 0:
        ax.axhline(profile.mean(), color="tomato", ls="--", lw=1.6, label="uniform baseline")
        ax.legend(fontsize=10)

    ax.set_xlabel("Angle (°)", fontsize=13)
    ax.set_ylabel("Density", fontsize=13)
    ax.set_xticks(angle_centers)
    ax.set_xticklabels([f"{int(a)}°" for a in angle_centers], fontsize=11)
    ax.set_title(
        f"{gene_name}   d = {d_lo:.0f}~{d_hi:.0f} μm\n"
        f"peripheral_score={res['peripheral_score']:.4f}   "
        f"coverage={res['coverage']:.3f}   "
        f"uniformity={res['uniformity']:.3f}   "
        f"rho1={res['rho1']:.3f}   rho2={res['rho2']:.3f}",
        fontsize=11,
    )
    plt.tight_layout()
    plt.show()


def plot_marker_profiles(
    polar_counts_list: List[np.ndarray],
    selected_genes: List[str],
    subset_grid=None,
    min_vector_df=None,
    markers: Optional[Dict[str, List[str]]] = None,
    d_lo: float = 10.0,
    d_hi: float = 50.0,
    bin_size: int = 10,
    mode: str = "interactive",
) -> None:
    """Plot polar slice viewer or angle profile for representative marker genes.

    Only genes present in selected_genes are plotted; others are silently skipped.

    Parameters
    ----------
    mode : "interactive"  -> plot_polar_slice_interactive (full viewer with spatial map)
           "profile"      -> plot_angle_profile (angle bar chart only)
    markers : dict mapping cell-type label to list of gene names.
              Defaults to IMMUNE_BLOOD_MARKERS if None.
    """
    from .polar_slice_viewer import plot_polar_slice_interactive

    if markers is None:
        markers = IMMUNE_BLOOD_MARKERS

    gene_to_counts = dict(zip(selected_genes, polar_counts_list))

    for cell_type, gene_list in markers.items():
        for gene in gene_list:
            if gene not in gene_to_counts:
                continue
            print(f"=== {cell_type}: {gene} ===")
            if mode == "interactive":
                plot_polar_slice_interactive(
                    A_counts=gene_to_counts[gene],
                    gene_name=gene,
                    subset_grid=subset_grid,
                    min_vector_df=min_vector_df,
                    bin_size=bin_size,
                )
            else:
                plot_angle_profile(
                    gene_to_counts[gene],
                    gene_name=gene,
                    d_lo=d_lo,
                    d_hi=d_hi,
                )
