"""
Polar ("Baumkuchen") rendering of SCOMV radius x angle histograms.

SCOMV summarizes a gene (or any non-negative weight, e.g. a multi-gene
signature score) as a 2D histogram over signed distance to the reference
boundary (radius) and direction (angle). ``plot_polar_heatmap`` draws such a
histogram on a polar axis as concentric annular sectors.

``pcolormesh`` connects the corners of each cell with straight lines, so a
coarse angle bin (e.g. 30-45 deg) is drawn as a polygon and leaves a gap to
the circular frame. Each angle bin is therefore split into ``theta_sub``
thin wedges painted with the same value, which renders smooth arcs.

Angle convention: the same as ``compute_min_vectors_polar`` with
``invert_y=True`` -- 0 deg points right and angles increase counter-clockwise
when the tissue is displayed with the image y axis pointing down.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter, MaxNLocator


def _subdivide_angle_edges(angle_bins_deg: np.ndarray, theta_sub: int) -> np.ndarray:
    """Split every angle bin into ``theta_sub`` equal wedges (handles uneven bins)."""
    parts = [np.linspace(a0, a1, theta_sub + 1)[:-1]
             for a0, a1 in zip(angle_bins_deg[:-1], angle_bins_deg[1:])]
    return np.append(np.concatenate(parts), angle_bins_deg[-1])


def plot_polar_heatmap(
    values: np.ndarray,
    radius_bins: Sequence[float],
    angle_bins_deg: Sequence[float],
    ax=None,
    *,
    cmap="viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    norm: Optional[Normalize] = None,
    theta_sub: int = 30,
    hole_frac: float = 0.15,
    mask_color: str = "0.85",
    ring_edges: str | bool = "auto",
    sector_edges: bool = True,
    edge_color: str = "white",
    edge_lw: float = 0.8,
    boundary_radius: Optional[float] = 0.0,
    boundary_kwargs: Optional[dict] = None,
    highlight: Optional[np.ndarray] = None,
    highlight_kwargs: Optional[dict] = None,
    radius_ticks: Optional[Sequence[float]] = None,
    radius_label_angle: float = 75.0,
    radius_unit: Optional[str] = "µm",
    angle_ticks_deg: Sequence[float] = (0, 90, 180, 270),
    center_label: Optional[str] = None,
    colorbar: bool = True,
    cbar_label: Optional[str] = None,
    cbar_pad: float = 0.08,
    title: Optional[str] = None,
    fontsize: float = 14,
    min_dpi: Optional[int] = 300,
) -> Tuple[plt.Axes, object]:
    """
    Draw a radius x angle histogram as a filled polar ("Baumkuchen") map.

    Parameters
    ----------
    values : np.ndarray, shape (n_radius_bins, n_angle_bins)
        Same layout as ``np.histogram2d(radius, angle, bins=[radius_bins,
        angle_bins_deg])[0]`` (e.g. ``SCOMVPipeline.polar_counts_list[i]``).
        NaN cells are drawn in ``mask_color``.
    radius_bins, angle_bins_deg : array-like
        Bin edges (µm, degrees). Negative radii (inside the boundary) are
        allowed; the radial axis spans ``radius_bins[0]`` to ``radius_bins[-1]``.
    ax : matplotlib polar Axes, optional
        Created (with a new figure) if omitted. Must use ``projection="polar"``.
    cmap, vmin, vmax, norm
        Colour mapping, as in ``pcolormesh``.
    theta_sub : int
        Number of thin wedges each angle bin is split into. Larger = smoother arcs.
    hole_frac : float
        Radius of the empty centre, as a fraction of the radial range.
    mask_color : str
        Colour of NaN cells (the axis background).
    ring_edges : "auto" | bool
        Draw thin circles at the radius bin edges. "auto" draws them only
        when there are at most 12 radius bins.
    sector_edges : bool
        Draw thin radial lines at the angle bin edges.
    boundary_radius : float or None
        Radius of the reference boundary (0 = tumor edge). Drawn as a thick
        white dashed circle (override with ``boundary_kwargs``, e.g.
        ``dict(color="black")`` on a light colormap) when it lies strictly
        inside the radial range; None disables it.
    highlight : bool array, shape like ``values``, optional
        Cells to outline (e.g. significant enrichment).
    radius_ticks : array-like, optional
        Radial tick positions (µm). Default: about 4 ticks.
    radius_label_angle : float
        Angle (deg) at which the radial tick labels are drawn.
    radius_unit : str or None
        Unit appended to the outermost radial tick label (e.g. "300 µm").
    center_label : str, optional
        Text written in the empty centre (e.g. "tumor").
    min_dpi : int or None
        Raise the figure's dpi to at least this value. The coloured cells are
        rasterized, so a low dpi (matplotlib's default 100) makes them and the
        circles look jagged. None leaves the figure dpi unchanged.
    colorbar : bool
        Add a colour bar to the axis' figure.
    cbar_pad : float
        Gap between the polar axis and its colour bar (fraction of the axis
        width). Increase it when the 0° label touches the colour bar.

    Returns
    -------
    ax, mesh
        The polar Axes and the ``QuadMesh`` (pass it to ``fig.colorbar`` when
        sharing one colour bar across panels with ``colorbar=False``).
    """
    values = np.asarray(values, dtype=float)
    radius_bins = np.asarray(radius_bins, dtype=float)
    angle_bins_deg = np.asarray(angle_bins_deg, dtype=float)
    n_r, n_a = len(radius_bins) - 1, len(angle_bins_deg) - 1
    if values.shape != (n_r, n_a):
        raise ValueError(f"values must have shape {(n_r, n_a)}, got {values.shape}")

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})
    elif ax.name != "polar":
        raise ValueError('ax must be a polar Axes (projection="polar")')
    if min_dpi is not None and ax.figure.get_dpi() < min_dpi:
        ax.figure.set_dpi(min_dpi)

    r0, r1 = radius_bins[0], radius_bins[-1]
    theta = np.radians(_subdivide_angle_edges(angle_bins_deg, theta_sub))
    T, R = np.meshgrid(theta, radius_bins)
    vals = np.ma.masked_invalid(np.repeat(values, theta_sub, axis=1))

    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_facecolor(mask_color)
    # edgecolors="face" closes the hairline seams between the thin wedges
    mesh = ax.pcolormesh(T, R, vals, cmap=cmap, norm=norm,
                         vmin=None if norm is not None else vmin,
                         vmax=None if norm is not None else vmax,
                         shading="flat", edgecolors="face", linewidth=0.3, rasterized=True)

    t_circ = np.linspace(0, 2 * np.pi, 361)
    if ring_edges == "auto":
        ring_edges = n_r <= 12
    if ring_edges:
        for r in radius_bins[1:-1]:
            ax.plot(t_circ, np.full_like(t_circ, r), color=edge_color, lw=edge_lw, zorder=3)
    if sector_edges:
        for a in np.radians(angle_bins_deg[:-1]):
            ax.plot([a, a], [r0, r1], color=edge_color, lw=edge_lw, zorder=3)
    if boundary_radius is not None and r0 < boundary_radius < r1:
        kw = dict(color="white", lw=3, ls="--", zorder=4)
        kw.update(boundary_kwargs or {})
        ax.plot(t_circ, np.full_like(t_circ, boundary_radius), **kw)

    if highlight is not None:
        highlight = np.asarray(highlight, dtype=bool)
        if highlight.shape != (n_r, n_a):
            raise ValueError(f"highlight must have shape {(n_r, n_a)}, got {highlight.shape}")
        kw = dict(fill=False, edgecolor="black", linewidth=2.5, zorder=5)
        kw.update(highlight_kwargs or {})
        for i, j in zip(*np.where(highlight)):
            a0, a1 = angle_bins_deg[j], angle_bins_deg[j + 1]
            ax.bar(np.radians((a0 + a1) / 2), radius_bins[i + 1] - radius_bins[i],
                   width=np.radians(a1 - a0), bottom=radius_bins[i], **kw)

    ax.set_rorigin(r0 - hole_frac * (r1 - r0))
    ax.set_ylim(r0, r1)
    if radius_ticks is None:
        radius_ticks = [t for t in MaxNLocator(nbins=4).tick_values(r0, r1) if r0 < t <= r1]
    ax.set_rticks(radius_ticks)
    unit_tick = max(radius_ticks) if len(radius_ticks) else None

    def _radius_label(v, _):
        lbl = f"{v:g}".replace("-", "\u2212")
        return f"{lbl} {radius_unit}" if radius_unit and unit_tick is not None and np.isclose(v, unit_tick) else lbl

    ax.yaxis.set_major_formatter(FuncFormatter(_radius_label))
    ax.set_axisbelow(False)  # keep radial tick labels above the mesh (styles may set axisbelow=True)
    ax.set_rlabel_position(radius_label_angle)
    ax.tick_params(axis="y", labelsize=fontsize * 0.8, colors="0.15")
    for lbl in ax.get_yticklabels():   # keep radial labels readable on dark cells
        lbl.set_bbox(dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1))
    ax.set_thetagrids(list(angle_ticks_deg), [f"{int(a)}°" for a in angle_ticks_deg], fontsize=fontsize)
    ax.tick_params(axis="x", pad=fontsize * 0.9)  # keep angle labels (e.g. the ° of 180°) off the disc
    ax.grid(False)

    if center_label:
        ax.text(0.5, 0.5, center_label, transform=ax.transAxes, ha="center", va="center",
                fontsize=fontsize, color="0.35")
    if title:
        ax.set_title(title, fontsize=fontsize * 1.3, weight="bold", pad=fontsize * 1.4)
    if colorbar:
        cbar = ax.figure.colorbar(mesh, ax=ax, fraction=0.046, pad=cbar_pad)
        if cbar_label:
            cbar.set_label(cbar_label, fontsize=fontsize)
        cbar.ax.tick_params(labelsize=fontsize * 0.85)
    return ax, mesh
