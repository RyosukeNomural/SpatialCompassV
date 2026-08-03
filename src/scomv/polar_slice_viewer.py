from __future__ import annotations

from typing import Optional, Any

import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display


def plot_polar_slice_interactive(
    A_counts: np.ndarray,
    gene_name: str = "gene",
    radius_bins: Optional[np.ndarray] = None,
    angle_bins_deg: Optional[np.ndarray] = None,
    subset_grid: Optional[Any] = None,
    min_vector_df: Optional[Any] = None,
    bin_size: int = 10,
    clim_ratio: float = 0.25,
    show_highlight: bool = True,
    expr_values: Optional[np.ndarray] = None,
    expr_label: str = "expression",
    n_total: Optional[float] = None,
) -> None:
    """Interactive polar slice viewer.

    Top row  : 2D heatmap (distance x angle) | polar bar chart or distance bar chart
    Bottom row: 2D spatial map colored by expression, with selected-bin cells overlaid in red/orange

    The bottom row is shown only when subset_grid and min_vector_df are provided.
    """
    if radius_bins is None:
        radius_bins = np.arange(-150, 310, 10)
    if angle_bins_deg is None:
        angle_bins_deg = np.arange(-180, 181, 30)

    dist_centers = (radius_bins[:-1] + radius_bins[1:]) / 2    # (45,)
    angle_centers = (angle_bins_deg[:-1] + angle_bins_deg[1:]) / 2  # (12,)
    n_dist, n_angle = A_counts.shape

    # ---- Precompute per-cell representative distance and angle ----
    has_spatial = subset_grid is not None and min_vector_df is not None
    cell_dist = cell_angle_deg = coords = expr = None

    if has_spatial:
        n_cells = len(min_vector_df)
        _cell_dist = np.full(n_cells, np.nan)
        _cell_angle = np.full(n_cells, np.nan)

        for i in range(n_cells):
            radiis = min_vector_df["radii"].iloc[i]
            angles = min_vector_df["angle"].iloc[i]
            if len(radiis) > 0:
                _cell_dist[i] = float(np.mean(radiis)) * bin_size
                rad = np.asarray(angles, dtype=float)
                _cell_angle[i] = float(np.degrees(
                    np.arctan2(np.mean(np.sin(rad)), np.mean(np.cos(rad)))
                ))

        cell_dist = _cell_dist
        cell_angle_deg = _cell_angle
        coords = subset_grid.obsm["spatial"]  # (n_cells, 2)

        if expr_values is not None:
            expr = np.asarray(expr_values, dtype=float).ravel()
        elif gene_name in subset_grid.var_names:
            gene_idx = int(np.where(subset_grid.var_names == gene_name)[0][0])
            raw = subset_grid.X[:, gene_idx]
            if hasattr(raw, "toarray"):
                raw = raw.toarray().ravel()
            expr = np.asarray(raw, dtype=float).ravel()
        else:
            expr = np.zeros(n_cells)

    # ---- Widgets ----
    mode_btn = widgets.ToggleButtons(
        options=["Angle dist. (fixed distance)", "Distance dist. (fixed angle)"],
        button_style="info",
        layout=widgets.Layout(width="440px"),
    )
    dist_slider = widgets.IntSlider(
        min=0, max=n_dist - 1, value=18,
        description="Dist. bin:",
        continuous_update=False,
        layout=widgets.Layout(width="460px"),
        style={"description_width": "80px"},
    )
    angle_slider = widgets.IntSlider(
        min=0, max=n_angle - 1, value=6,
        description="Angle bin:",
        continuous_update=False,
        layout=widgets.Layout(width="460px"),
        style={"description_width": "80px"},
    )
    dist_label = widgets.Label("")
    angle_label = widgets.Label("")
    highlight_btn = widgets.ToggleButton(
        value=show_highlight,
        description="Highlight ON" if show_highlight else "Highlight OFF",
        button_style="danger" if show_highlight else "",
        layout=widgets.Layout(width="140px", height="32px"),
    )
    output = widgets.Output()

    def _update_labels() -> None:
        di, ai = dist_slider.value, angle_slider.value
        dist_label.value = f"→ {radius_bins[di]:.0f} ~ {radius_bins[di+1]:.0f} μm"
        angle_label.value = f"→ {angle_bins_deg[ai]:.0f}° ~ {angle_bins_deg[ai+1]:.0f}°"

    def _circular_stats(slice_: np.ndarray):
        total = slice_.sum()
        if total == 0:
            return None, None
        p = slice_ / total
        rad = np.deg2rad(angle_centers)
        R_x = float(np.sum(p * np.cos(rad)))
        R_y = float(np.sum(p * np.sin(rad)))
        R_bar = float(np.sqrt(R_x**2 + R_y**2))
        theta_mean = float(np.degrees(np.arctan2(R_y, R_x)))
        return R_bar, (theta_mean + 180) % 360

    _vmax = float(A_counts.max()) * clim_ratio if A_counts.max() > 0 else 1.0

    def _draw_heatmap(ax, highlight_dist=None, highlight_angle=None) -> None:
        im = ax.imshow(
            A_counts.T,
            aspect="auto", origin="lower",
            extent=[radius_bins[0], radius_bins[-1],
                    angle_bins_deg[0], angle_bins_deg[-1]],
            cmap="viridis",
            vmin=0, vmax=_vmax,
        )
        ax.set_xlabel("Distance (μm)", fontsize=12)
        ax.set_ylabel("Angle (°)", fontsize=12)
        ax.set_title(f"{gene_name}  polar histogram", fontsize=13)
        plt.colorbar(im, ax=ax, label="density")
        if highlight_dist is not None:
            d_lo, d_hi, d_cen = highlight_dist
            ax.axvline(d_cen, color="red", lw=1.5, ls="--")
            ax.axvspan(d_lo, d_hi, alpha=0.15, color="red")
        if highlight_angle is not None:
            a_lo, a_hi, a_cen = highlight_angle
            ax.axhline(a_cen, color="orange", lw=1.5, ls="--")
            ax.axhspan(a_lo, a_hi, alpha=0.15, color="orange")

    def _draw_polar(ax, di) -> None:
        d_lo, d_hi = radius_bins[di], radius_bins[di + 1]
        slice_ = A_counts[di, :]
        max_r = float(slice_.max()) if slice_.max() > 0 else 1.0

        ax.set_facecolor("#f0f0f0")  # gray background keeps the full circle always visible
        ax.bar(
            np.deg2rad(angle_centers), slice_,
            width=np.deg2rad(28), bottom=0,
            color="steelblue", alpha=0.75, edgecolor="white", linewidth=0.5,
        )
        ax.set_rmax(max_r * 1.05)
        ax.set_thetalim(-np.pi, np.pi)  # force full 360 deg display

        ax.set_theta_zero_location("E")
        ax.set_theta_direction(1)
        ax.set_xticks(np.deg2rad(angle_centers))
        ax.set_xticklabels([f"{int(a)}°" for a in angle_centers], fontsize=8)

        R_bar, _ = _circular_stats(slice_)
        title = f"{gene_name}  d = {d_lo:.0f}~{d_hi:.0f} μm\n"
        if n_total is not None:
            n_bin = float(slice_.sum()) * n_total
            title += f"n ≈ {n_bin:.0f}   "
        if R_bar is not None:
            title += f"R_bar = {R_bar:.3f}"
        ax.set_title(title, fontsize=11, pad=12)

    def _draw_dist_bar(ax, ai) -> None:
        a_lo, a_hi = angle_bins_deg[ai], angle_bins_deg[ai + 1]
        slice_ = A_counts[:, ai]
        ax.bar(dist_centers, slice_, width=9, color="coral", alpha=0.75, edgecolor="white")
        ax.axvline(0, color="gray", lw=1, ls=":", label="boundary")
        ax.set_xlabel("Distance (μm)", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title(f"{gene_name}  θ = {a_lo:.0f}°~{a_hi:.0f}°", fontsize=13)
        ax.legend(fontsize=10)

    def _draw_spatial(ax, mask, highlight_color, subtitle, fig_width_in: float = 8.0, _show_hl: bool = True) -> None:
        # dot size: fill each 10 μm bin as one pixel-ish dot
        x_range = float(coords[:, 0].max() - coords[:, 0].min()) or 1.0
        pts_per_bin = fig_width_in * 72.0 / x_range * bin_size   # points per bin width
        pt_size = max(8.0, pts_per_bin ** 2 * 0.5)

        _vmax_expr = float(expr.max()) * 0.2 if expr.max() > 0 else 1.0
        sc = ax.scatter(
            coords[:, 0], coords[:, 1],
            c=expr, cmap="viridis",
            vmin=0, vmax=_vmax_expr,
            s=pt_size, alpha=0.9, linewidths=0,
        )
        plt.colorbar(sc, ax=ax, label=expr_label, shrink=0.8)
        if _show_hl and mask.any():
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                c=expr[mask], cmap="viridis",
                vmin=0, vmax=_vmax_expr,
                s=pt_size * 2.0,
                linewidths=1.5, edgecolors="red",
                alpha=1.0, zorder=3,
                label=f"selected ({mask.sum()} cells)",
            )
            ax.legend(fontsize=9, loc="upper right")
        ax.set_xlabel("x (μm)", fontsize=11)
        ax.set_ylabel("y (μm)", fontsize=11)
        ax.set_title(f"{gene_name}  |  {subtitle}", fontsize=12)
        margin = bin_size * 2
        ax.set_xlim(coords[:, 0].min() - margin, coords[:, 0].max() + margin)
        ax.set_ylim(coords[:, 1].max() + margin, coords[:, 1].min() - margin)
        ax.set_aspect("equal", adjustable="datalim")

    def draw(_=None) -> None:
        _update_labels()
        with output:
            output.clear_output(wait=True)

            is_angle_mode = mode_btn.value == "Angle dist. (fixed distance)"

            # ---- Figure 1: heatmap (left) + polar or bar (right) ----
            # sufficient height so the polar subplot renders as a full circle
            fig1 = plt.figure(figsize=(13, 5.5))
            ax_heat = fig1.add_subplot(1, 2, 1)

            if is_angle_mode:
                di = dist_slider.value
                d_lo, d_hi = radius_bins[di], radius_bins[di + 1]
                _draw_heatmap(ax_heat, highlight_dist=(d_lo, d_hi, dist_centers[di]))
                ax_pol = fig1.add_subplot(1, 2, 2, projection="polar")
                _draw_polar(ax_pol, di)
            else:
                ai = angle_slider.value
                a_lo, a_hi = angle_bins_deg[ai], angle_bins_deg[ai + 1]
                _draw_heatmap(ax_heat, highlight_angle=(a_lo, a_hi, angle_centers[ai]))
                ax_bar = fig1.add_subplot(1, 2, 2)
                _draw_dist_bar(ax_bar, ai)

            fig1.tight_layout()
            plt.show()

            # ---- Figure 2: spatial map ----
            if has_spatial:
                if is_angle_mode:
                    mask = (cell_dist >= d_lo) & (cell_dist < d_hi)
                    subtitle = f"highlighted: d = {d_lo:.0f}~{d_hi:.0f} μm"
                    color = "red"
                else:
                    mask = (cell_angle_deg >= a_lo) & (cell_angle_deg < a_hi)
                    subtitle = f"highlighted: theta = {a_lo:.0f}~{a_hi:.0f} deg"
                    color = "orange"

                fig2, ax_sp = plt.subplots(1, 1, figsize=(8, 8))
                _draw_spatial(ax_sp, mask, color, subtitle, _show_hl=highlight_btn.value)
                fig2.tight_layout()
                plt.show()

    def on_highlight_change(change) -> None:
        if change["new"]:
            highlight_btn.description = "Highlight ON"
            highlight_btn.button_style = "danger"
        else:
            highlight_btn.description = "Highlight OFF"
            highlight_btn.button_style = ""
        draw()

    def on_mode_change(change) -> None:
        is_dist = change["new"] == "Angle dist. (fixed distance)"
        dist_slider.layout.display = "" if is_dist else "none"
        dist_label.layout.display = "" if is_dist else "none"
        angle_slider.layout.display = "none" if is_dist else ""
        angle_label.layout.display = "none" if is_dist else ""
        draw()

    mode_btn.observe(on_mode_change, names="value")
    dist_slider.observe(draw, names="value")
    angle_slider.observe(draw, names="value")
    highlight_btn.observe(on_highlight_change, names="value")

    angle_slider.layout.display = "none"
    angle_label.layout.display = "none"

    display(widgets.VBox([
        widgets.HBox([mode_btn, highlight_btn]),
        widgets.HBox([dist_slider, dist_label]),
        widgets.HBox([angle_slider, angle_label]),
        output,
    ]))
    draw()
