# History

## 0.1.7 (2026-11-23)

### New

* **`plot_tumor_solids`** (`scomv.tumor_solid`): shows the tumor mask with each
  solid's bounding box and id number. The numbers are drawn as matplotlib text,
  so they stay sharp at any figure size and dpi (the numbers burnt into
  `solid.uns["individual_tumor_solid"]` were pixelated). `extract_tumor_solids`
  now also stores the plain mask image (`solid.uns["tumor_mask_image"]`) and the
  boxes (`solid.uns["solid_bbox"]`) that it uses; `individual_tumor_solid` is
  unchanged.
* **Gene signatures around the tumor** (`scomv.signature`):
  `signature_polar_enrichment` maps a gene list (or several, as a dict) in the
  polar space: per-square signature score (normalise, log1p, z-score, mean of
  the member genes), log2 enrichment per polar cell relative to the tissue
  (30° sectors; outer radius `r_max` and ring width `r_step` selectable;
  `include_inside=True` adds the tumor as one cell) and a permutation test
  against decile-matched random gene sets (BH within each signature `q` and
  across all signatures x cells `q_global`). `hallmark_polar_enrichment` runs
  all MSigDB Hallmark sets (`load_hallmark_gene_sets`, cached JSON);
  `plot_signature_polar`, `plot_significant_hallmarks` (only the sets with a
  significant cell, optionally by Hallmark category) and
  `plot_signature_backprojection` draw the results. Works on a run
  `SCOMVPipeline` or on one tumor solid (`PolarROI.from_solid`).
* **Gap infiltration score for any reference gene** (`scomv.pgi`):
  `compute_pgi(gene_pipe, ref_gene=...)` (POSTN by default) ranks the genes
  concentrated in the stromal sectors where the reference gene is thin (PGI,
  rotation z, mutual-exclusion coefficient); `plot_pgi_ranking`,
  `plot_pgi_polar`.
* **Gene clustering step by step** (`scomv.gene_clustering`):
  `compute_gene_features` (PCoA, uniformity, inside/outside density ratio A),
  `cluster_genes_two_stage` (internal genes by PCoA + A, the rest by
  PCoA + uniformity), `plot_gene_clusters`, `cluster_mean_maps` /
  `plot_cluster_mean_polar` / `plot_cluster_presence` and
  `cluster_enrichment` (GO per cluster).
* **Grouping tumor solids** (`scomv.solid_clustering`):
  `solid_gene_uniformity` (distance-weighted uniformity of any genes around
  every solid, size-factor corrected by default), `solid_cell_uniformity` (the
  same for the selected cell types), `cluster_solids` (Gaussian mixture with k
  by BIC, or Ward; any subset of the columns) and `plot_solid_clusters`.
* **`add_distance_regions`** (`scomv.preparation.choose_roi`): adds the
  `region` / `region_10` columns that `extract_roi` needs (the block the
  tutorials built by hand).

### Faster

* **Gene x gene steps for whole-transcriptome panels.**
  `SCOMVPipeline.minas_similarity_matrix` computes the histogram intersection
  of all gene pairs with a parallel numba kernel (same numbers up to
  floating-point summation order, < 1e-14), and
  `SCOMVPipeline.run_pcoa_from_distance_df` computes only the leading 10 axes
  (ARPACK) when there are more than 3,000 genes (`full_max_n`) or when
  `dimensions=` is given: same eigenvalues and coordinates as the full
  decomposition (each axis up to its sign), with `proportion_explained`
  relative to the trace of the centred matrix (as skbio's `method="fsvd"`).
  Up to 3,000 genes the PCoA is skbio's full decomposition as before, so
  results for targeted panels are unchanged. On the Xenium WTA breast sample
  (17,878 genes) the two steps take ~10 s instead of ~11 min.

### Changes that affect results (only with `mask_other_solids=True`)

* **Other tumors are masked completely.** `mask_other_solids=True`
  (`compute_solid_min_vectors`, `compute_per_solid_polar_counts`,
  `compute_expression_fold_factor`, `compute_gene_uniformity(_pair)`) now also
  removes the cavities (lumens) enclosed by the other solids and their outermost
  rim just outside the boundary contour; before, only their interior squares
  were removed, so lumens and rims were still counted as this solid's stroma.
  The squares come from the solid label image, stored by `extract_tumor_solids`
  as `solid.uns["solid_labels"]`. Results with the default
  `mask_other_solids=False` are unchanged.

### Fixes

* **`plot_3d`**: hovering over the 3D map no longer draws white contour lines on
  the surfaces or black spike lines to the axes (Plotly's hover contour
  highlight and axis spikes are now turned off). The hover tooltip is unchanged.

### Documentation

* Gene-analysis tutorial: removed the GO enrichment run without a background
  gene set; only the run with this panel's genes as background is kept.
* Tumor-solid tutorial: the solid overview figure uses `plot_tumor_solids`.
* Gene-analysis, tumor-solid and cell-analysis tutorials: repeated computation
  and plotting code replaced by the new functions (same results). The
  gene-analysis Hallmark maps now use 30° sectors (previously 45°).
* New tutorials: signature_analysis, gap_infiltration, gene_clustering,
  solid_gene_clustering and solid_cell_clustering.

## 0.1.6 (2026-11-21)

### Fixes that change results (re-run analyses made with 0.1.5)

* **Grid-square indexing is now consistent across all steps** (new module
  `scomv.preparation.grid_index`).
  * stlearn numbers grid squares from 0 (`grid_0`, ...), but
    `calculate_distance` / `extract_tumor_solids` assumed 1, so the SKNY tumour
    mask, `grid.shortest` and the solid label image were shifted by one square
    (10 µm) along y relative to the expression. The numbering is now detected
    from the square centres (0- or 1-based). In notebooks, build square names
    with `grid_index_names(grid)` instead of `"grid_" + str(i + 1)`.
  * Grid squares are indexed from their names instead of
    `coordinate // bin_size` (`compute_solid_min_vectors`, `extract_roi`),
    which drifted by one square once the edge offset and the non-integer square
    width (e.g. 10.0013 µm) added up.
  * Cells are placed in grid squares with the grid's own edges when a `grid` is
    passed: `annotate_cells(..., grid=)`, `assign_solid_ids(..., grid=)`,
    `CellPolarPipeline(..., grid=)`; `compute_per_solid_polar_counts` does this
    automatically. With `floor(x / bin_size)` only ~45 % of the breast Xenium
    cells landed in their own square and ~30 % had no distance vector (they were
    dropped from the polar histograms but still counted in n).
  * Effect on the tutorial data (0.1.5 vs 0.1.6): per-solid cell-type
    uniformity rank correlation 0.91–0.99, values +0.01 to +0.06; Ward clusters
    of solids ARI 0.64 (5 of 40 solids move); gene uniformity (WU) Spearman
    0.997; spatial DEGs 140 of 145 shared.
* **Gene-analysis tutorial**: internal genes are now decided per gene
  (A ≥ `A_THRESH`) and then clustered among themselves, instead of keeping the
  Stage-1 clusters whose mean A passes the threshold. The cluster-mean rule let
  low-A genes (e.g. POSTN, A ≈ 0.1) into an internal cluster.
* `calculate_distance` rendered the tumour-grid image through a matplotlib
  figure of N/100 inch at 100 dpi, which is not pixel-exact for some grid sizes
  (e.g. N_COL = 987 shifted the mask by one square) and differed between
  backends. The colours are now mapped directly. Unchanged for the breast
  (752 x 547) and lung (561 x 564) datasets.
* `label_solid_regions` cast the raw label image to uint8, so with ≥ 256
  connected components every component whose raw id was a multiple of 256 was
  dropped. Unchanged for the breast dataset; the lung dataset at `min_area=50`
  now keeps 21 solids instead of 20.

### New

* `scomv.polar_plot.plot_polar_heatmap`: draws a radius x angle histogram as a
  filled circular ("Baumkuchen") map, with negative radii inside a dashed
  boundary circle, NaN masking and cell outlines. Defaults: 300 dpi
  (`min_dpi`), white dashed boundary, centre hole (`hole_frac=0.15`), unit on
  the outermost radius label (`radius_unit="µm"`), `cbar_pad`.
  `plot_gene_polar_hist2d` and `make_plots=True` in `SCOMVPipeline` /
  `build_polar_distributions_for_genes` now use it (`style="rect"` gives the
  previous rectangle).
* `mask_other_solids` option for `compute_solid_min_vectors`,
  `compute_per_solid_polar_counts`, `compute_expression_fold_factor` and
  `compute_gene_uniformity(_pair)`: drop the grid squares (and cells) inside
  other solids from a solid's field, keeping distances/angles from the solid's
  own boundary (default False).
* `keep_noise` option for `cluster_genes_by_pcoa`: keep HDBSCAN noise genes
  unassigned (label -1) instead of soft-assigning them (default False).
* `scomv.save_preprocessed` / `scomv.load_preprocessed`: save the grid after
  `calculate_distance` (including `grid.shortest` and `uns` images) to one h5ad
  file and reload it exactly.
* Plot options: `plot_go_enrichment_per_cluster` (`wrap_width`,
  `height_by_terms`, `xtick_fontsize`), `plot_gene_polar_hist2d` (`dpi`,
  `show`), `plot_gene_uniformity_scatter` (`label_fontsize`); the legends of
  `plot_cluster_map` / `plot_cell_type_over_clusters` are placed outside the
  axes.

### Speed and memory (outputs identical to 0.1.5)

* `calculate_distance` (SKNY): the pixel graph and multi-source Dijkstra use
  `scipy.sparse.csgraph` instead of networkx (whole breast sample:
  75 s -> 0.7 s).
* `compute_min_vectors_polar`: KD-tree candidates instead of the dense
  N x M distance matrix, same tie rule and order (whole breast sample:
  ~137 GB -> < 0.1 GB).
* `build_polar_distributions_for_genes`: flattened arrays instead of a per-cell
  `DataFrame.iloc` loop (10–50x faster); histograms with `np.histogram2d`, so
  no figure is created per gene.
* `scomv.gene.plot_gene_polar_hist2d` is now the same function as
  `scomv.gene_view_tool.plot_gene_polar_hist2d` (duplicate removed).

## 0.1.5 (2026-08-03)

* Add `scomv.tumor_solid`: tumor-solid extraction, per-solid uniformity, Ward
  clustering, two-tumor gene-level / DEG comparison, and pathway enrichment.
* Add `scomv.gene_module`: PCoA-based gene clustering (inside/outside density
  ratio + uniformity), presence-fraction maps, and GO enrichment per cluster.
* Fix a bug where clustering weights were applied before z-score
  standardization, making them no-ops for any positive weight.

## 0.1.0 (2025-12-18)

* First release on PyPI.
