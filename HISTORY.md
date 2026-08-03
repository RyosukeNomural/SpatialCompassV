# History

## 0.1.5 (2026-08-03)

* Add `scomv.tumor_solid`: tumor-solid extraction, per-solid uniformity, Ward
  clustering, two-tumor gene-level / DEG comparison, and pathway enrichment.
* Add `scomv.gene_module`: PCoA-based gene clustering (inside/outside density
  ratio + uniformity), presence-fraction maps, and GO enrichment per cluster.
* Fix a bug where clustering weights were applied before z-score
  standardization, making them no-ops for any positive weight.

## 0.1.0 (2025-12-18)

* First release on PyPI.
