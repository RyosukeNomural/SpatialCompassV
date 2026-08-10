<p align="left">
  <img
    src="https://raw.githubusercontent.com/RyosukeNomural/SpatialCompassV/main/images/logo.png"
    width="220"
    height="86"
    alt="SCOMV logo"
    align="left"
    style="margin-right: 16px;"
  />
  <h1>SpatialCompassV</h1>
</p>

![PyPI version](https://img.shields.io/pypi/v/scomv.svg?dummy=99)
[![Documentation Status](https://readthedocs.org/projects/spatialcompassv/badge/?version=latest)](https://spatialcompassv.readthedocs.io/en/latest/?badge=latest)


Spatial omics analysis tools for cell/gene clustering from a astandard region

* PyPI package: https://pypi.org/project/scomv/
* Free software: MIT License
* Documentation: https://spatialcompassv.readthedocs.io


## Overview of the SpatialCompassV (SCOMV) Workflow

The overall workflow of **SpatialCompassV (SCOMV)** is summarized as follows:

- **Extraction of a reference region**  
  A reference region (e.g., a tumor region) is identified using the **[SpatialKnifeY (SKNY)](https://github.com/shusakai/skny)** algorithm.

### Vector construction from spatial grids

<table border="0" style="border-collapse: collapse; border: none;">
  <tr>
    <td style="vertical-align: top; padding-right: 14px; border: none;">
      The AnnData object is discretized into spatial grids, and for each grid,
      the shortest-distance vector to the reference region is computed.
    </td>
    <td style="vertical-align: top; width: 200px; border: none;">
      <img width="200" alt="vector"
           src="https://raw.githubusercontent.com/RyosukeNomural/SpatialCompassV/main/images/vector.png" />
    </td>
  </tr>
</table>

<table border="0" style="border-collapse: collapse; border: none;">
  <tr>
    <td style="vertical-align: top; padding-right: 14px; border: none;">
      This vector information is stored for each cell/gene and projected onto a
      <b>polar coordinate map</b>.
      The horizontal axis represents distance, and the vertical axis also represents distance. 
      Distances are defined as negative for locations inside the reference region.
    </td>
    <td style="vertical-align: top; border: none;">
        <img
          alt="polar_map"
          src="https://raw.githubusercontent.com/RyosukeNomural/SpatialCompassV/main/images/polar.png"
          style="width:800px; height:auto; display:block;"
        />
    </td>
  </tr>
</table>

<table border="0" style="border-collapse: collapse; border: none;">
  <tr>
    <td style="vertical-align: top; padding-right: 14px; border: none;">
      A <b>uniformity</b> score is also computed for each distance bin, measuring how
      evenly a cell type or gene surrounds the reference region across all angular
      directions. High uniformity indicates a symmetric, ring-like distribution, while
      low uniformity means the signal is concentrated on one side.
    </td>
    <td style="vertical-align: top; border: none;">
      <img
        alt="uniformity"
        src="https://raw.githubusercontent.com/RyosukeNomural/SpatialCompassV/main/images/uniformity.png"
        style="width:1000px; height:auto; display:block;"
        />
    </td>
  </tr>
</table>


<table border="0" style="border-collapse: collapse; border: none;">
  <tr>
    <td style="vertical-align: top; padding-right: 14px; border: none;">
      A <b>similarity matrix</b> is then constructed, followed by <b>PCoA and clustering</b>,
      to classify spatial distribution patterns.
    </td>
    <td style="vertical-align: top; border: none;">
      <img
        alt="PCoA"
        src="https://raw.githubusercontent.com/RyosukeNomural/SpatialCompassV/main/images/pcoa.png"
        style="width:800px; height:auto; display:block;"
        />
    </td>
  </tr>
</table>



### Additional functionality
- Gene distributions can also be visualized as **3D density maps**, allowing direct comparison of the spatial distributions of two genes.
<p>
  <img src="https://raw.githubusercontent.com/RyosukeNomural/SpatialCompassV/main/images/overview.png"
       alt="overview"
       width="700"/>
</p>





## Installation

### Stable release

To install SpatialCompassV, run this command in your terminal:

```sh
pip install scomv
```

### From source

The source files for SpatialCompassV can be downloaded from the [Github repo](https://github.com/RyosukeNomural/SpatialCompassV).

You can either clone the public repository:

```sh
git clone git://github.com/RyosukeNomural/SpatialCompassV
```

Once you have a copy of the source, you can install it with:

```sh
cd SpatialCompassV
pip install .
```


## Test Dataset & Quick Start

This repository includes a small example dataset — a 10x Genomics Xenium
human breast cancer FFPE section, from the
[10x Genomics Xenium Human Breast Cancer preview dataset](https://www.10xgenomics.com/jp/products/xenium-in-situ/preview-dataset-human-breast) —
under [`docs/tutorials/tutorial_data/`](https://github.com/RyosukeNomural/SpatialCompassV/tree/main/docs/tutorials/tutorial_data),
so SCOMV can be run end-to-end without downloading anything separately.

Open any of the tutorial notebooks below in Jupyter; each one loads the test
dataset above and runs a full SCOMV pipeline on it:

```bash
git clone https://github.com/RyosukeNomural/SpatialCompassV.git
cd SpatialCompassV
jupyter notebook docs/tutorials/gene_analysis/index.ipynb   # or cell_analysis / tumor_solid_analysis
```

**[Tumor Clustering based on Uniformity](https://github.com/RyosukeNomural/SpatialCompassV/blob/main/docs/tutorials/tumor_solid_analysis/index.ipynb)**
— input: `tutorial_data/xenium_data/` + `cell.xlsx` → output: tumor "solid" region clusters by immune-cell uniformity

<img src="https://raw.githubusercontent.com/RyosukeNomural/SpatialCompassV/main/docs/_static/images/tumor_solid_analysis.png" width="320" alt="tumor_solid_analysis example output"/>

**[Cell Distribution Clustering](https://github.com/RyosukeNomural/SpatialCompassV/blob/main/docs/tutorials/cell_analysis/index.ipynb)**
— input: `tutorial_data/xenium_data/` + `Cell_Barcode_Type_Matrices.xlsx` → output: cell-type groupings by distance-to-tumor pattern

<img src="https://raw.githubusercontent.com/RyosukeNomural/SpatialCompassV/main/docs/_static/images/cell_analysis.png" width="320" alt="cell_analysis example output"/>

**[Gene Distribution Clustering](https://github.com/RyosukeNomural/SpatialCompassV/blob/main/docs/tutorials/gene_analysis/index.ipynb)**
— input: `tutorial_data/xenium_data/` → output: gene clusters by spatial distribution pattern

<img src="https://raw.githubusercontent.com/RyosukeNomural/SpatialCompassV/main/docs/_static/images/gene_analysis.png" width="320" alt="gene_analysis example output"/>

For full step-by-step instructions, parameters, and expected intermediate
results, see the tutorials: https://spatialcompassv.readthedocs.io/en/latest/tutorials.html


## Credits

This package was created with [Cookiecutter](https://github.com/audreyfeldroy/cookiecutter) and the [audreyfeldroy/cookiecutter-pypackage](https://github.com/audreyfeldroy/cookiecutter-pypackage) project template.
