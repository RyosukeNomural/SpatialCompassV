# SpatialCompassV
<p align="left">
  <img
    src="https://raw.githubusercontent.com/RyosukeNomural/SpatialCompassV/main/images/logo.png"
    width="370"
    height="145"
    alt="SCOMV logo"
  />
</p>



![PyPI version](https://img.shields.io/pypi/v/scomv.svg)
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
    <td style="vertical-align: top; width: 200px; border: none;">
        <img width="800" height="400" alt="polar_map" src="https://raw.githubusercontent.com/RyosukeNomural/SpatialCompassV/main/images/polar.png" />
    </td>
  </tr>
</table>


<table border="0" style="border-collapse: collapse; border: none;">
  <tr>
    <td style="vertical-align: top; padding-right: 14px; border: none;">
      A <b>similarity matrix</b> is then constructed, followed by <b>PCoA and clustering</b>,
      to classify spatial distribution patterns.
    </td>
    <td style="vertical-align: top; width: 300px; border: none;">
      <img width="1491" height="638" alt="PCoA"
        src="https://raw.githubusercontent.com/RyosukeNomural/SpatialCompassV/main/images/pcoa.png" />
    </td>
  </tr>
</table>



- **Integration across multiple fields of view**  
  By integrating results from multiple regions of interest, clustering of the reference region itself (e.g., tumor malignancy states) can be performed.  
  - Gene-wise contributions are calculated using **PCA**, enabling the identification of **spatially differentially expressed genes (Spatial DEGs)**.

### Additional functionality
- Gene distributions can also be visualized as **3D density maps**, allowing direct comparison of the spatial distributions of two genes.
<p>
  <img src="https://raw.githubusercontent.com/RyosukeNomural/SpatialCompassV/main/images/overview.png"
       alt="overview"
       width="700"/>
</p>





## Credits

This package was created with [Cookiecutter](https://github.com/audreyfeldroy/cookiecutter) and the [audreyfeldroy/cookiecutter-pypackage](https://github.com/audreyfeldroy/cookiecutter-pypackage) project template.
>>>>>>> 79d3344 (Initial commit (cookiecutter-scientific-python))
