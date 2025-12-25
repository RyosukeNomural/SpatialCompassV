# SpatialCompassV
<img width="370" height="145" alt="scomv" src="https://github.com/user-attachments/assets/636006da-884f-4577-8c04-0ae0b718e65e" />



![PyPI version](https://img.shields.io/pypi/v/scomv.svg)
[![Documentation Status](https://readthedocs.org/projects/scomv/badge/?version=latest)](https://scomv.readthedocs.io/en/latest/?version=latest)

Spatial omics analysis tools for cell/gene clustering from a astandard region

* PyPI package: https://pypi.org/project/scomv/
* Free software: MIT License
* Documentation: https://scomv.readthedocs.io.


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
           src="https://github.com/user-attachments/assets/4e0b175a-51a7-4397-a04c-ae3f53a68f92" />
    </td>
  </tr>
</table>

<table border="0" style="border-collapse: collapse; border: none;">
  <tr>
    <td style="vertical-align: top; padding-right: 14px; border: none;">
      This vector information is stored for each cell/gene and projected onto a
      <b>polar coordinate map</b>.
    </td>
    <td style="vertical-align: top; width: 200px; border: none;">
        <img width="350" height="330" alt="polar_map" src="https://github.com/user-attachments/assets/aee34cef-994b-427c-8947-9bb25518a353" />
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
        src="https://github.com/user-attachments/assets/860a60c9-6cf8-425c-a824-f5c9664c8654" />
    </td>
  </tr>
</table>



- **Integration across multiple fields of view**  
  By integrating results from multiple regions of interest, clustering of the reference region itself (e.g., tumor malignancy states) can be performed.  
  - Gene-wise contributions are calculated using **PCA**, enabling the identification of **spatially differentially expressed genes (Spatial DEGs)**.

### Additional functionality
- Gene distributions can also be visualized as **3D density maps**, allowing direct comparison of the spatial distributions of two genes.
<img width="300" height="230" alt="3dmap" src="https://github.com/user-attachments/assets/aa0893fb-6e65-4125-b6d7-a7168ab7227f" />



## Credits

This package was created with [Cookiecutter](https://github.com/audreyfeldroy/cookiecutter) and the [audreyfeldroy/cookiecutter-pypackage](https://github.com/audreyfeldroy/cookiecutter-pypackage) project template.
>>>>>>> 79d3344 (Initial commit (cookiecutter-scientific-python))
