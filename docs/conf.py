# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'SpatialCompassV'
copyright = '2025, Ryosuke Nomura'
author = 'Ryosuke Nomura'
release = '0.1.0'
html_logo = "images/logo.png"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'nbsphinx',
    'sphinx_design',
]

nb_execution_mode = "off"

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
autosummary_generate = True

exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']
html_css_files = ['css/custom.css']

def setup(app):
    app.add_css_file("css/custom.css")

import os
import sys
sys.path.insert(0, os.path.abspath("../src"))

autodoc_mock_imports = [
    "scanpy",
    "anndata",
    "squidpy",
    "stlearn",
    "plotly",
]

