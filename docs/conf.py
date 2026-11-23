# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'SpatialCompassV'
copyright = '2026, Ryosuke Nomura'
author = 'Ryosuke Nomura'
release = '0.1.7'
html_logo = "images/logo.png"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'nbsphinx',
   # 'myst_parser',
    'myst_nb',
    'sphinx_design',
    'sphinx_copybutton',   # copy button on every code block
]

nb_execution_mode = "off"

# render $...$ / $$...$$ math in notebook markdown (MathJax)
myst_enable_extensions = ["dollarmath", "amsmath"]

# copy button: strip Python / shell prompts so the copied text runs as is
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: "
copybutton_prompt_is_regexp = True
copybutton_only_copy_prompt_lines = False
# code blocks only: no button on notebook outputs (printed text, tables)
copybutton_selector = "div:not(.output) > div.highlight pre"

# source_suffix = {
     #".rst": "restructuredtext",
    # ".md": "markdown",
# }

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

