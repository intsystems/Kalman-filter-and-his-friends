# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath('../../src/'))

from kalman import __version__

# -- Project information -----------------------------------------------------

project = 'Kalman filter and his friends'
copyright = '2025, Matvei Kreinin, Maria Nikitina, Petr Babkin, Anastasia Voznyuk'
author = 'Matvei Kreinin, Maria Nikitina, Petr Babkin, Anastasia Voznyuk'

version = __version__
master_doc = 'index'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.doctest', 
              'sphinx.ext.intersphinx', 'sphinx.ext.todo',
              'sphinx.ext.ifconfig', 'sphinx.ext.viewcode',
              'sphinx.ext.inheritance_diagram',
              'sphinx.ext.autosummary', 'sphinx.ext.mathjax',
              'sphinx_rtd_theme']

autodoc_mock_imports = ["numpy", "scipy", "sklearn", "torch", "overrides"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

html_extra_path = []

html_context = {
    "display_github": True, # Integrate GitHub
    "github_user": "Intelligent-Systems-Phystech", # Username
    "github_repo": "Kalman-filter-and-his-friends", # Repo name
    "github_version": "main", # Version
    "conf_py_path": "./doc/source/", # Path in the checkout to the docs root
}


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
