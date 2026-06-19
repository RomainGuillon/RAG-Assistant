"""Configuration Sphinx pour la documentation du projet RAG."""
import os
import sys

# Permet à autodoc de trouver les modules Python du projet
sys.path.insert(0, os.path.abspath(".."))

# -- Informations du projet --------------------------------------------------
project = "Projet RAG"
author = "Romain Guillon"
release = "1.0"
language = "fr"

# -- Extensions --------------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",       # Génère la doc depuis les docstrings
    "sphinx.ext.napoleon",      # Supporte le style Google (Args, Returns, Raises)
    "sphinx.ext.viewcode",      # Ajoute un lien "voir le code source"
    "sphinx.ext.intersphinx",   # Liens vers la doc Python standard
]

intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}

# -- Autodoc -----------------------------------------------------------------
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "private-members": False,
    "show-inheritance": True,
}
autodoc_member_order = "bysource"  # Respecte l'ordre du code source

# -- Napoleon (style Google) -------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True

# -- Options HTML ------------------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 3,
    "collapse_navigation": False,
}
html_static_path = ["_static"]
html_title = "Documentation — Projet RAG"
