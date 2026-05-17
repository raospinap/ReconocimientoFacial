# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'PROTOTIPO FUNCIONAL DE SISTEMA DE CONTROL DE ASISTENCIA A CLASES MEDIANTE RECONOCIMIENTO FACIAL BIOMÉTRICO'
copyright = '2026, Daniela Martínez, Juan Diego Amaya, Santiago Quiroga, Rolando Ospina'
author = 'Daniela Martínez, Juan Diego Amaya, Santiago Quiroga, Rolando Ospina'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',      # Extrae docstrings de Python
    'sphinx.ext.napoleon',     # Soporta formatos Google/NumPy (más legibles)
    'sphinx.ext.viewcode',     # Añade enlaces al código fuente
    #'sphinx_rtd_theme'         # Tema visual profesional
]

templates_path = ['_templates']
exclude_patterns = []

language = 'es'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

import os
import sys
# dos niveles arriba de docs/source/  para que encuentre el archivo gui_app.py
sys.path.insert(0, os.path.abspath('../..'))

