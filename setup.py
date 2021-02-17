# -*- coding: utf-8 -*-
from setuptools import setup

extras = {
    "test": ["pytest", "pytest-benchmark"],
    "docs": [
        "Sphinx~=3.0",
        "recommonmark>=0.5.0",
        "sphinx_book_theme==0.38.0",
        "nbsphinx",
        "sphinx_copybutton",
    ],
    "dev": [],
}
extras["all"] = sum(extras.values(), [])
extras["dev"] += extras["test"]

setup(extras_require=extras)

