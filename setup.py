# coding: utf-8

"""Setup file.

Installs the modules.
"""

import os
from setuptools import setup, find_packages


__currdir__ = os.getcwd()
__readme__ = os.path.join(__currdir__, "README.md")


install_requires = [
    "attrs",
    "numpy",
    "torch"
]

setup(
    name="sgcn",
    version="0.1",
    packages=find_packages(),
    author="CERC DS4DM",
    license="MIT",
    description="Sparse graph neural networks in PyTorch",
    long_description=open(__readme__).read(),
    install_requires=install_requires
)
