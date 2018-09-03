# coding: utf-8

"""Setup file.

Install the modules.
"""

from setuptools import setup, find_packages

setup(
    name="sgcn",
    version="0.1",
    packages=find_packages(),
    author="CERC DS4DM",
    license="MIT",
    description="Sparse graph neural networks in PyTorch",
    long_description=open("README.md").readlines(),
    install_requires=["torch"]
)
