#!/usr/bin/env python3
from setuptools import find_packages, setup

setup(
    name="instruments",
    license="MIT",
    packages=find_packages(exclude=("tests",)),
    python_requires=">=3.7",
)
