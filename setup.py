#!/usr/bin/env python

from setuptools import setup

setup(name="qcifc",
    version="0.1.6",
    packages = ["qcifc"],
    author="Olav Vahtras",
    author_email="vahtras@kth.se",
    install_requires = ["daltools", "util"],
    description = 'Quantum Chemistry Interface',
    )
