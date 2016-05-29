# -*- coding:utf-8 -*-
from setuptools import setup

setup(
    name = "dmr",
    packages = ["dmr"],
    version = "0.0.1",
    description = "Topic Models with Dirichlet-multinomial Regression",
    author = "Makoto P. Kato",
    author_email = "kato@dl.kuis.kyoto-u.ac.jp",
    license     = "MIT License",
    url = "https://github.com/mpkato/dmr",
    install_requires = [
        'numpy'
    ],
    tests_require=['nose']
)
