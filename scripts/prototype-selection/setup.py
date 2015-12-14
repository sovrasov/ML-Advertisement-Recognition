from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
        Extension(
            'prototype_selection',
            ['prototype_selection.pyx', 'psmethods.c'],
            include_dirs=[numpy.get_include()])
]

setup(
        name = 'prototype_sel',
        version = '0.0.2',
        install_requires = ['numpy'],
        ext_modules = cythonize(extensions)
)

