#!/usr/bin/env python
# -*- coding: utf-8 -*-

import setuptools
from setuptools import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read().replace('.. :changelog:', '')

requirements = [
    'Cython',
    'numpy',
    'matplotlib',
    'ciabatta',
    'agaro',
    'spatious',
    'cellulist',
    'fealty',
    'clustrous',
]

test_requirements = [
    'Cython',
    'numpy',
    'ciabatta',
    'agaro'
    'spatious',
    'cellulist',
    'fealty',
    'clustrous',
]

extensions = cythonize([
    Extension("bannock.numerics", ["bannock/numerics.pyx"],
              include_dirs=[numpy.get_include()]),
])

console_scripts = [
    'bplot_chi_fs = bannock.utils.scripts:plot_chi_fs',
    'bplot_chi_ks = bannock.utils.scripts:plot_chi_ks',
    'bplot_t_fracs = bannock.utils.scripts:plot_t_fracs',
    'bplot_t_ks = bannock.utils.scripts:plot_t_ks',
    'bplot_t_pmeans = bannock.utils.scripts:plot_t_pmeans',
    'bplot_vis = bannock.utils.scripts:plot_vis',
]

setup(
    name='bannock',
    version='0.4.0',
    description="Agent-based simulations in confined environments",
    long_description=readme + '\n\n' + history,
    author="Elliot Marsden",
    author_email='elliot.marsden@gmail.com',
    url='https://github.com/eddiejessup/bannock',
    packages=setuptools.find_packages(exclude=['docs', 'tests']),
    include_package_data=True,
    install_requires=requirements,
    license="BSD",
    zip_safe=False,
    keywords='bannock',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
    ],
    test_suite='tests',
    tests_require=test_requirements,
    ext_modules=extensions,
    entry_points={
        'console_scripts': console_scripts,
    }
)
