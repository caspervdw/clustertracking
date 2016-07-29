from __future__ import print_function
import os
from setuptools import setup

try:
    descr = open(os.path.join(os.path.dirname(__file__), 'README.md')).read()
except IOError:
    descr = ''

try:
    from pypandoc import convert
    descr = convert(descr, 'rst', format='md')
except ImportError:
    pass


setup_parameters = dict(
    name="clustertracking",
    version='1.0',
    description="An algorithm to track overlapping features in video images",
    author="Casper van der Wel",
    install_requires=['numpy', 'scipy<0.18', 'pandas>=0.17.0',
                      'trackpy>=0.3.0', 'pims>0.3.3'],
    author_email="wel@physics.leidenuniv.nl",
    url="https://github.com/caspervdw/clustertracking",
    packages=['clustertracking',
              'clustertracking.tests'],
    long_description=descr)

setup(**setup_parameters)
