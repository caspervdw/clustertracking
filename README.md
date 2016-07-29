clustertracking
===============
An algorithm to track overlapping features in video images

[![build status](https://travis-ci.org/caspervdw/clustertracking.png?branch=master)](https://travis-ci.org/caspervdw/clustertracking)

This package contains Python code for tracking overlapping features in video images,
such as present in for example confocal images of colloidal clusters. The code
is heavily based on [Trackpy](http://github.com/soft-matter/trackpy).

Installation
------------
If you are new to python, please checkout [this page](https://github.com/soft-matter/trackpy/wiki/Guide-to-Installing-Python-and-Python-Packages).

With pip:

```
pip install https://github.com/caspervdw/clustertracking/archive/master.zip
```

With git:

```
git clone https://github.com/caspervdw
python setup.py install
```

Dependencies
------------
- numpy
- scipy
- pandas
- trackpy
- pims

Documentation
-------------
API documentation (work in progress) is available [over here](https://caspervdw.github.io/clustertracking/).


Support
-------
This package was developed in by Casper van der Wel, as part of his
PhD thesis work in Daniela Kraft's group at the Huygens-Kamerlingh-Onnes laboratory,
Institute of Physics, Leiden University, The Netherlands. This work was
supported by the Netherlands Organisation for Scientific Research (NWO/OCW).
