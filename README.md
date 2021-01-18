# FredMD
![Build](https://github.com/joe5saia/FredMD/workflows/FredMD%20Build/badge.svg)


This package downloads the [FRED-MD dataset](https://research.stlouisfed.org/econ/mccracken/fred-databases/) and estimates common factors. It also implements the [Bai-Ng (2002)](http://www.columbia.edu/~sn2294/pub/ecta02.pdf) factor selection information critrea. The alogrithms in this package are adapted from the matlab programs provided on the FRED-MD web page.

## Installation
This package can be installed via pip.

## Useage
```python
from FredMD import FredMD

fmd = FredMD(Nfactor=None, vintage=None, maxfactor=8, standard_method=2, ic_method=2)
fmd.estimate_factors()
f = fmd.factors
```

## References
* [Bai, Jushan and Ng, Serena (2002), "Determining the number of factors in approximate factor models"](https://onlinelibrary.wiley.com/doi/pdf/10.1111/1468-0262.00273).

* [McCracken, Michael W. and Ng, Serena (2015), "FRED-MD and FRED-QD: Monthly and Quarterly Databases for Macroeconomic Research"](https://research.stlouisfed.org/econ/mccracken/fred-databases/).


* [Bai, Jushan and Ng, Serena (2019), "Matrix Completion, Counterfactuals, and Factor Analysis of Missing Data"](https://arxiv.org/abs/1910.06677).
