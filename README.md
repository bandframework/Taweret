# Taweret

<img align="right" width="200" src="./logos/taweret_logo.PNG">

Welcome to the GitHub repo for Taweret, the state of the art Python package for applying Bayesian Model Mixing! 

## About
Taweret is a new generalized package to help with applying Bayesian model mixing methods, developed by members of the BAND collaboration, to a wide variety of problems in physics. 

## Features
At present, this package possesses the following BMM methods:
- Linear model mixing ( With simultaneous model mixing and calibration)
- Multivariate BMM 
- Bayesian Trees

## Documentation
See Taweret's docs webpage [here](https://taweretorg.github.io/Taweret/).

## Testing
The test suite requires the [pytest](https://pypi.org/project/pytest/) package to be installed and can be run from the `test/` directory. To test the current BMM methods, run the following three lines of code:

```
pytest test_bivariate_linear.py
pytest test_gaussian.py
pytest test_trees.py
```

## Citing BAND software
If you have benefited from Taweret, please cite the BAND collaboration software suite using the format [here](https://github.com/bandframework/bandframework#citing-the-band-framework).

## Contact
To contact the Taweret team, please submit an issue through the Issues page. 

Authors: Kevin Ingles, Dan Liyanage, Alexandra Semposki, and John Yannotty.
