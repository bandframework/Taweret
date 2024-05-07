

# Taweret
[![codecov](https://codecov.io/gh/bandframework/Taweret/graph/badge.svg?token=BQ7ZAD5ONP)](https://codecov.io/gh/bandframework/Taweret)

<img align="right" width="200" src="logos/taweret_logo.PNG">

Welcome to the GitHub repo for Taweret, the state of the art Python package for applying Bayesian Model Mixing! 

## About
Taweret is a new generalized package to help with applying Bayesian model mixing methods, developed by members of the [BAND](https://bandframework.github.io/) collaboration, to a wide variety of problems in physics. 

## Features
At present, this package possesses the following BMM methods:
- Linear model mixing ( With simultaneous model mixing and calibration)
- Multivariate BMM 
- Bayesian Trees

## Documentation
See Taweret's docs webpage [here](https://bandframework.github.io/Taweret/).

### Cloning
This repository uses submodules. 
To clone this repository and automatically checkout all the submodules, use
```terminal
git clone --recursive https://github.com/bandframework/Taweret.git 
```

If you want to limit the size of the repository (this or the submodules), you can use the `depth` flag
```terminal
git clone --depth=1 https://github.com/bandframework/Taweret.git
```
Inside the directory containing the cloned repository, you then run
```terminal
git submodule update --init --depth=1
```

### Prerequisites

The Trees module depends on [OpenMPI](https://www.open-mpi.org/). Please ensure OpenMPI is installed with shared/built libraries prior to using the Trees module.

## Testing
The test suite requires the [pytest](https://pypi.org/project/pytest/) package to be installed and can be run from the `test/` directory. To test the current BMM methods, first install the required packages and then run the following three lines of code:

To installing requirements, first navigate to the Taweret directory. The requirements.txt file is located in the root of this directory. Once in the Taweret directory, then execute the following line of code from the terminal.

```
pip install -e .
```

Once all installation is complete, proceed with testing by naviagating to the `test/` directory and executing the following three lines of code.

```
pytest test_bivariate_linear.py
pytest test_gaussian.py
pytest test_trees.py
```

## Windows Users:
 
Taweret also depends on the OpenBT Mixing package in order to execute the trees modulde. This package is built with OpenMPI thus Windows users can work with the trees module using Windows Subsystem for Linux. Installation instructions are shown below.

OpenBT will run within the Windows 10 Windows Subsystem for Linux (WSL) environment. For instructions on installing WSL,
please see (https://ubuntu.com/wsl). We recommend installing the Ubuntu 20.04 WSL build. There are also instructions
[here](https://wiki.ubuntu.com/WSL?action=subscribe&_ga=2.237944261.411635877.1601405226-783048612.1601405226#Installing_Packages_on_Ubuntu) 
on keeping your Ubuntu WSL up to date, or installing additional features like X support. Once you have installed the WSL Ubuntu layer, start the WSL Ubuntu shell from the start menu and then you can begin working with Taweret.

## Running on Codespaces
GitHub's Codespaces is a great place to test using Taweret. Right now, you can try out Taweret's Bivariate Linear BMM and Multivariate BMM methods there, by following the instructions below. 

1. Click the dropdown arrow on the green 'code' button found at the top of this page.
2. Click on the tab there that says 'codespaces'.
3. Click the button for 'create Codespace on main'.
4. Wait for the terminal to be finish spinning up a virtual environment and loading all needed variables (this can take a few minutes).
5. Once that is done, navigate on the file tree to a notebook you wish to run. To run a file, you need to set a kernel for the Jupyter notebook, so click on 'choose a kernel' in the upper right hand corner of the notebook. If you haven't gotten this message already, a message will pop up that says 'install preferred Python extension?', and you should click 'yes'.
6. When you click 'choose a kernel' it will offer a preferred Python version or a base version (usually a newer Python version). Choose whichever you prefer, and then you can run the notebook!

## Citing Taweret
If you have benefited from Taweret, please cite our software using the following format:

```
@inproceedings{Taweret,
    author = "Liyanage, Dan and Semposki, Alexandra and Yannotty, John and Ingles, Kevin",
    title  = "{{Taweret: A Python Package for Bayesian Model Mixing}}",
    year   = "2023",
    url    = {https://github.com/bandframework/Taweret}
}
```

and our explanatory paper:

```
@article{Ingles:2023nha,
    author = "Ingles, Kevin and Liyanage, Dananjaya and Semposki, Alexandra C. and Yannotty, John C.",
    title = "{Taweret: a Python package for Bayesian model mixing}",
    eprint = "2310.20549",
    archivePrefix = "arXiv",
    primaryClass = "nucl-th",
    month = "10",
    year = "2023"
}
```

Please also cite the BAND collaboration software suite using the format [here](https://github.com/bandframework/bandframework#citing-the-band-framework).

## BAND SDK compliance
Check out our SDK form [here](https://github.com/bandframework/Taweret/blob/main/Taweretbandsdk.md).

## Contact
To contact the Taweret team, please submit an issue through the Issues page. 

Authors: Kevin Ingles, Dan Liyanage, Alexandra Semposki, and John Yannotty.


