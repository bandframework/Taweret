Installation
============

Taweret requires the following for for basic functionality.
    - python (>3.7)
    - matplotlib
    - numpy (>=1.20.3)
    - scipy (>=1.7.0)
    - seaborn
    - emcee
    - corner
    - scikit-learn
    - cycler
    - statistics
    - bilby
    - jupyter
    - ptemcee
    - pytest

Follow these steps to install Taweret from github. 

.. code-block:: bash

    git clone https://github.com/bandframework/Taweret.git
    #If you want to use SMABA toy models please clone samba repo
    #git clone https://github.com/asemposki/SAMBA.git
    cd Taweret
    conda env create -f environment.yml
    conda activate test_env
    cd docs/source/notebooks
    jupyter notebook --browser=safari

You can look at the available notebooks in the directory and modify it for your own use case. If \
you need to add a new mixing method please refer to the **For Developers** section. 

The pip installation is not available yet. 

.. .. code-block:: bash
.. 
..     pip install Taweret



Additional Requirements
-----------------------

Certain Taweret modules may require additional steps to properly setup an environment which can \
execute the code. These modules and their respective requirements are listed below.

**Bayesian Trees**
^^^^^^^^^^^^^^^^^^

The Trees module is a Python interface for a C++ backend. For Mac OS/X and Linux users, the compiled libraries  \
are installed as a dependency with Taweret. This package is developed as a part of the \
Open Bayesian Trees Project (OpenBT). See references [1] and [2] for details. The package relies on OpenMPI \
thus Windows users must use Windows subsytem for linux in order to use the Trees module. Further installation \
instructions are listed below. 


**Windows:**

OpenBT will run within the Windows 10 Windows Subsystem for Linux (WSL) environment. For instructions on installing WSL, \
please see (https://ubuntu.com/wsl). We recommend installing the Ubuntu 20.04 WSL build. \
There are also instructions \
(https://wiki.ubuntu.com/WSL?action=subscribe&_ga=2.237944261.411635877.1601405226-783048612.1601405226#Installing_Packages_on_Ubuntu) \
on keeping your Ubuntu WSL up to date, or installing additional features like X support. Once you have \
installed the WSL Ubuntu layer, start the WSL Ubuntu shell from the start menu and then you can begin working with Taweret.

 
**OpenBT References**

1. OpenBT Repository (https://bitbucket.org/mpratola/openbt/src/master/).

2. OpenBT Repository with Model Mixing (https://github.com/jcyannotty/OpenBT).   
