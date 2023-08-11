Installation
============

Taweret requires the following for for basic functionality.
    - python
    - numpy
    - seaborn
    - jupyter
    - bilby
    - ptemcee

Follow these steps to install Taweret from github. 

.. code-block:: bash

    git clone https://github.com/danOSU/Taweret.git
    #If you want to use SMABA toy models please clone samba repo
    #git clone https://github.com/asemposki/SAMBA.git
    cd Taweret
    conda env create -f environment.yml
    conda activate test_env
    cd doc/source/notebooks
    jupyter notebook --browser=safari

You can look at the available notebooks in the directory and modify it for your own use case. If \
you need to add a new mixing method please refer to the **For Developers** section. 

The pip installation is not available yet. 

.. code-block:: bash

    pip install Taweret



Additional Requirements
-----------------------

Certain Taweret modules may require additional steps to properly setup an environment which can \
execute the code. These modules and their respective requirements are listed below.

**Bayesian Trees**
^^^^^^^^^^^^^^^^^^

The Trees module is a Python interface which calls and executes a Ubuntu package in order \
to fit the mixing model and obtain the resulting predictions. This package is developed as a part of the \
Open Bayesian Trees Project (OpenBT). See references [1] and [2] for details. To install the Ubuntu package, \
please follow the steps below based on the operating system of choice.


**Linux:**

1. Download the OpenBT Ubuntu Linux 20.04 package:

.. code-block:: bash
    
    $ wget -q https://github.com/jcyannotty/OpenBT/raw/main/openbt_mixing0.current_amd64-MPI_Ubuntu_20.04.deb 
    

2. Install the package and reset the library cache:

.. code-block:: bash
    
    $ cd /location/of/downloaded/.deb
    $ dpkg -i openbt_mixing0.current_amd64-MPI_Ubuntu_20.04.deb
    $ ldconfig


**Mac OS/:X**

1. Install the OS/X OpenMPI package by running the following `brew` commands in a terminal window:

.. code-block:: bash
    
    $ /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"
    $ brew install open-mpi

2. Download the OpenBT OSX binary package:

3. Install the OpenBT OSX package by double-clicking on the downloaded .pkg file and follow the on-screen instructions.


**Windows:**

OpenBT will run within the Windows 10 Windows Subsystem for Linux (WSL) environment. For instructions on installing WSL, \
please see (https://ubuntu.com/wsl). We recommend installing the Ubuntu 20.04 WSL build. \
There are also instructions \
(https://wiki.ubuntu.com/WSL?action=subscribe&_ga=2.237944261.411635877.1601405226-783048612.1601405226#Installing_Packages_on_Ubuntu) \
on keeping your Ubuntu WSL up to date, or installing additional features like X support. Once you have \
installed the WSL Ubuntu layer, start the WSL Ubuntu shell from the start menu and then install the package:

.. code-block:: bash
    
    $ cd /mnt/c/location/of/downloaded/.deb
    $ dpkg -i openbt_mixing0.current_amd64-MPI_Ubuntu_20.04.deb


**OpenBT References**

1. OpenBT Repository (https://bitbucket.org/mpratola/openbt/src/master/).

2. OpenBT Repository with Model Mixing (https://github.com/jcyannotty/OpenBT).   