Installation
============

Taweret is available via pip install

.. code-block:: bash

    pip install Taweret

If you prefer to use conda for your package management, you can still pip install Taweret, but be sure to `conda install pip` first, so you conda environment knows where to look.

Alternative Installation
------------------------

Alternatively, you can clone the `repository <https://github.com/bandframework/Taweret.git>`.
Open cloning, the dependencies for Taweret dependencies by running the command

.. code-block:: bash

   pip install -e .


From here, you can run the notebooks, for example, in CodeSpaces.

Prerequisites
-------------

The Trees module depends on OpenMPI. Please ensure OpenMPI is installed with shared/built libraries prior to using the Trees module.


Windows Users
--------------

The Trees module is a Python interface for a C++ backend. For Mac OS/X and Linux users, the compiled libraries  \
are installed as a dependency with Taweret. This module is developed as a part of the \
Open Bayesian Trees Project (OpenBT). See references [1] and [2] for details. The package relies on OpenMPI \
thus Windows users must use Windows subsytem for linux in order to use the Trees module. Further installation \
instructions are listed below. 

OpenBT will run within the Windows 10 Windows Subsystem for Linux (WSL) environment. For instructions on installing WSL, \
please see (https://ubuntu.com/wsl). We recommend installing the Ubuntu 20.04 WSL build. \
There are also instructions \
(https://wiki.ubuntu.com/WSL?action=subscribe&_ga=2.237944261.411635877.1601405226-783048612.1601405226#Installing_Packages_on_Ubuntu) \
on keeping your Ubuntu WSL up to date, or installing additional features like X support. Once you have \
installed the WSL Ubuntu layer, start the WSL Ubuntu shell from the start menu and then you can begin working with Taweret.

 
**OpenBT References**

1. OpenBT Repository (https://bitbucket.org/mpratola/openbt/src/master/).

2. OpenBR Repository with Model Mixing (https://github.com/jcyannotty/OpenBT).
