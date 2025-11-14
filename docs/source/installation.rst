Installation
============

Prerequisites
-------------
.. _OpenBTMixing: https://pypi.org/project/openbtmixing/
.. _documentation: https://github.com/jcyannotty/OpenBT?tab=readme-ov-file#installation
.. _ninja: https://ninja-build.org
.. _bilby: https://pypi.org/project/bilby/

The Trees module uses the MPI-based `OpenBTMixing`_ Python package (See [1] and
[2]).  Before installing Taweret, please prepare your system as indicated in the
OpenBTMixing installation `documentation`_.  Note that for some package
managers, developer library packages such as ``libopenmpi-dev`` or
``libmpich-dev`` might need to be installed in addition to the base MPI packages
such as ``openmpi-bin`` or ``mpich``. OpenBTMixing also requires the `ninja`_
build system to be preinstalled, so this may need to be added to the user's
system prior to installing Taweret and therefore OpenBTMixing.

It is important to note that ``pip`` maintains a cache of previously built
wheels. Therefore a new Taweret installation may be faulty if OpenBTMixing was
installed using a previously cached wheel that was built with an MPI
installation that is different from or incompatible with the current MPI
installation.   To determine if ``pip`` has an OpenBTMixing wheel cached,
inspect the output of

.. code:: console

    python -m pip cache list

If an OpenBTMixing wheel is present, consider removing it prior to installing
Taweret with a command such as

.. code:: console

    python -m pip cache remove openbtmixing

Otherwise, the list of dependences is as short as possible to keep the installation process streamlined and allow for minimal, clean installations; however, if a user would like to run 
the Jupyter notebooks in the associated Jupyter Book, dependences for the notebooks will need to be installed in the relevant environment.
These dependences are located in the Jupyter notebooks, and hence can be quickly installed by running the import cell at the top of each notebook.

If you try building OpenBTMixing and it fails due to not finding the ``ninja`` package, install ``ninja`` via

.. code:: console

    pip install ninja

The `bilby`_ sampler comes with the ability to use a suite of samplers---in the case of Taweret, we also have not listed all
samplers as dependences. However, the user can (and should) install any samplers that they wish to use and ``bilby`` will be able
to use them through its wrapper in the Taweret package.

**OpenBT References**

1. OpenBT Repository (https://bitbucket.org/mpratola/openbt/src/master/).
2. OpenBT Repository with Model Mixing (https://github.com/jcyannotty/OpenBT).

Windows Users
^^^^^^^^^^^^^
.. _instructions: https://wiki.ubuntu.com/WSL?action=subscribe&_ga=2.237944261.411635877.1601405226-783048612.1601405226#Installing_Packages_on_Ubuntu

While in the past Taweret was known to work with Windows 10 using an
appropriately configured Ubuntu 20.04 Windows Subsystem for Linux (WSL) build,
it is presently tested only using macOS and Ubuntu installations.  Windows users
that would like to see if Taweret will work for them might find these
`instructions`_ useful for helping to keep their Ubuntu WSL up to date, or to
install additional features like X support.

Standard Installation
---------------------
Taweret is available via pip install

.. code-block:: bash

    pip install Taweret

If you prefer to use conda to setup your Python environment, you can still pip install Taweret, but be sure to `conda install pip` first, so you conda environment knows where to look.

Alternative Installation
------------------------
.. _repository: https://github.com/bandframework/Taweret.git

Alternatively, you can clone the `repository`_, checkout the desired commit (ideally the latest tagged release), and install Taweret into your
Python environment in developer or editable mode from the clone by running

.. code-block:: bash

   pip install -e .

Testing
-------
A Taweret installation can be tested directly by executing

.. code-block:: python

    >>> import Taweret
    >>> Taweret.__version__
    >>> Taweret.test()

The version output should be consistent with the version of the release that was installed or the commit that was used to install from your local clone.
