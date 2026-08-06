Installation
============

Prerequisites
-------------
.. _openbt Python package: https://pypi.org/project/openbt/
.. _OpenBT documentation: https://openbt.readthedocs.io
.. _bilby: https://pypi.org/project/bilby/

The Trees module uses the MPI-based `openbt Python package`_, which provides an
interface between the C++ OpenBT command line tools and Taweret.  Before
installing Taweret, please prepare your system as indicated in the `OpenBT
documentation`_.  Users experiencing Taweret installation issues may benefit
from explicitly installing and testing `openbt` prior to installing Taweret.

It is important to note that ``pip`` maintains a cache of previously built
wheels. Therefore a new Taweret installation may be faulty if `openbt` was
installed using a previously cached wheel that was built with an MPI
installation that is different from or incompatible with the current MPI
installation.   To determine if ``pip`` has an `openbt` wheel cached,
inspect the output of

.. code:: console

    python -m pip cache list

If an `openbt` wheel is present, consider removing it prior to installing
Taweret with a command such as

.. code:: console

    python -m pip cache remove openbt

Otherwise, the list of dependences is as short as possible to keep the installation process streamlined and allow for minimal, clean installations; however, if a user would like to run 
the Jupyter notebooks in the associated Jupyter Book, dependences for the notebooks will need to be installed in the relevant environment.
These dependences are located in the Jupyter notebooks, and hence can be quickly installed by running the import cell at the top of each notebook.

The `bilby`_ sampler comes with the ability to use a suite of samplers---in the case of Taweret, we also have not listed all
samplers as dependences. However, the user can (and should) install any samplers that they wish to use and ``bilby`` will be able
to use them through its wrapper in the Taweret package.

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

Clone-based Installation
------------------------
.. _repository: https://github.com/bandframework/Taweret.git

Alternatively, you can clone the `repository`_, checkout the desired commit (ideally the latest tagged release), and install Taweret into your
Python environment in developer or editable mode from the clone by running

.. code-block:: bash

   pip install -e .

Conda Installation
------------------
While our set of GitHub actions currently test Anaconda installations, the setup
of those tests within the action runner is less than desirable.  In particular,
the action no longer succeeds to build `openbt` if an MPI implementation is
installed using Conda.  Rather, the action installs an MPI implementation from
PyPI using ``pip``, which is less clean than a Conda installation.  Users who
prefer to use Conda should proceed with extra caution.

Testing
-------
A Taweret installation can be tested directly by executing

.. code-block:: python

    >>> import Taweret
    >>> Taweret.__version__
    >>> Taweret.test()

The version output should be consistent with the version of the release that was installed or the commit that was used to install from your local clone.
