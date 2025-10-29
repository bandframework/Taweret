.. _ToxDevGuide:

`tox`_ Development Environments
===============================
.. _tox: https://tox.wiki/en/latest/index.html
.. _editable: https://setuptools.pypa.io/en/latest/userguide/development_mode.html

The repository has been configured so that developers and GitHub actions can
execute certain routine tasks using ``tox``, which builds and manages an
independent, minimal Python virtual environment for each task.  This can help
isolate the relatively exploratory and dirty work of development, for example,
from explicitly-managed, minimal, clean virtual environments used to perform
scientific studies with Taweret.

.. note::
    Developers that would like to use ``tox`` should learn, for example, the
    difference between running ``tox -e <task>`` and ``tox -r -e <task>`` so
    that they can understand what ``-r`` does and when it should be used.

If a developer would like to use ``tox`` on their individual platform, they
must first identify the Python that they would like ``tox`` to use for building
virtual environment and use this to install ``tox`` as a command line tool.  To
install ``tox``, execute some version of

.. code-block:: console

    $ cd
    $ deactivate (if already in a virtual environment)
    $ /path/to/target/python --version
    $ /path/to/target/python -m venv ~/.toxbase
    $ ~/.toxbase/bin/pip list
    $ ~/.toxbase/bin/python -m pip install --upgrade pip
    $ ~/.toxbase/bin/python -m pip install --upgrade setuptools
    $ ~/.toxbase/bin/python -m pip install tox
    $ ~/.toxbase/bin/tox --version
    $ ~/.toxbase/bin/pip list

Rather than require that developers activate ``.toxbase``, we follow the
suggestions in the `webinar <https://www.youtube.com/watch?v=PrAyvH-tm8E>`_
and add ``tox`` to the ``PATH`` by executing an appropriate variation of

.. code-block:: console

    $ mkdir ~/local/bin
    $ ln -s ~/.toxbase/bin/tox ~/local/bin/tox
    $ vi ~/.bash_profile (add ~/local/bin to PATH)
    $ . ~/.bash_profile
    $ which tox
    $ tox --version

If the environment variable ``COVERAGE_FILE`` is set, then this is the name of
the coverage results file that will be used with all coverage tasks;
``COVERAGE_XML``, the name of the XML-format coverage report to generate;
``COVERAGE_HTML``, the name of the HTML-format coverage report to generate.

``tox`` will not carry out any work by default with the calls ``tox`` or ``tox
-r``.  The following commands can be run from the root of a Taweret clone

* ``tox -r -e coverage``

  * Run the full Taweret test suite and save the coverage results to the
    coverage file.
  * Note that this task installs the Taweret code in its virtual environment as
    an editable_ installation of the local clone, which mimics an interactive
    developer setup.

* ``tox -r -e nocoverage``

  * Run the full Taweret test suite without coverage.
  * Note that this task installs the Taweret code in its virtual environment by
    building and installing a wheel, which mimics a user setup.

* ``tox -r -e oldest``

  * Run the full Taweret test suite using the oldest allowable Python version
    and the oldest versions of external dependencies that have a limited set of
    acceptable versions specified in `pyproject.toml`.
  * Note that this task installs the Taweret code in its virtual environment as
    an editable_ installation of the local clone, which mimics an interactive
    developer setup.

* ``tox -r -e report``

  * Generate XML- and HTML-format coverage reports based on the results of the
    last execution of the ``coverage`` task.  Typically this will be run just
    after or with ``coverage``

* ``tox -r -e check``

  * Run all Python code in the package through ``flake8`` to identify issues
    and nonadherence to PEP8 standards.
  * Note that this does **not** alter any files, but rather only reports issues.

* ``tox -r -e html``

  * Render the Taweret documentation in HTML.

* ``tox -r -e pdf``

  * Render the Taweret documentation as a PDF document.

* ``tox -r -e book``

  * Render the Taweret Jupyter book from scratch and with conservative error
    checking.  When building the book for publishing, for review, or checking
    for issues, this is the task to use.

* ``tox -r -e bookdev``

  * Rendering the book from scratch is slow.  Developers can use this task when
    actively adding/updating content to save time.  It is intended that they
    run ``book`` upon finishing their work to ensure that all is well.

Note that you can combine different tasks into a single call such as ``tox -e
report,coverage``.

The virtual environments created by ``tox`` can be activated for general use by
developers.  In particular, the ``coverage`` environment can be useful for
interactive development and testing since Taweret is installed in
editable_/developer mode.  Assuming that the ``coverage`` task has already been
run, activate this environment by calling

.. code-block:: console

    . /path/to/Taweret/.tox/coverage/bin/activate
