How to add a new mixing method to Taweret
=========================================

Taweret is meant to be extensible and is willing to accept any mixing methods the communities develops.
These notes serve as instructions on how you can add your mixing methods to the Taweret repository.
All mixing methods in Taweret must inherit from the base class in ``Taweret/core``.
To add a new mixing method (or model), you need to:
- Step 1: Fork the repository, and clone it

.. code-block:: bash

   git clone <your fork>

- Step 2: Navigate to the ``Taweret/mix`` directory and open a file to contain your new mixing method. Sans any comments you wish to add, the first several lines should look like

.. code-block:: python

    from Taweret.core.base_mixer import BaseMixer

    class MyMixer(BaseMixer):
        def __init__(self, ...):
            ...

The ``BaseMixer`` is an abstract base class which has certain methods that need to be defined for its interpretation by the Python interpreter to succeed. Which methods, and their descriptions, can be found in the API documentation for the ``BaseMixer``

- Step 3: Add unit tests for mixing method to the the pytest directory. To make sure the python interpreter sees the add modules, the first several lines of your test file shoud read

.. code-block:: python

   import os
   import sys

   dirname = __file__
   taweret_wd = dirname.split('test')[0]
   sys.path.append(taweret_wd)

   from Taweret.mix.<your_module> import *
   import pytest

   # All functions starting with `test_` will be register by pytest

- Step 4: You need to document your code well, following the examples you see in existing mxing methods, this includes type annotations and RST style code comments. The documentation generations should automatically identify your code

- Step 5: Format your code using the ``autopep8`` code formatter. We recommend using the following command in the base directory of the repository

.. code-block:: bash

   autopep8 --recursive --in-place --aggresive --aggresive .

- Step 5: Create a pull request for your addition. This should trigger a github action. Should the action fail, please try to diagnose the failure. Always make sure the test execute successfully, locally before opening a pull request
