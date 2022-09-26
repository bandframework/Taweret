Installation
============

Taweret requires the following for for basic functionality.
    - numpy
    - seaborn

.. code-block:: bash

    pip install Taweret

Installation from sources
________
The simplest way to install Taweret from sources is to run

.. code-block:: bash

    pip install git+https://github.com/danOSU/Taweret.git

you can also do it step by step: clone the repo, install dependencies. Use Taweret by appending it to the system path.

.. code-block:: bash

    git clone https://github.com/danOSU/Taweret.git
    cd Taweret
    pip install -r requirements/requirements.txt
    export PATH=$PATH:[path to your local Taweret repo]