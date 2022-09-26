Installation
============

Taweret requires the following for for basic functionality.
    - numpy
    - seaborn

Follow these steps to install Taweret from github. 

.. code-block:: bash

    git clone https://github.com/danOSU/Taweret.git
    #If you want to use SMABA toy models please close samba repo too
    #git clone https://github.com/asemposki/SAMBA.git
    cd Taweret
    conda env create -f environment.yml
    conda activate test_env
    cd Taweret/doc/build/html/notebooks

You can look at the available notebooks in the directory and modify it for your own use case. If \
you need to add a new mixing method please refer to the **For Deveopers* section. 

The pip instalation is not available yet. We are working on it. 

.. code-block:: bash
    pip install Taweret