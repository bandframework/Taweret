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
you need to add a new mixing method please refer to the **For Deveopers** section. 

The pip instalation is not available yet. We are working on it. 

.. code-block:: bash
    pip install Taweret
