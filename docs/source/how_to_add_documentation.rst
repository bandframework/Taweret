How to update documentation in Taweret `website <https://danosu.github.io/Taweret/index.html>`_
===============================================================================================

We use ``sphinx`` python documentation generator to generate the webpage 
for our python package. It uses ``reStructuredText`` as the plaintext markup 
language. 

You might find this 
`cheatsheet <https://docs.generic-mapping-tools.org/6.2/rst-cheatsheet.html>`_ 
useful. 

You can also refer to 
`sphinx website <https://www.sphinx-doc.org/en/master/usage/quickstart.html>`_ 
for more information. 

**Steps**

- Step 1: Fork Taweret repo
- Step 2: Get the fork and chekout the ``develop`` branch

.. code:: bash

        git clone <url_to_the_fork> --branch develop
        cd Taweret

- Step 3: Create a virtual environment for Taweret

With conda:

.. code:: bash

        conda env create --file=environment.yml
        conda activate test_env

With virtualenv: 

.. code:: bash

        virtualenv test_env
        source test_env/bin/activate


- Step 4: Locally generate documentation webpage

.. code:: bash

        cd docs
        sh run_to_rebuild_tawret_rst.sh

This will create a webpage locally and open it in your default web browser. 
You can modify the files inside ``Taweret/docs/source`` to make changes to 
the webpage.

``Taweret/docs/source/index.rst`` determines the overall structure of the 
webpage. Each file that is referenced in the ``index.rst`` can be found in
the same folder.

For example, if you want to modify the introduction, 
change ``Taweret/docs/source/intro.rst``.

After you make changes and locally build the web page, 
you can push these changes to the original webpage by following 
the below set of instructions. 

.. code:: bash

        git add <file_you changed_inside_source_directory>
        git commit -m <you commit messege>
        git push origin develop

Then make a pull request from your forked repository to 
the ``danOSU/Taweret`` repository, ``develop`` branch. 
**Note** : You do not have to add or commit anything in 
the ``Taweret/docs/build`` folder. 
