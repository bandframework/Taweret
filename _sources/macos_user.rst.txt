MacOS Users
===========

To run the `bilby` examples using the `ptemcee` module, you will need \ 
to run the following commands at the beginning of your notebook

.. code-block:: python
    
    import multiprocessing
    multiprocessing.set_start_method('fork')
There is currently an `issue <https://git.ligo.org/lscsoft/bilby/-/issues/722>`_ open on the `bilby` \
repository that will address this automatically in the future. \
For now, you will receive a warning whenever you use `ptemcee` with more than one `thread`, even if you have set the start method.
We will remove this warning once `bilby` patches their bug.
