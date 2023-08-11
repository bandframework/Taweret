How to add a new mixing method to Taweret
=========================================

- Step 1 : Use the ``base_mixer`` abstract class in ``Taweret/core`` and implement all class methods for your mixing approach. \
The base mixer contains the minimum requirements for a Taweret mixing module and thus serves as a starting point for each new mixing \
method. It is also possible to create additional class methods which do not appear in the base class.

- Step 2 : Add the new mixing module to ``Taweret/mix``\. This is the location where all of the mixing modules in Taweret will be stored.

**Take a look at the other mixing methods that are in** ``Taweret/mix`` **for examples**.





