Taweret.mix.gp_bmm
==================
..
    The GPRwrapper class is inherited from a scikit-learn class, which required
    the use of intersphinx in this documentation system to get the documenation
    rendering without errors/warnings.  As a result, clicking on that base class
    does take the user to the scikit-learn docs for that class.  The decision
    was made to explicitly leave out of the docs for our class the set_* member
    functions inherited from scikit-learn.  Our goal is for that class to only
    contain docs on its fit() member function.

.. automodule:: Taweret.mix.gp_bmm
   :members:
   :exclude-members: set_fit_request, set_predict_request, set_score_request
   :undoc-members:
   :show-inheritance:
