How to update `Taweret documentation <https://taweretdocs.readthedocs.io>`_
===========================================================================
We use the ``sphinx`` documentation system to encode and manage the
documentation for our Python package.  The official documentation is published
by read the docs, test builds are made available in PRs by read the docs, and
users can build and access locally the documents using our ``tox`` setup.  In
addition, HTML and PDF-format builds of the documentation are available for
download as artifacts from documentation build GitHub actions.

``sphinx`` uses ``reStructuredText`` as the plaintext markup language.  You
might find this 
`cheatsheet <https://docs.generic-mapping-tools.org/6.2/rst-cheatsheet.html>`_ 
useful.  You can also refer to 
`sphinx website <https://www.sphinx-doc.org/en/master/usage/quickstart.html>`_ 
for more information. 

Structure
---------
``docs/source/index.rst`` determines the overall structure of the
documentation. Each file that is referenced in ``index.rst`` can be found in
the same folder.  For example, if you want to modify the general introduction
documentation, change ``docs/source/intro.rst``.

Inline documentation of Python code in the package is automatically built into
the documentation based on the content of the ``sphinx``-based inline
docstrings associated with the code itself (i.e., the Python files are
self-documenting).  Therefore, there should be no need to alter any of the
``.rst`` files to include your inline documentation unless you are adding in
new subpackages or altering the structure of the package.  In such cases,
please contact one of the core Taweret developers for help.

Developers should not add or commit anything in the ``docs/build_pdf`` or
``docs/build_html`` folders, to which local documentation build results are
written.

Building
--------
While the generation and publishing of the sphinx documentation is automated
and includes sanity checks to confirm correct building, there is no automation
that confirms correctness of the content.  Our git workflow has been designed
to help maintain the documentation up-to-date and correct.

**TODO**:  Add in information regarding how documentation is managed within the
general git workflow to motivate and provide context for the following.

To help developers interactively improve the documentation and manually
confirm correct content and rendering, we suggest that developers use the
``tox`` documentation tasks ``html`` and ``pdf`` to locally visualize their
changes.  Please refer to :ref:`ToxDevGuide` for more information.  For those
developers who prefer to avoid ``tox``, please read ``tox.ini`` to determine
how to setup and use your environment manually for building documentation.
