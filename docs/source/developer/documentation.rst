Writing documentation
======================

To write documentation, please refer to the existing for examples. To add new pages:
 - Add a new ``.rst`` file in the basics/advanced/developer folder within ./docs.
 - Add the link to the file in the docs/index.rst
 - run ``bash make_docs.sh``. This will run sphinx locally, this allows one to
   preview if the results are the desired results. Several modules need be
   installed in order to run this script.
 - Add the ``.rst`` file, the ``index.rst`` to git.

Updating ``docs/reference``
---------------------------
The ``docs/reference`` is automatically updated with ``bash make_docs.sh``.
In case modules are added/removed, one needs to rerun this script to and commit
the changes to the files in ``docs/reference``.