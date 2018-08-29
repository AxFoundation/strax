#!/usr/bin/env bash
make clean
rm -r source/reference
sphinx-apidoc -o source/reference ../strax
cp ../notebooks/datastructure.rst source/reference
rm source/reference/modules.rst
make html