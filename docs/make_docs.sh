#!/usr/bin/env bash
make clean
rm -r source/reference
sphinx-apidoc -o source/reference ../strax
make html