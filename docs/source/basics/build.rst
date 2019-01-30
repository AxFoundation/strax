
Build strax
================

XYZ = 372

> conda create -n strax_pyXYZ python=X.Y.Z

> source activate strax_pyXYZ
> conda install -n strax_pyXYZ numpy pandas numba blosc zstd tqdm dill psutil numexpr boto3

> python setup.py build
> python setup.py install


Test strax
================

> source activate strax_pyXYZ
> conda install -n strax_pyXYZ pytest
> pip install boltons
> pip install hypothesis

> python setup.py test


Document strax
================

> source activate strax_pyXYZ
> conda install -n strax_pyXYZ sphinx sphinx_rtd_theme
> pip install nbsphinx recommonmark

> cd strax/docs
> ./make_docs.sh

