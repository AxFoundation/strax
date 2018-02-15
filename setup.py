from setuptools import setup

setup(name='strax',
      version='0.0.1',
      description='Streaming analysis for XENON',
      author='Jelle Aalbers',
      author_email='j.aalbers@uva.nl',
      url='https://github.com/jelleaalbers/strax',
      py_modules=['multihist'],
      install_requires='numpy numba zstd'.split(),
      packages=['strax'])
