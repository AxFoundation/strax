try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

requires = 'numpy pandas numba blosc zstd tqdm dill'.split()

setup(name='strax',
      version='0.0.1',
      description='Streaming analysis for XENON',
      author='Jelle Aalbers',
      author_email='j.aalbers@uva.nl',
      url='https://github.com/jelleaalbers/strax',
      setup_requires=['pytest-runner'],
      install_requires=requires,
      tests_require=requires + ['pytest', 'boltons', 'hypothesis'],
      packages=['strax'])
