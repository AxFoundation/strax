try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

requires = open('requirements.txt').read().splitlines()

setup(name='strax',
      version='0.0.1',
      description='Streaming analysis for XENON',
      author='Jelle Aalbers',
      author_email='j.aalbers@uva.nl',
      url='https://github.com/jelleaalbers/strax',
      setup_requires=requires + ['pytest-runner'],
      tests_require=requires + ['pytest'],
      packages=['strax'])
