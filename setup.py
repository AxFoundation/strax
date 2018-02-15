try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


setup(name='strax',
      version='0.0.1',
      description='Streaming analysis for XENON',
      author='Jelle Aalbers',
      author_email='j.aalbers@uva.nl',
      url='https://github.com/jelleaalbers/strax',
      py_modules=['multihist'],
      requires=open('requirements.txt').read().splitlines(),
      packages=['strax'])
