try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

# Get requirements from requirements.txt, stripping the version tags
with open('requirements.txt') as f:
    requires = [x.strip().split('=')[0]
                for x in f.readlines()]

setup(name='strax',
      version='0.1.0',
      description='Streaming analysis for XENON',
      author='Jelle Aalbers',
      author_email='j.aalbers@uva.nl',
      url='https://github.com/jelleaalbers/strax',
      setup_requires=['pytest-runner'],
      install_requires=requires,
      tests_require=requires + ['pytest',
                                'boltons',
                                'hypothesis'],
      packages=['strax',
                'strax.processing',
                'strax.storage',
                'strax.xenon'],
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',],
      zip_safe = False,
    )
