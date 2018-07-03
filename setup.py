try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

# Get requirements from requirements.txt, stripping the version tags
with open('requirements.txt') as f:
    requires = [x.strip().split('=')[0]
                for x in f.readlines()]

with open('README.md') as file:
    readme = file.read()

with open('HISTORY.md') as file:
    history = file.read()

setup(name='strax',
      version='0.2.0',
      description='Streaming analysis for XENON',
      author='Jelle Aalbers',
      author_email='j.aalbers@uva.nl',
      url='https://github.com/AxFoundation/strax',
      setup_requires=['pytest-runner'],
      install_requires=requires,
      tests_require=requires + ['pytest',
                                'boltons',
                                'hypothesis'],
      long_description=readme + '\n\n' + history,
      extras_require={
          'docs': ['sphinx',
                   'sphinx_rtd_theme',
                   'nbsphinx',
                   'recommonmark'],
          'xenon': ['keras'
                    'tensorflow',
                    'scipy']
      },
      long_description_content_type="text/markdown",
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
