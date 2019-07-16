import setuptools

# Get requirements from requirements.txt, stripping the version tags
with open('requirements.txt') as f:
    requires = [x.strip().split('=')[0]
                for x in f.readlines()]

with open('README.md') as file:
    readme = file.read()

with open('HISTORY.md') as file:
    history = file.read()

setuptools.setup(name='strax',
                 version='0.7.5',
                 description='Streaming analysis for xenon TPCs',
                 author='Jelle Aalbers',
                 url='https://github.com/AxFoundation/strax',
                 setup_requires=['pytest-runner'],
                 install_requires=requires,
                 tests_require=requires + ['pytest',
                                           'boltons',
                                           'hypothesis'],
                 long_description=readme + '\n\n' + history,
                 python_requires=">=3.6",
                 extras_require={
                     'docs': ['sphinx',
                              'sphinx_rtd_theme',
                              'nbsphinx',
                              'recommonmark']
                 },
                 long_description_content_type="text/markdown",
                 packages=setuptools.find_packages(),
                 classifiers=[
                     'Development Status :: 4 - Beta',
                     'License :: OSI Approved :: BSD License',
                     'Natural Language :: English',
                     'Programming Language :: Python :: 3.6',
                     'Intended Audience :: Science/Research',
                     'Programming Language :: Python :: Implementation :: CPython',
                     'Topic :: Scientific/Engineering :: Physics',
                 ],
                 zip_safe = False)
