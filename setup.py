import setuptools


# Get requirements from requirements.txt, stripping the version tags
def open_requirements(path):
    with open(path) as f:
        requires = [
            r.split('/')[-1] if r.startswith('git+') else r
            for r in f.read().splitlines()]
    return requires


requires = open_requirements('requirements.txt')
docs_requires = open_requirements('extra_requirements/requirements-docs.txt')
tests_requires = open_requirements('extra_requirements/requirements-tests.txt')

with open('README.md') as file:
    readme = file.read()

with open('HISTORY.md') as file:
    history = file.read()

setuptools.setup(name='strax',
                 version='1.0.0-rc0',
                 description='Streaming analysis for xenon TPCs',
                 author='Jelle Aalbers',
                 url='https://github.com/AxFoundation/strax',
                 setup_requires=['pytest-runner'],
                 install_requires=requires,
                 tests_require=requires + tests_requires,
                 long_description=readme + '\n\n' + history,
                 # We don't test 3.6 anymore, see #503
                 python_requires=">=3.6",
                 extras_require={
                     'docs': docs_requires
                 },
                 long_description_content_type="text/markdown",
                 packages=setuptools.find_packages() + ['extra_requirements'],
                 package_dir={'extra_requirements': 'extra_requirements'},
                 package_data={'extra_requirements': ['requirements-docs.txt',
                                                      'requirements-tests.txt']},
                 classifiers=[
                     'Development Status :: 5 - Production/Stable',
                     'License :: OSI Approved :: BSD License',
                     'Natural Language :: English',
                     'Programming Language :: Python :: 3.6',
                     'Programming Language :: Python :: 3.7',
                     'Programming Language :: Python :: 3.8',
                     'Programming Language :: Python :: 3.9',
                     'Intended Audience :: Science/Research',
                     'Programming Language :: Python :: Implementation :: CPython',
                     'Topic :: Scientific/Engineering :: Physics',
                 ],
                 zip_safe=False)
