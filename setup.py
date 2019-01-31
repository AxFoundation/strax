import setuptools

# Get requirements from requirements.txt, stripping the version tags
def get_file_lines_as_list(fname):
    with open(fname) as f:
        return f.read().splitlines()

def strip_version_specifiers(fname):
    return [x.split('=')[0] for x in get_file_lines_as_list(fname)]

# Get requirements for package, test, docs
requires      = strip_version_specifiers('requirements.txt'     )
requires_test = strip_version_specifiers('requirements_test.txt')
requires_docs = strip_version_specifiers('requirements_docs.txt')

with open('README.md') as file:
    readme = file.read()

with open('HISTORY.md') as file:
    history = file.read()

setuptools.setup(name='strax',
                 version='0.6.1',
                 description='Streaming analysis for xenon TPCs',
                 author='Jelle Aalbers',
                 url='https://github.com/AxFoundation/strax',
                 setup_requires=['pytest-runner'],
                 install_requires=requires,
                 tests_require=requires + requires_test,
                 long_description=readme + '\n\n' + history,
                 python_requires=">=3.6",
                 extras_require={'docs': requires_docs},
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
