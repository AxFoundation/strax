try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

def req_file(filename):
    with open(filename) as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    return [x.strip().split('=')[0] for x in content] 
    
setup(name='strax',
      version='0.0.1',
      description='Streaming analysis for XENON',
      author='Jelle Aalbers',
      author_email='j.aalbers@uva.nl',
      url='https://github.com/jelleaalbers/strax',
      setup_requires=['pytest-runner'],
      install_requires=req_file('requirements.txt'),
      tests_require=req_file('requirements.txt') + ['pytest', 'boltons', 'hypothesis'],
      packages=['strax'])
