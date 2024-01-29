# Run this script from the origin folder as:
#   > "python setup.py clean" in order to clean previous builds
#   > "python setup.py test" in order to execute all the unittests
#   > "python setup.py sdist" in order to build the library
#
# The package can then be published with:
#   > twine upload dist/*

from setuptools import find_packages, setup

# set up the library metadata and make the build
with open('README.md', 'r') as readme:
    setup(
        name='powerplantsim',
        version='0.1.0',
        maintainer='Luca Giuliani',
        maintainer_email='luca.giuliani13@unibo.it',
        author='University of Bologna - DISI',
        description='PowerPlantSim: a power plant simulator',
        long_description=readme.read(),
        long_description_content_type='text/markdown',
        packages=find_packages(include=['powerplantsim*']),
        python_requires='~=3.10',
        install_requires=[
            'matplotlib~=3.8.2',
            'networkx~=3.2.1',
            'numpy~=1.26.2',
            'pandas~=2.1.3',
            'pyomo~=6.7.0',
            'scikit-learn~=1.3.2',
            'tqdm~=4.66.1'
        ],
        test_suite='test'
    )
