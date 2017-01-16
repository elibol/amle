from setuptools import setup

import amle

with open('requirements.txt', 'r') as fh:
    requirements = filter(lambda x: x, fh.read().split("\n"))
    # print requirements

setup(
    name='amle',
    version=amle.__version__,
    author='Nicolo Fusi and Melih Elibol',
    author_email='fusi@microsoft.com, v-huelib@microsoft.com',
    description=('Code to recreate experimental results reported in '
                 '"Probabilistic Matrix Factorization for Automated Machine Learning."'),
    packages=['amle'],
    install_requires=requirements,
    )

print 'please run: pip install -r requirements.txt'
