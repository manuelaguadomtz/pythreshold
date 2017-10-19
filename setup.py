# -*- coding:utf-8 -*-

from setuptools import setup, find_packages

# Get the long description from the README file
with open('README.rst') as f:
    readme = f.read()

setup(
    # Package name
    name='pythreshold',

    # Package version
    version='0.1.8',

    # Included packages
    packages=find_packages(),

    # Package author information
    author=u'BSc. Manuel Aguado Martínez',
    author_email='manuelaguadomtz@gmail.com',
    url='https://www.researchgate.net/profile/Manuel_Aguado_Martinez2',

    # Package requirements
    install_requires=['numpy',
                      'scipy',
                      'scikit-image',
                      'matplotlib'],

    # Package description
    description='Numpy/Scipy implementations of state-of-the-art image thresholding algorithms',
    long_description=readme,
    keywords='thresholding entropy',

)
