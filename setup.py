# -*- coding:utf-8 -*-

from setuptools import setup, find_packages

setup(
    # Package name
    name='pythreshold',

    # Package version
    version='0.1.2',

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
    description='Numpy/Scipy implementations of some image thresholding algorithms',
    keywords='thresholding entropy'

)
