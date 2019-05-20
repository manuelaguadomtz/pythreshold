# -*- coding:utf-8 -*-

from setuptools import setup, find_packages

# Get the long description from the README file
with open('README.md') as f:
    readme = f.read()

setup(
    # Package name
    name='pythreshold',

    # Package version
    version='0.2.1',

    # Included packages
    packages=find_packages(),

    # Package author information
    author=u'BSc. Manuel Aguado Martínez',
    author_email='manuelaguadomtz@gmail.com',

    # Repository URL
    url='https://github.com/manuelaguadomtz/pythreshold',

    # Package requirements
    install_requires=['numpy',
                      'scipy',
                      'scikit-image',
                      'matplotlib'],

    # Package description
    description='Numpy/Scipy implementations of state-of-the-art image'
                ' thresholding algorithms',
    long_description=readme,
    long_description_content_type='text/markdown',
    keywords='thresholding entropy',

    classifiers=[
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python',
        'License :: OSI Approved :: MIT License'
    ],
)
