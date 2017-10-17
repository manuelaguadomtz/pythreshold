# -*- coding:utf-8 -*-

from distutils.core import setup

setup(
    name='pythreshold',
    version='0.1.1',
    py_modules=['pythreshold/utils', 'pythreshold/local_th/bernsen', 'pythreshold/local_th/bradley_roth',
                'pythreshold/local_th/contrast', 'pythreshold/local_th/feng', 'pythreshold/local_th/lmean',
                'pythreshold/local_th/niblack', 'pythreshold/local_th/nick', 'pythreshold/local_th/sauvola',
                'pythreshold/local_th/singh', 'pythreshold/local_th/wolf', 'pythreshold/global_th/min_err',
                'pythreshold/global_th/otsu', 'pythreshold/global_th/p_tile', 'pythreshold/global_th/two_peaks',
                'pythreshold/global_th/entropy/johannsen', 'pythreshold/global_th/entropy/kapur',
                'pythreshold/global_th/entropy/pun'],
    author=u'Lic. Manuel Aguado Martínez',
    author_email='manuelaguadomtz@gmail.com',
    url='https://www.researchgate.net/profile/Manuel_Aguado_Martinez2',
    description='Numpy/Scipy implementations of some image thresholding algorithms',
    install_requires=['numpy',
                      'scipy',
                      'skimage',
                      'matplotlib']
)
