# -*- coding:utf-8 -*-

from distutils.core import setup

setup(
    name='pythreshold',
    version='1.0.0',
    py_modules=['test_thresholds', 'local_th/bernsen', 'local_th/bradley_roth', 'local_th/contrast',
                'local_th/feng', 'local_th/lmean', 'local_th/niblack', 'local_th/nick', 'local_th/sauvola',
                'local_th/singh', 'local_th/wolf', 'global_th/min_err', 'global_th/otsu', 'global_th/p_tile',
                'global_th/two_peaks', 'global_th/entropy/johannsen', 'global_th/entropy/kapur',
                'global_th/entropy/pun'],
    author=u'Lic. Manuel Aguado Martínez',
    author_email='manuelaguadomtz@gmail.com',
    url='https://www.researchgate.net/profile/Manuel_Aguado_Martinez2',
    description='Numpy/Scipy implementations of some image thresholding algorithms',
)
