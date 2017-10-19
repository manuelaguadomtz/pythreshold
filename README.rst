PyThreshold
===========

**PyThreshold** is a python package featuring Numpy/Scipy implementations of state-of-the-art image thresholding algorithms.

Installing
----------

.. code:: sh

  pip install pythreshold


Usage
-----

.. code:: python

  from pythreshold.utils import test_thresholds
  from scipy.misc import ascent
  
  # Testing all the included thresholding algorithms
  test_thresholds()

  # Testing all the included thresholding algorithms using a custom image
  test_thresholds(ascent())

Additional Information
----------------------

Do you find **PyThreshold** useful? You can collaborate with us:

`Link Gitlab <https://gitlab.com/manuelaguadomtz/pythreshold>`_

Additional materials and information can be found at:

`Link ResearchGate <https://www.researchgate.net/project/Numpy-Scipy-implementations-of-image-thresholding-algorithms>`_
