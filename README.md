# PyThreshold

**PyThreshold** is a python package featuring Numpy/Scipy implementations of state-of-the-art image thresholding algorithms.

## Installing

**PyThreshold** can be easily installed by typing the following command

    pip install pythreshold

## Usage

    from pythreshold.utils import test_thresholds
    from scipy.misc import ascent

    # Testing all the included thresholding algorithms
    test_thresholds()

    # Testing all the included thresholding algorithms using a custom image
    img = ascent()
    test_thresholds(img)

## Additional Information

Do you find **PyThreshold** useful? You can collaborate with us:

[Gitlab](https://gitlab.com/manuelaguadomtz/pythreshold)

Additional materials and information can be found at:

[ResearchGate](https://www.researchgate.net/project/Numpy-Scipy-implementations-of-image-thresholding-algorithms>)
