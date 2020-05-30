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

Or just type in a terminal:

    pythreshold -i /path/to/input/image -o /output/directory/for/thresholded/images

## Included Algorithms

* Global thresholding
    * Parker, J. R. (2010). Algorithms for image processing and
      computer vision. John Wiley & Sons. (**Two peaks**)
    * Parker, J. R. (2010). Algorithms for image processing and
      computer vision. John Wiley & Sons. (**p-tile**)
    * Otsu, Nobuyuki. "A threshold selection method from gray-level
      histograms." IEEE transactions on systems, man, and cybernetics
      9.1 (1979): 62-66.
    * Kittler, J. and J. Illingworth. "On Threshold Selection Using Clustering
      Criteria,"" IEEE Transactions on Systems, Man, and Cybernetics 15, no. 5
      (1985): 652–655.
    * Entropy thresholding
        * Johannsen, G., and J. Bille "A Threshold Selection Method Using
          Information Measures,"" Proceedings of the Sixth International Conference
          on Pattern Recognition, Munich, Germany (1982): 140–143.
        * Kapur, J. N., P. K. Sahoo, and A. K. C.Wong. "A New Method for Gray-Level
          Picture Thresholding Using the Entropy of the Histogram,"" Computer Vision,
          Graphics, and Image Processing 29, no. 3 (1985): 273–285.
        * Pun, T. "A New Method for Grey-Level Picture Thresholding Using the
          Entropy of the Histogram,"" Signal Processing 2, no. 3 (1980): 223–237.
* Global thresholding (Multi-threshold)
    * Entropy thresholding
        * Kapur, J. N., P. K. Sahoo, and A. K. C.Wong. "A New Method for Gray-Level
          Picture Thresholding Using the Entropy of the Histogram,"" Computer Vision,
          Graphics, and Image Processing 29, no. 3 (1985): 273–285.
* Local thresholding
    * Bernsen, J (1986), "Dynamic Thresholding of Grey-Level Images",
      Proc. of the 8th Int. Conf. on Pattern Recognition
    * Bradley, D., & Roth, G. (2007). Adaptive thresholding
      using the integral image. Journal of Graphics Tools, 12(2), 13-21.
    * Parker, J. R. (2010). Algorithms for image processing and
      computer vision. John Wiley & Sons. (**Contrast thresholding**)
    * Meng-Ling Feng and Yap-Peng Tan, "Contrast adaptive thresholding of
      low quality document images", IEICE Electron. Express, Vol. 1, No.
      16, pp.501-506, (2004).
    * Parker, J. R. (2010). Algorithms for image processing and
      computer vision. John Wiley & Sons. (**Local mean thresholding**)
    * Niblack, W.: "An introduction to digital image
      processing" (Prentice- Hall, Englewood Cliffs, NJ, 1986), pp. 115–116
    * Sauvola, J., Seppanen, T., Haapakoski, S., and Pietikainen, M.:
      "Adaptive document thresholding". Proc. 4th Int. Conf. on Document
      Analysis and Recognition, Ulm Germany, 1997, pp. 147–152.
    * Singh, O. I., Sinam, T., James, O., & Singh, T. R. (2012). Local contrast
      and mean based thresholding technique in image binarization. International
      Journal of Computer Applications, 51, 5-10.
    * C. Wolf, J-M. Jolion, "Extraction and Recognition of Artificial Text in
      Multimedia Documents", Pattern Analysis and Applications, 6(4):309-326, (2003).


## Additional Information

Do you find **PyThreshold** useful? You can collaborate with us:

[GitHub](https://github.com/manuelaguadomtz/pythreshold)

Additional materials and information can be found at:

[ResearchGate](https://www.researchgate.net/project/Numpy-Scipy-implementations-of-image-thresholding-algorithms>)
