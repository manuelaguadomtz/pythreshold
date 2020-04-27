import argparse

from os.path import basename

import cv2

from .utils import test_thresholds


def test_thresholds_main():
    """Main entry point for the test thresholds script"""

    # Parsing arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Input image")
    ap.add_argument("-o", "--out_dir", required=True, help="Output directory")
    args = ap.parse_args()

    # Reading image
    img = cv2.imread(args.image, 0)

    if img is None:
        print("Invalid input image")
        return

    img_name = basename(args.image).split(".")[0]

    test_thresholds(img, args.out_dir, img_name)
