import datetime
import time
from tesserocr import PyTessBaseAPI, PSM, get_languages
import string
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import os

# Factort to upscale letter by
SCALE_FACTOR = 3
INTERPOLATION_METHOD = cv2.INTER_LANCZOS4

VALID_CHARS = string.ascii_uppercase + string.digits


class TesseractCharReader:
    def __init__(self, path=os.environ['TESSDATA_PREFIX']) -> None:
        # Initialize tesseract with uppercase alphanumeric and single char recognition
        self.api = PyTessBaseAPI(psm=PSM.SINGLE_CHAR, path=path)
        self.api.SetVariable("tessedit_char_whitelist", VALID_CHARS)

        # Probably will be unused
        # self.osd_api = PyTessBaseAPI(psm=PSM.OSD_ONLY, path=path)

    def read(self, img):
        img = self._preprocess(img)
        self.api.SetImage(img)

        # print(self.api.AllWordConfidences()) # Optionally get confidence values
        return self.api.GetUTF8Text()[0]

    def osd(self, img):
        self.osd_api.SetImage(img)
        os = self.osd_api.DetectOS()
        print(
            "Orientation: {orientation}\nOrientation confidence: {oconfidence}\n"
            "Script: {script}\nScript confidence: {sconfidence}".format(**os)
        )

    def _preprocess(self, img):
        """
        Binarizes and (hopefully) isolates character before returning as a PIL Image.
        This will likely need tuning in the future, and the performance is not promising.

        References these:
        https://stackoverflow.com/a/37914138/6440256
        https://stackoverflow.com/a/62768802/6440256
        """
        img = cv2.resize(
            img,
            None,
            fx=SCALE_FACTOR,
            fy=SCALE_FACTOR,
            interpolation=INTERPOLATION_METHOD,
        )

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpen = cv2.filter2D(gray, -1, sharpen_kernel)

        thresh = cv2.threshold(sharpen, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        contours, hierarchy = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )[-2:]

        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        big_contour = contours[0]
        x, y, w, h = cv2.boundingRect(contours[1])

        # draw filled contour on black background
        mask = np.zeros_like(thresh)
        cv2.drawContours(mask, [big_contour], 0, (255, 255, 255), -1)

        # apply mask to input image
        new_image = cv2.bitwise_and(thresh, mask)

        # crop
        return Image.fromarray(new_image[y : y + h, x : x + w])
