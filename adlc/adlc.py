from pathlib import Path
from adlc.color_match import rgb_to_text
from adlc.geolocation import geolocate
from adlc.object_detection import TargetDetector
from adlc.single_ocr import TesseractCharReader
from adlc.shape_match import ContourShapeMatcher

from PIL import Image
import cv2
import numpy as np

TESSDATA_PATH = "/tessdata"

class ADLC:
    def __init__(self) -> None:
        # DISABLED for speed during testing
        self.ocr = TesseractCharReader(path=TESSDATA_PATH)
        # self.targetDetector = TargetDetector()
        self.shapeMatcher = ContourShapeMatcher()

    def xywh_to_xyxy(self, x, y, w, h):
        return x, y, x + w, y + h

    def get_center(self, x, y, w, h):
        return x + (0.5 * w), y + (0.5 * h)

    def _get_contours(self, image):
        """
        Thresholds image and finds contours.
        """

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpen = cv2.filter2D(gray, -1, sharpen_kernel)

        _, thresh = cv2.threshold(sharpen, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Create contours and order by area
        contours, hierarchy = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )[-2:]

        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        return thresh, contours

    def find_targets(self, images):
        """
        Returns a list of predicted bounding boxes for targets
        """
        # Invoke model
        # DISABLED for speed during testing
        # targetDetector.find_targets(images)

        # DEBUG HARDCODED VALUES:
        # For now just return test values for flight_238_im32-33
        return [[(2763, 981, 54, 54)], [(2898, 0, 69, 48), (177, 288, 39, 42)]]
        # return [[(2898, 0, 69, 48), (177, 288, 39, 42)]]

    def read_character(self, image) -> str:
        """
        Reads all text in image
        """

        return self.ocr.read(image)

    def match_colors(self, image):
        """
        Returns a tuple for target and text colors respectively
        """
        thresh, contours = self._get_contours(image)

        # Create mask for target background
        bg_mask = np.zeros_like(thresh)
        # Include target
        cv2.drawContours(bg_mask, contours, 0, 255, -1)
        # Subtract text
        cv2.drawContours(bg_mask, contours, 1, 0, -1)
        # Calculate mean
        bg_mean = cv2.mean(image, mask=bg_mask)

        # Repeat for foreground (text)
        # Because of the thin lines, this performs poorly, and may be unnecessary
        fg_mask = np.zeros_like(thresh)
        # Only include text
        cv2.drawContours(fg_mask, contours, 1, 255, -1)
        fg_mean = cv2.mean(image, mask=fg_mask)

        # Convert BGR to RGB and convert to closest color
        bg_text = rgb_to_text(tuple(bg_mean[0:3][::-1]))
        fg_text = rgb_to_text(tuple(fg_mean[0:3][::-1]))
        
        return bg_text, fg_text

    def match_shapes(self, image):
        image = cv2.resize(image, (50,50))

        _, contours = self._get_contours(image)

        return self.shapeMatcher.match_contour(contours[0])

    def process_images(self, images, metadata):
        """
        Perform DLC on a batch of images and return list of target centers + characteristics
        for each.

        Metadata currently unused, but may be used to store the orientation of the aircraft
        for each image to estimate location. For now, only finding coord in image.
        """

        # Perform object detection
        targets = self.find_targets(images)

        # Holds results for each image passed
        batch_results = []

        # Iterate over detected objects
        for i, image in enumerate(images):
            # Store a list of dicts for each target detected
            image_targets = []

            for j, target in enumerate(targets[i]):
                x, y, w, h = target
                img_crop = image[y : y + h, x : x + w]

                # Pixel center of target
                center_x = x + (w // 2)
                center_y = y + (h // 2)

                # Geolocate target
                long, lat = geolocate(center_x, center_y, metadata)

                # Color check
                bg_color, txt_color = self.match_colors(img_crop)

                # OCR
                char = self.read_character(img_crop)

                # Shape matching
                shape = self.match_shapes(img_crop)

                # Store characteristics as key-value dict and add current image
                image_targets += [
                    {
                        "xywh": (x,y,w,h),
                        "bg_color": bg_color,
                        "txt_color": txt_color,
                        "predicted_character": char,
                        "predicted_shape": shape,
                        "longitude": long,
                        "latitude": lat,
                    }
                ]
            batch_results += image_targets

        return batch_results
