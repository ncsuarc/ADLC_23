from pathlib import Path
from object_detection import TargetDetector
from single_ocr import TesseractCharReader
from shape_match import match_contour

from PIL import Image
import cv2 
import numpy as np

TESSDATA_PATH = str(Path.home() / ".local/share/tessdata")

def xywh_to_xyxy(x,y,w,h):

    return x,y,x+w,y+h

def find_targets(images):
    """
    Returns a list of predicted bounding boxes for targets
    """ 
    # Invoke model
    # targetDetector.find_targets(images)

    # For now just return test values for flight_238_im32-33
    return [[(2763, 981, 54, 54)],[(2898, 0, 69, 48), (177, 288, 39, 42)]]
    

def read_character(image) -> str:
    """
    Reads all text in image
    """

    return ocr.read(image)

def match_colors(image):
    """
    Returns a tuple for target and text colors respectively
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpen = cv2.filter2D(gray, -1, sharpen_kernel)

    thresh = cv2.threshold(sharpen, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Create contours and order by area
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

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

    cv2.imwrite("tmp/mask.png", fg_mask)

    print(bg_mean[0:3][::-1], fg_mean[0:3][::-1])

    # Return RGB for now, need to make a text conversion
    return bg_mean[0:3][::-1], fg_mean[0:3][::-1]

def match_shapes(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpen = cv2.filter2D(gray, -1, sharpen_kernel)

    thresh = cv2.threshold(sharpen, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Create contours and order by area
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]

    return match_contour(contours[0])


if __name__ == "__main__":
    # targetDetector = TargetDetector()
    ocr = TesseractCharReader(path=TESSDATA_PATH)

    images = [cv2.imread("./data/flight_238/flight_238_im32.jpg"), cv2.imread("./data/flight_238/flight_238_im33.jpg")]

    targets = find_targets(images)

    for i, image in enumerate(images):
        for j, target in enumerate(targets[i]):
            x,y,w,h = target
            img_crop = image[y:y+h, x:x+w]

            # Color check
            match_colors(img_crop)

            # OCR
            print(read_character(img_crop))

            # Shape matching
            print(match_shapes(image))
