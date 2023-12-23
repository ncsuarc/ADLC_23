from keras_cv import bounding_box
from object_detection import TargetDetector
import cv2 
from single_ocr import TesseractCharReader

def find_targets(images):
    """
    Returns a list of predicted bounding boxes for targets
    """ 
    # Invoke model and format 
    bounding_boxes = bounding_box.to_ragged(targetDetector.predict(images))
    return bounding_boxes

def read_character(image) -> str:
    """
    Reads all text in image
    """
    return ocr.read(image)


if __name__ == "__main__":
    targetDetector = TargetDetector()
    ocr = TesseractCharReader()
