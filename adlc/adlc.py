from object_detection import TargetDetector
from single_ocr import TesseractCharReader

from PIL import Image
import cv2 

def find_targets(images):
    """
    Returns a list of predicted bounding boxes for targets
    """ 
    # Invoke model
    targetDetector.find_targets(images)
    

def read_character(image) -> str:
    """
    Reads all text in image
    """
    return ocr.read(image)


if __name__ == "__main__":
    targetDetector = TargetDetector()

    images = [Image.open("./data/flight_238/flight_238_im32.jpg"), Image.open("./data/flight_238/flight_238_im33.jpg")]

    print(find_targets(images))

    ocr = TesseractCharReader()
