from tesserocr import PyTessBaseAPI, PSM
import string

class TesseractCharReader:
    def __init__(self) -> None:
        # Initialize tesseract with uppercase alphanumeric and single char recognition
        self.api = PyTessBaseAPI(psm=PSM.SINGLE_CHAR)
        self.api.SetVariable('tessedit_char_whitelist', string.ascii_uppercase+string.digits)

    def read(self, img):
        self.api.SetImage(img)
        return self.api.GetUTF8Text()
