from PIL import Image
import requests
import json
import os

XML_NS_NCSU = "http://art1.mae.ncsu.edu/xmp/"
URL = "http://api:8087/process_img/"

if __name__ == "__main__":
    # DEBUG HARDCODED VALUES:
    # For now use flight_238_im32-33 (hardcoded)
    image_filenames = [
        # "./test/test_data/flight_238_im32.jpg",
        "./test/test_data/flight_238_im33.jpg",
        # "./test/test_data/flight_263_im00200.jpg"
    ]

    for f in image_filenames:
        im = Image.open(f)
        files = {'file': (f, open(f, 'rb'), "image/jpeg")}
        res = requests.post(URL, files=files)

        print(json.dumps(res.json(), indent=2))
