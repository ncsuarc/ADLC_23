from PIL import Image
import requests
import json
import os

URL = "http://api:8087/process_img/"

if __name__ == "__main__":
    # DEBUG HARDCODED VALUES:
    # For now use flight_238_im32-33 (hardcoded)
    image_filenames = [
        "./test/test_data/flight_238_im32.jpg",
        "./test/test_data/flight_238_im33.jpg"
    ]

    print(os.listdir("."))
    print(os.listdir("./test"))
    print(os.listdir("./test/test_data"))

    for f in image_filenames:
        im = Image.open(f)
        print(im.size)
        files = {'file_upload': open(f, 'rb')}
        res = requests.post(URL, files=files)

        # print(json.dumps(res.json(), indent=2))
