# ADLC

ADLC is deployed as a FastAPI service on port `8087` via Docker Compose. The `fastapi-adlc` provides a single `POST` endpoint `process_img` that takes a single image and returns detections and properties.

Example usage (see `test/test_adlc.py`):

```python
from PIL import Image
import requests
import json

im = Image.open(filename)
files = {'file': (filename, open(filename, 'rb'), "image/jpeg")}
res = requests.post("http://api:8087/process_img/", files=files)

print(json.dumps(res.json(), indent=2))
```

## Limitations / TODOs

* As of right now __object detection is disabled__ pending re-training on new targets.

* Geolocation needs to be calibrated and debugged

* Although OCR is attempted, the images are usually too blurry

* Similarly, shape detection usually will just guess circles since contours are too hard to see.

## Data Annotation

For now, I have done data annotation using OpenCVs GUI tool [`opencv_annotation`](https://docs.opencv.org/4.x/dc/d88/tutorial_traincascade.html#Preparation-of-the-training-data), which uses a XYWH bounding-box format. There are annotations in `data/annotation_238.txt`.

To include the corresponding images, you will need to download them from the Kraken computer and place them in `data/flight_238/*.jpg`. They are located in `/RAID/Flights/Flight_238/*.jpg`.

## Setup Development Environment

All of the following is included in the Dockerfiles, but if you want to set up locally:

### Pip

```sh
sudo apt install pip
pip install -r requirements.txt
```

### OpenCV

OpenCV binaries must be set up in addition to the Python wrapper libraries

```sh
export OPENCV_VERSION=4.5.4
sudo apt install libopencv-dev python3-opencv
```

### Setup Tesseract

Tesseraact is the OCR library used by the `tesserocr` wrapper. It is relatively large and can be tempermental.

```sh
sudo apt install tesseract-ocr libtesseract-dev libleptonica-dev pkg-config

curl -o ~/.local/share/tessdata/eng.traineddata https://github.com/tesseract-ocr/tessdata/raw/main/eng.traineddata
```

### Using CuDNN Acceleration on VCL

NCSU provides VLCs with RTX 2080 GPUs that can be used for training the CNN quickly. CUDA is already installed on these systems but you will need to install CuDNN as well:

```sh
sudo apt install libcudnn8 libcudnn8-dev libcudnn8-samples
```

To check that CuDNN was set up correctly, run built-in test suite:

```sh
cp -r /usr/src/cudnn_samples_v8/ $HOME
cd  $HOME/cudnn_samples_v8/mnistCUDNN
sudo apt install libfreeimage3 libfreeimage-dev
make clean && make
./mnistCUDNN
```

See [cuDNN install guide](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#package-manager-ubuntu-install) for more info.
