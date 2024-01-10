#!/bin/sh

# Speed up setup on VCL Ubuntu+CUDA machine, untested elsewhere
# Run in base of repo

# Install pip and requirements
sudo apt install -y pip

# Create and activate venv (optional)
python3 -m venv .venv
source .venv/bin/activate

# Auto confirm (since no '-y' option)
yes | pip install -r requirements.txt

# Tesseract requirements
sudo apt install -y tesseract-ocr libtesseract-dev libleptonica-dev pkg-config

# Tessdata (english and OSD)
mkdir ~/.local/share/tessdata
curl -LJ -o ~/.local/share/tessdata/eng.traineddata https://github.com/tesseract-ocr/tessdata/raw/main/eng.traineddata
curl -LJ -o ~/.local/share/tessdata/osd.traineddata https://github.com/tesseract-ocr/tessdata/raw/main/osd.traineddata

# CuDNN
sudo apt install  -y libcudnn8 libcudnn8-dev libcudnn8-samples

# Tensorflow GPU support
# Auto confirm
yes | pip install tensorflow[and-cuda]