#python3 -m pip install tensorflow[and-cuda]
cudnn_version="8.8.0.121"
cuda_version="cuda12.1"

sudo apt-get install libcudnn8=${cudnn_version}-1+${cuda_version}
sudo apt-get install libcudnn8-dev=${cudnn_version}-1+${cuda_version}
sudo apt-get install libcudnn8-samples=${cudnn_version}-1+${cuda_version}