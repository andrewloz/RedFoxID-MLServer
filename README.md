# Welcome
Welcome to the RedfoxID inference server for computer vision models.

### Create environment
`python3.9 -m venv venv`

### Activate environment
`source ./venv/bin/activate`

### Finish setting up torch cuda drivers
for some reason include the cuda drivers in the environment.yml doesn't always work, you have to install manually after activation of the python env. Probably because the installation process requires the environment activated.

A quick not when installing torch, go to the website and install your platform version depending on the hardware you want to run on:  
https://pytorch.org/get-started/locally/

Onnxruntime installation docs:  
https://onnxruntime.ai/docs/install/

Same for onnxruntime, installing process might vary depending on hardware.

`pip3 install ultralytics` -- this downloads seperate runtime environments depending on the hardware/model provided in the code. seems to 
automatically find the best way to run the model and installs the relevant drivers for it.

to check it worked, you can run the devices.py script.

### Testing (manual)
add your onnx model to the models directory, you can add any images you want to test by adding to the `./input` directory. Then running the `server.py` followed by the `test_client.py` will take you through a gRPC request cycle of each `.png` you have added to the input directory, and log various perforamnce metrics. please take a look at `test_client.py` to see what metrics you would be seeing here. 


### Setting up OpenVINO runtime
https://docs.openvino.ai/2025/get-started/install-openvino/install-openvino-apt.html

don't forget to do the additional configuration here: https://docs.openvino.ai/2025/get-started/install-openvino/configurations/configurations-intel-gpu.html
and install the extra deps:
`sudo apt-get install -y ocl-icd-libopencl1 intel-opencl-icd intel-level-zero-gpu level-zero`

you may need to install driver libraries from here: https://dgpu-docs.intel.com/driver/installation.html#ubuntu#ubuntu

double check this reddit post if you struggle to install package repository for you ubuntu system: https://github.com/intel/intel-extension-for-pytorch/issues/365

if selecting a device for openVINO, you can use intel:gpu, intel:npu and intel:cpu with the ultralytics library

# Configs

you can copy the example-config.ini and rename it to config.ini to start with. you will want to be changing values 
under the InferenceServer section.


# Profiling
We can use pythons built in profiling tool `cProfile` like so  
`python -m cProfile -o server_profile.prof`  

And then visualise the results with a tool called snakeviz like so  
```
# install 
pip install snakeviz
snakeviz server_profile.prof
```

# Docker
To get docker using the correct device:

building command
`docker build -t rfid-inference-server .`

## NVIDIA
you need to install the nvidia-container-toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

with that installed you can then run the container with --gpus all set and it should work.

so full run command should be 

`docker run -v "$(pwd)/config.ini:/app/config.ini:ro" -v "$(pwd)/model:/app/model/:ro" --gpus all --rm -p 50051:50051 --name inference-server rfid-inference-server`

### Notes
you might want to remove the --rm if you don't want to install the python deps every time.


## Intel - OpenVINO
the run command is mostly the same, apart from changing --gpus all to --device option

for intel gpu
--device=/dev/dri

for intel npu
--device=/dev/accel

`docker run -v "$(pwd)/config.ini:/app/config.ini:ro" -v "$(pwd)/model:/app/model/:ro" --device={replaceWithDevicePathAbove} -d -p 50051:50051 --name inference-server rfid-inference-server`



