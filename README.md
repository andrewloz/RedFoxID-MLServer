Welcome to the RedfoxID inference server for computer vision models.
# Setting up environment

### create environment
`python3.10 -m venv venv`

### activate environment
`source ./venv/bin/activate`

### finish setting up torch cuda drivers
for some reason include the cuda drivers in the environment.yml doesn't always work, you have to install manually after activation of the python env. Probably because the installation process requires the environment activated.

`pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126`
`pip3 install onnxruntime-gpu` --- or whatever version you need.
`pip3 install ultralytics onnx rfdetr`

get latest version for your driver from https://pytorch.org/get-started/locally/

to check it worked, you can run the devices.py script.
