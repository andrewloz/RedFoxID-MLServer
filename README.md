# Welcome
Welcome to the RedfoxID inference server for computer vision models.

### Create environment
`python3.10 -m venv venv`

### Activate environment
`source ./venv/bin/activate`

### Finish setting up torch cuda drivers
for some reason include the cuda drivers in the environment.yml doesn't always work, you have to install manually after activation of the python env. Probably because the installation process requires the environment activated.

A quick not when installing torch, go to the website and install your platform version depending on the hardware you want to run on:  
https://pytorch.org/get-started/locally/

Onnxruntime installation docs:  
https://onnxruntime.ai/docs/install/

Same for onnxruntime, installing process might vary depending on hardware.

`pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126`  
`pip3 install onnxruntime-gpu` --- or whatever version you need.  
`pip3 install ultralytics onnx`

to check it worked, you can run the devices.py script.

### Testing (manual)
add your onnx model to the models directory, you can add any images you want to test by adding to the `./input` directory. Then running the `server.py` followed by the `test_client.py` will take you through a gRPC request cycle of each `.png` you have added to the input directory, and log various perforamnce metrics. please take a look at `test_client.py` to see what metrics you would be seeing here. 
