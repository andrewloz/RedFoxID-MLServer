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

`pip3 install ultralytics` -- this downloads seperate runtime environments depending on the hardware/model provided in the code. seems to 
automatically find the best way to run the model and installs the relevant drivers for it.

to check it worked, you can run the devices.py script.

### Testing (manual)
add your onnx model to the models directory, you can add any images you want to test by adding to the `./input` directory. Then running the `server.py` followed by the `test_client.py` will take you through a gRPC request cycle of each `.png` you have added to the input directory, and log various perforamnce metrics. please take a look at `test_client.py` to see what metrics you would be seeing here. 


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