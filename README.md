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


### Proto files and gRPC.
we use gRPC, so code is generated using the appropriate libraries for whatever language you want to use. The `.proto` files are in the `./protos` directory. We already have generated code for Go, in the `./go_output` directory.


Here is an example of how we generate Go gRPC code.
```
protoc --go_out=./go_output --go_opt=paths=source_relative     --go-grpc_out=./go_output --go-grpc_opt=paths=source_relative ./protos/detect_object.proto
```


and here is how we generate python gRPC code
```
python -m grpc_tools.protoc -I./protos --python_out=. --pyi_out=. --grpc_python_out=. ./protos/detect_object.proto
```


# Configs

you can copy the example-config.ini and rename it to config.ini to start with. you will want to be changing values 
under the InferenceServer section.


# Profiling

When Profiling is set to `1` in config.ini it exposes Prometheus client at localhost:8000/metrics

You can capture a snapshot and analyze it offline using the provided read_metrics.py utility:

```
curl -s http://localhost:8000/metrics > metrics.txt
python read_metrics.py metrics.txt
```

OR 

``` 
curl http://localhost:8000/metrics | python read_metrics.py -
```

This script parses the Prometheus metrics and prints summaries for: 

**gRPC latency** — overall request time measured per RPC method.   
**Inference phases** — timing breakdown of each stage (deserialize, model, serialize, total) per model.   
**Process stats** — current CPU time and RSS (Resident Set Size), which represents the actual RAM used by the server.

For each metric group, the script reports:  
**count** – number of requests observed  
**avg** – average latency  
**p50 / p95 / p99** – estimated 50th, 95th, and 99th percentile latencies