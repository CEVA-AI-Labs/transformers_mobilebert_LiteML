# Introduction 
This repository is a forked from: 
https://github.com/huggingface/transformers

And it is specifically focused on supporting MobileBert with the LiteML functionality.
LiteML version supporting this project:
* ailabs-liteml     0.2.10
* ailabs-pruning    0.4.8
* ailabs-qat        0.2.20
* ailabs-shared     0.2.11

# Prerequisits
1) Before we run the example we need to make sure we created a new virtual environment
2) activate it and run the "install.sh" script, This will install the .whl files
3) install requirements.
4) make sure you installed transformers version 4.25.1 in the previous part 

# Getting Started

There are 3 modes of operation in this script:
* Floating point mode
  In this mode we simply run the pytorch floating point model "as-is" on the SQUAD validation set 
* PTQ mode
  Performing PTQ on the network
* QAT mode
  Performing QAT on the network - requires GPU

### Important
* In both PTQ and QAT modes we use the SQUAD training set for quantization and the valudation set is run on the resulting, quantized model.
* In floating point and in PTQ modes - we support both gpu and cpu processing
* In QAT - we support only gpu - meaning the machine must support CUDA and have a  GPU on it
# Running the code
## floating point mode
run_PTQ.py PyTorch_Float cpu/gpu
## PTQ mode
run_PTQ.py PyTorch_8Bit cpu/gpu
## QAT mode






more details on the quantization process can be found in the CEVA documentation.


Thank you and enjoy.
