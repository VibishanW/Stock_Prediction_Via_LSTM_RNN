# Stock_Prediction_Via_LSTM_RNN
This guide provides step by step instructions to build and run a LSTM RNN for Stock Prediction on to a Xilinx U280 target.
It also contains a hardware implementation of a conventional RNN and a python implementation of a LSTM RNN for comparison. 

# Instructions for running RNN in hardware
## Prerequisites
Access to a build machine equipped with Vitis 2023.1 is required. If you don't have access to one, we can offer assistance in providing access. Please refer to the instructions provided at [this link](https://github.com/OCT-FPGA/OCT-Tutorials/blob/master/nercsetup/nerc-vm-guide.md).

## Tools
- Vitis 2022.2 or Vitis 2023.1

## 1. Clone the repository
```bash
git clone https://github.com/VibishanW/Stock_Prediction_Via_LSTM_RNN/RNN_HW
```

## 2. Build
Make sure that ```XILINX_VITIS``` and ```XILINX_XRT``` environment variables are set. This can be done by

```bash
source /tools/Xilinx/Vitis/2022.2/settings64.sh
```

```bash
source /opt/xilinx/xrt/setup.sh
```

In Vitis...
Select 'Create HLS Component'
Configuration file selection can be left to default and select a location for the component
Select 'Empty File" for the config file in the next page

For source files...
Under design files select rnn.cpp and rnn.hpp from RNN_HW
Under testbench select testbench.cpp, testbench.hpp, data.txt, and out.gold.dat
and select rnn_sequence for the top function

For Hardware -> Part select 'xcu280-fsvh2892-2L-e'

For Settings...
Select 250MHz for clock speed
Select Vitis Kernel Flow Target for Flow Target
Select Generate a Vitis XO for Package Output Fromat

Run Simulation, Synthesis, and package to generate a XO file.

Then run the following to create the .xclbin

```bash
v++ -l -t hw --platform xilinx_u280_gen3x16_xdma_1_202211_1 -o rnn_sequencer.xclbin <path to .xo>
```

# Instructions for running RNN in software
## 1. Clone the repository

```bash
git clone https://github.com/VibishanW/Stock_Prediction_Via_LSTM_RNN/RNN_SW
```
cd into file location

## 2. Build
Set up environment variables and install dependencies

```bash
python -m venv .venv
```

```bash
source .venv/Scripts/activate
```

```bash
pip install tensorflow yfinance numpy pandas scikit-learn matplotlib
```

## 3. Run
Execute main.py

```bash
python main.py
```

Open Output file for results

# Project Update 1:
LSTM-RNN:
For this project update I first created a python project that takes yahoo finance data for the S&P500, a popular index fund.
Using a years worth of daily time scale data I scraped opening, closing, low, high, and volume data. 
Tensorflow was used to implement a RNN that analyzes the prices or data in each column to output a prediction for the next day.
The goal of this python implementation is to gain an understanding of how an LSTM-RNN works and understand how data handling works for neural networks.
Additionally, this implementation will act as the basis of comparision when analyzing preformance and accuracy of a hardware acclerated and software implementation of this neural network.

RNN:
While the goal of this project is to implement an LSTM-RNN into a u280, it was important to break down this project into incremental steps.
In this rendition I implemented a RNN to gain a better understanding of the basics of creating projects from scratch on Vitis HLS.
LSTMs also act as a additional improvement to a simple RNN so this implementation is a good half way point in synthesizing a full fledged LSTM-RNN.
This project includes design files rnn.h, rnn.cpp, testbench files, input, and a golden output file. The input file contains a normalized version of the data used in the LSTM-RNN python
implementation above. The output/golden output file prints out precitions for each column(opening, closing, etc) in normalized terms between 0 and 1.

Whats next?
For the final iteration of this project I will upgrade the RNN implementation to an LSTM-RNN design. I will also upgrade both implementations to print out multiple rows of predictions.
This will allow me to create accurate comparisons to document percent accuracy data between the designs. Additionally, I will created a script to easily convert between the 
normalized and priced data. After creating a full fledged LSTM-RNN I will be working on optimizing the design.
