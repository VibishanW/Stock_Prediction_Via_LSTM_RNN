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

Run Simulation and Synthesis to generate 

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
