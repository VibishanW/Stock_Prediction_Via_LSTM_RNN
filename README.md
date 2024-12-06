# Stock_Prediction_Via_LSTM_RNN
This guide provides step by step instructions to build and run a LSTM RNN for Stock Prediction on to a Xilinx U280 target.
It also contains a hardware implementation of a conventional RNN and a python implementation of a LSTM RNN for comparison. 

# Instructions for running RNN in hardware
## Prerequisites
Access to a build machine equipped with Vitis 2023.2 is required. If you don't have access to one, we can offer assistance in providing access. Please refer to the instructions provided at [this link](https://github.com/OCT-FPGA/OCT-Tutorials/blob/master/nercsetup/nerc-vm-guide.md).

## Tools
- Vitis 2023.2

## 1. Clone the repository
### Repository
```bash
git clone https://github.com/VibishanW/Stock_Prediction_Via_LSTM_RNN.git
```

## 2. Build
Make sure that ```XILINX_VITIS``` and ```XILINX_XRT``` environment variables are set. This can be done by

```bash
source /tools/Xilinx/Vitis/2023.2/settings64.sh
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
Set top level function to rnn_sequencer
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

Generate rnn_sequence.exe with the following command

```bash
g++ -std=c++17 -I/opt/xilinx/xrt/include -L/opt/xilinx/xrt/lib -o rnn_sequence.exe host.cpp -lxrt_coreutil -pthread
```

Verify that you are on a machine connected to a u280 or access a NERC server with access to one.
Verify that rnn_sequencer.exe and rnn_sequencer.xclbin exist on your machine or the server and cd to its location then use the following command to program the FPGA:

```bash
./rnn_sequence.exe rnn_sequence.xclbin
```

# Instructions for running LSTM RNN in hardware
## Prerequisites
Access to a build machine equipped with Vitis 2023.2 is required. If you don't have access to one, we can offer assistance in providing access. Please refer to the instructions provided at [this link](https://github.com/OCT-FPGA/OCT-Tutorials/blob/master/nercsetup/nerc-vm-guide.md).

## Tools
- Vitis 2023.2

## 1. Clone the repository
### LSTM RNN Repository
```bash
git clone https://github.com/VibishanW/Stock_Prediction_Via_LSTM_RNN/LSTM_RNN_HW
```

## 2. Build
Make sure that ```XILINX_VITIS``` and ```XILINX_XRT``` environment variables are set. This can be done by

```bash
source /tools/Xilinx/Vitis/2023.2/settings64.sh
```

```bash
source /opt/xilinx/xrt/setup.sh
```

In Vitis...
Select 'Create HLS Component'
Configuration file selection can be left to default and select a location for the component
Select 'Empty File" for the config file in the next page

For source files...
Under design files select lstm_rnn.cpp and lstm_rnn.hpp from LSTM_RNN_HW
Set top level function to lstm_sequencer
Under testbench select testbench.cpp, data.txt, and out.gold.dat
and select lstm_sequence for the top function

For Hardware -> Part select 'xcu280-fsvh2892-2L-e'

For Settings...
Select 250MHz for clock speed
Select Vitis Kernel Flow Target for Flow Target
Select Generate a Vitis XO for Package Output Fromat

Run Simulation, Synthesis, and package to generate a XO file.

Then run the following to create the .xclbin and executable

### Instructions on generating Bitstream and Executable
```bash
v++ -l -t hw --platform xilinx_u280_gen3x16_xdma_1_202211_1 -o lstm_sequencer.xclbin <path to .xo>
```

Generate host_xrt (executable) with the following command or cd to Bitstream folder and run make.

```bash
g++ -std=c++17 -I/opt/xilinx/xrt/include -L/opt/xilinx/xrt/lib -o host_xrt.exe host.cpp -lxrt_coreutil -pthread
```

Verify that you are on a machine connected to a u280 or access a NERC server with access to one.
Verify that host_xrt, lstm_sequencer.xclbin, and a desired data.txt exist on your machine or the server and cd to its location then use the following command to program the FPGA:

```bash
./host_xrt lstm_sequencer.xclbin data.txt
```

### Instructions on using prebuilt files in cloud lab
After cloning repository in OCT run the following commands:

```bash
cd Stock_Prediction_Via_LSTM_RNN/LSTM_RNN_HW/Bitstream/
chmod +x host_xrt
./host_xrt lstm_sequencer.xclbin data.dat
cat output.dat 
```

# Instructions for running RNN in software
There are 2 implementations: LSTM_RNN_Via_YFinance uses values S&P500 values via Yahoo Finance API and LSTM_RNN_Via_Input_Files uses 10 input files that are also used in the hardware implementation.

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

For LSTM_RNN_Via_YFinance use...
```bash
pip install tensorflow yfinance numpy pandas scikit-learn matplotlib
```
or for LSTM_RNN_Via_Input_Files use...
```bash
pip install numpy tensorflow scikit-learn
```

## 3. Run
Execute main.py

```bash
python main.py
```

Open Output file for results

# Analytics Tools
## Accuracy Analytics Calculator
To determine accuracy of the software and hardware implementation outputs I created a Python script to calculate percent accuracy.

The data was taken from real S&P500 data from 2020. The first 100 lines are split into 10 files of 10 rows each consecutively. A output_real file of 10 rows contains the next day's actual value. This enables both implementations to create a prediction for the 11th day's values which the python script compares to the real value. It then calculates a percent accuracy for each column (Open, Close, etc) and the implementation as whole by average the percentages out.

To run this tool open LSTM_RNN_SW/Accuracy_Analytics_Calc then run the following commands:
```bash
python -m venv .venv
source .venv/Scripts/activate
pip install numpy tensorflow scikit-learn
python calculateAccuracy
```
The result should be a prediction_accuracy_metrics.txt

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
implementation above. The output/golden output file prints out predictions for each column(opening, closing, etc) in normalized terms between 0 and 1.

Whats next?
For the final iteration of this project I will upgrade the RNN implementation to an LSTM-RNN design. I will also upgrade both implementations to print out multiple rows of predictions.
This will allow me to create accurate comparisons to document percent accuracy data between the designs. Additionally, I will created a script to easily convert between the 
normalized and priced data. After creating a full fledged LSTM-RNN I will be working on optimizing the design.

# Project Update 2:
LSTM-RNN: Implemented full LSTM-RNN and use is Xavier's formula to initialize weights. New files include lstm_rnn.h, lstm_rnn.cpp, testbench, and the same input file is used from the RNN implementation. Unfortunately, I ran into error while trying to simulate the design related to the lstm_sequence method. No obvious errors appear, but i believe it has something to with my Xaviers weight initialization step.
After fixing the errors, the goal is to create a bitstream and exe similar to how the RNN implementation operates.

RNN: In this update I completed the RNN implementation by creating a host.cpp and Makefile to synthesize a rnn_sequence.xclbin and rnn_sequence.exe. The files used to generate the bitstream and .exe are present as well as the prebuilt files. The readme is also updated with instructions how to program the u280 with the RNN hardware implementation.

Whats next?
The LSTM-RNN implementation has some errors that need to be fixed. After these errors are resolved I will be working on data analysis and optimization. This involves creating a output file of predictions to compare to actual data resulting in a percent accuracy for the model. After doing some comparisons between the hardware and software implementations, I plan on looking back at previous papers to see how I can optimize my design and maybe introduce larger data sets.

# Final Update:
LSTM-RNN: 
