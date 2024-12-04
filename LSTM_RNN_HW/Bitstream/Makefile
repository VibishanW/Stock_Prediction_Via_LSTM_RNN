# Makefile for building XRT-based host application targeting U280

SHELL := /bin/bash

# Target device
DEVICE := xilinx_u280_gen3x16_xdma_1_202211_1

# Host compiler and flags
CXX := g++
CXXFLAGS := -std=c++17 -g -Wall -I$(XILINX_XRT)/include
LDFLAGS := -L$(XILINX_XRT)/lib -lxrt_coreutil -pthread

# Executable and source files
EXECUTABLE := host_xrt
HOST_SRCS := host.cpp

# Default target
all: $(EXECUTABLE)

$(EXECUTABLE): $(HOST_SRCS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

# Clean target
clean:
	rm -f $(EXECUTABLE) *.o
