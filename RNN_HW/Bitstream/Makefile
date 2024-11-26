# Makefile for building host executable

CXX := g++
CXXFLAGS := -std=c++17 -I/opt/xilinx/xrt/include
LDFLAGS := -L/opt/xilinx/xrt/lib -lxrt_coreutil -pthread
TARGET := host.exe
SRC := host.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(TARGET)

