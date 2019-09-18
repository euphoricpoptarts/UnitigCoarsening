.POSIX:
CXX      = g++
CXXFLAGS = -Wall -fopenmp -O3
LDLIBS   = 

all: mtx2csr cmg

mtx2csr: mtx2csr.cpp
	$(CXX) $(CXXFLAGS) -o mtx2csr mtx2csr.cpp

cmg: cmg.cpp
	$(CXX) $(CXXFLAGS) -o cmg     cmg.cpp     $(LDLIBS)

clean:
	rm -f mtx2csr cmg
