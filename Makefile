KOKKOS_PATH = ${HOME}/kokkos
KOKKOS_DEVICES = "OpenMP"
KOKKOS_ARCH = "BDW"
.POSIX:
CXX      = g++
CC       = gcc
CXXFLAGS = -Wall -O3     
CFLAGS   = -Wall -O3 -std=c99
OMPFLAGS = -fopenmp
LDLIBS   = 

include $(KOKKOS_PATH)/Makefile.kokkos

all: mtx2csr sgpar sgpar_lg sgpar_hg sgpar_c

mtx2csr: mtx2csr.cpp
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -o mtx2csr  mtx2csr.cpp

sgpar: sgpar.c sgpar.h
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -o sgpar    sgpar.c     $(LDLIBS)

sgpar_kokkos: sgpar.c sgpar.h
	$(CXX) $(CXXFLAGS) $(KOKKOS_LDFLAGS) $(KOKKOS_LIBS) $(OMPFLAGS) -o sgpar    sgpar.c     $(LDLIBS)

sgpar_lg: sgpar.c sgpar.h
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -DSGPAR_LARGEGRAPHS -o sgpar_lg  sgpar.c     $(LDLIBS)

sgpar_hg: sgpar.c sgpar.h
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -DSGPAR_HUGEGRAPHS  -o sgpar_hg  sgpar.c     $(LDLIBS)

sgpar_c: sgpar.c sgpar.h
	$(CC) $(CFLAGS) $(OMPFLAGS) -o sgpar_c  sgpar.c   -lm $(LDLIBS)

clean:
	rm -f mtx2csr sgpar_c sgpar sgpar_lg sgpar_hg

%.o:%.cpp $(KOKKOS_CPP_DEPENDS)
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(EXTRA_INC) -c $<
