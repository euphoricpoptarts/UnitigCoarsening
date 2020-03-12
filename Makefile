.POSIX:
CXX      = g++
CC       = gcc
CXXFLAGS = -Wall -O3 -D USE_GNU_PARALLELMODE
CFLAGS   = -Wall -O3 -std=c11
OMPFLAGS = -fopenmp
LDLIBS   = 

KOKKOS_PATH = ${HOME}/kokkos
KOKKOS_DEVICES = "OpenMP"
KOKKOS_ARCH = "BDW"

#dummy commit

include $(KOKKOS_PATH)/Makefile.kokkos
SRC = $(wildcard *par.c)
OBJ = $(SRC:.c=.o)

all: mtx2csr sgpar sgpar_lg sgpar_hg sgpar_c sgpar_kokkos sgpar_hg_srefine sgpar_srefine

mtx2csr: mtx2csr.cpp
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -o mtx2csr  mtx2csr.cpp

sgpar: sgpar.c sgpar.h
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -DMP_REFINE -o sgpar    sgpar.c     $(LDLIBS)

sgpar_srefine: sgpar.c sgpar.h
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -o sgpar_srefine sgpar.c     $(LDLIBS)

sgpar_kokkos: $(OBJ) $(KOKKOS_LINK_DEPENDS)
	$(CXX) $(CXXFLAGS) $(KOKKOS_LDFLAGS) $(OBJ) $(KOKKOS_LIBS) $(LDLIBS) -o sgpar.kokkos

sgpar_lg: sgpar.c sgpar.h
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -DMP_REFINE -DSGPAR_LARGEGRAPHS -o sgpar_lg  sgpar.c     $(LDLIBS)

sgpar_hg: sgpar.c sgpar.h
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -DMP_REFINE -DSGPAR_HUGEGRAPHS  -o sgpar_hg  sgpar.c     $(LDLIBS)

sgpar_hg_srefine: sgpar.c sgpar.h
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -DSGPAR_HUGEGRAPHS  -o sgpar_hg_srefine  sgpar.c     $(LDLIBS)

sgpar_c: sgpar.c sgpar.h
	$(CC) $(CFLAGS) $(OMPFLAGS) -DMP_REFINE -o sgpar_c  sgpar.c   -lm $(LDLIBS)

clean:
	rm -f mtx2csr sgpar_c sgpar sgpar_lg sgpar_hg sgpar.kokkos sgpar_hg_srefine sgpar_srefine *.o KokkosCore_config.h KokkosCore_config.tmp

%.o:%.c $(KOKKOS_CPP_DEPENDS)
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(EXTRA_INC) $(OMPFLAGS) -D_KOKKOS -c $<
