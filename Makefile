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

include $(KOKKOS_PATH)/Makefile.kokkos
SRC = $(wildcard *par.c)
OBJ = $(SRC:.c=.o)

all: standard coarse_ec experiments single_thread_refine

standard: mtx2csr mtx2csr_hg sgpar sgpar_lg sgpar_hg sgpar_c sgpar_kokkos

coarse_ec: sgpar_coarse_ec sgpar_hg_coarse_ec 

experiments: sgpar_exp sgpar_hg_exp

single_thread_refine: sgpar_hg_srefine sgpar_srefine

mtx2csr: mtx2csr.cpp
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -o mtx2csr  mtx2csr.cpp

mtx2csr_hg: mtx2csr_large.cpp
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -o mtx2csr_hg  mtx2csr_large.cpp

sgpar: sgpar.c sgpar.h
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -DMP_REFINE -o sgpar    sgpar.c     $(LDLIBS)

sgpar_coarse_ec: sgpar.c sgpar.h
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -DMP_REFINE -DCOARSE_EIGEN_EC -DEXPERIMENT -o sgpar_coarse_ec sgpar.c     $(LDLIBS)

sgpar_srefine: sgpar.c sgpar.h
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -o sgpar_srefine sgpar.c     $(LDLIBS)

sgpar_exp: sgpar.c sgpar.h
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -DMP_REFINE -DEXPERIMENT -o sgpar_exp    sgpar.c     $(LDLIBS)

sgpar_kokkos: $(OBJ) $(KOKKOS_LINK_DEPENDS)
	$(CXX) $(CXXFLAGS) $(KOKKOS_LDFLAGS) $(OBJ) $(KOKKOS_LIBS) $(LDLIBS) -o sgpar.kokkos

sgpar_lg: sgpar.c sgpar.h
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -DMP_REFINE -DSGPAR_LARGEGRAPHS -o sgpar_lg  sgpar.c     $(LDLIBS)

sgpar_hg: sgpar.c sgpar.h
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -DMP_REFINE -DSGPAR_HUGEGRAPHS -o sgpar_hg  sgpar.c     $(LDLIBS)

sgpar_hg_exp: sgpar.c sgpar.h
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -DMP_REFINE -DSGPAR_HUGEGRAPHS -DEXPERIMENT -o sgpar_hg_exp  sgpar.c     $(LDLIBS)

sgpar_hg_coarse_ec: sgpar.c sgpar.h
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -DMP_REFINE -DSGPAR_HUGEGRAPHS -DCOARSE_EIGEN_EC -DEXPERIMENT -o sgpar_hg_coarse_ec sgpar.c     $(LDLIBS)

sgpar_hg_srefine: sgpar.c sgpar.h
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -DSGPAR_HUGEGRAPHS  -o sgpar_hg_srefine  sgpar.c     $(LDLIBS)

sgpar_c: sgpar.c sgpar.h
	$(CC) $(CFLAGS) $(OMPFLAGS) -DMP_REFINE -o sgpar_c  sgpar.c   -lm $(LDLIBS)

clean:
	rm -f mtx2csr sgpar_c sgpar sgpar_coarse_ec sgpar_lg sgpar_hg sgpar_hg_coarse_ec sgpar.kokkos sgpar_hg_srefine sgpar_srefine sgpar_exp sgpar_hg_exp *.o KokkosCore_config.h KokkosCore_config.tmp

%.o:%.c $(KOKKOS_CPP_DEPENDS)
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(EXTRA_INC) $(OMPFLAGS) -D_KOKKOS -c $<
