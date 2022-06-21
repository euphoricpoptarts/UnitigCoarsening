# A unitig compression algorithm using graph coarsening

Use a kmer counter like KMC to compute a list of k-mers into a text file, with kmers separated by line-breaks.
Append a line to the beginning of the kmer file with the number of kmers.

Splits kmers into buckets using L-minimizers, which reduces the memory capacity needed on the GPU significantly.
However, the max problem size is still limited by main memory capacity.

Requires Kokkos and KokkosKernels for parallelism. Build with CMake.
