# A unitig compression algorithm using graph coarsening

Use a kmer counter like KMC to compute a list of k-mers and a list of k+1-mers. Annotate the first line of each file with the number of k/k+1-mers in each.

Requires Kokkos and KokkosKernels for parallelism. Build with CMake.
