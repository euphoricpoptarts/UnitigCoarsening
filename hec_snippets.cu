#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <curand_kernel.h>

#define CUB_STDERR
#include <cub/util_allocator.cuh>
#include <cub/device/device_radix_sort.cuh>

namespace cg = cooperative_groups;
using namespace cg;

using namespace cub;

#define GPU_DEBUG 0

#define NUM_THREADS_PER_BLOCK 32
#define INFTY INT_MAX

static __device__ int ncoarse;
// static __device__ int nrand_accesses;

bool g_verbose = false;
CachingDeviceAllocator g_allocator(true);

/* using ints throughout for simplicity */
typedef struct {
    int n;
    int m;
    int *adj;
    int *num_edges;
    int *eweights;
} graph_t;

struct gpuTimer {
    cudaEvent_t start, stop;
    gpuTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }
    ~gpuTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    void startTimer() {
        cudaEventRecord(start, 0);
    }
    float stopTimer() {
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float elt;
        cudaEventElapsedTime(&elt, start, stop);
        return elt*1e-3;
    }
};

static int load_graph(graph_t *g, const char* const csr_filename) {

    FILE *infp = fopen(csr_filename, "rb");
    assert(infp != NULL);

    long n, m;
    long unused_vals[4];
    assert(fread(&n, sizeof(long), 1, infp) != 0);
    assert(fread(&m, sizeof(long), 1, infp) != 0);
    assert(fread(unused_vals, sizeof(long), 4, infp) != 0);
    g->n = n;
    g->m = m/2;
    fprintf(stdout, "n: %d, m: %d\n", g->n, g->m);

    g->num_edges = (int *) malloc((g->n + 1) * sizeof(int));
    assert(g->num_edges != NULL);
    
    g->adj = (int *) malloc(2 * g->m * sizeof(int));
    assert(g->adj != NULL);

    g->eweights = (int *) malloc(2 * g->m * sizeof(int));
    assert(g->eweights != NULL);

    size_t nitems_read = fread(g->num_edges, sizeof(int), g->n + 1, infp);
    assert(nitems_read == ((size_t) g->n + 1));
    
    nitems_read = fread(g->adj, sizeof(int), 2 * g->m, infp);
    assert(nitems_read == ((size_t) 2 * g->m));
    
    assert(fclose(infp) == 0);
    
    return EXIT_SUCCESS;
}

static int free_graph(graph_t *g) {
    
    if (g->num_edges != NULL) {
        free(g->num_edges);
        g->num_edges = NULL;
    }

    if (g->adj != NULL) {
        free(g->adj);
        g->adj = NULL;
    }

    if (g->eweights != NULL) {
        free(g->eweights);
        g->eweights = NULL;
    }

    return EXIT_SUCCESS;
}

static int copy_graph_to_device(graph_t *d_g, graph_t *g) {

    d_g->n = g->n;
    d_g->m = g->m;

    assert(cudaMalloc(&(d_g->num_edges), (d_g->n + 1) * sizeof(int)) == cudaSuccess);
    assert(cudaMalloc(&(d_g->adj), 2 * d_g->m * sizeof(int)) == cudaSuccess);
    assert(cudaMalloc(&(d_g->eweights), 2 * d_g->m * sizeof(int)) == cudaSuccess);

    assert(cudaMemcpy(d_g->num_edges, g->num_edges, 
                     (d_g->n + 1) * sizeof(int), cudaMemcpyHostToDevice) == cudaSuccess);
    assert(cudaMemcpy(d_g->adj, g->adj, 
                     2 * d_g->m * sizeof(int), cudaMemcpyHostToDevice) == cudaSuccess);
    assert(cudaMemcpy(d_g->eweights, g->eweights, 
                     2 * d_g->m * sizeof(int), cudaMemcpyHostToDevice) == cudaSuccess);

    return EXIT_SUCCESS;
}

static int free_graph_from_device(graph_t *d_g) {

    cudaFree(d_g->num_edges);
    cudaFree(d_g->adj);
    cudaFree(d_g->eweights);
    return EXIT_SUCCESS;
}

__global__ void triad(int * const __restrict C, 
                      const int * const __restrict__ A, 
                      const int * const __restrict__ B,
                      const int array_size) {
    grid_group g = this_grid();
    const int start = g.thread_rank();
    const int incr  = g.size();
    // 4 + 4 + 4 bytes
    for (int i = start; i < array_size; i += incr) { 
        C[i] = A[i] + B[i];
    }
}

__global__ void rng_init(curandStatePhilox4_32_10_t *state) {

    grid_group g = this_grid();
    const int tid = g.thread_rank();
    curand_init(1234, tid, 0, &state[tid]);
}

__global__ void gen_rng_for_perm(unsigned int * __restrict__ R, int * __restrict__ P, 
                const int n, curandStatePhilox4_32_10_t *state) {

    grid_group g    = this_grid();
    const int start = g.thread_rank();
    const int incr  = g.size();

    curandStatePhilox4_32_10_t localState = state[start];
    // unsigned int x = curand(&localState);

    for (int i = start; i < n; i += incr) {
        R[i] = curand(&localState);
        P[i] = i;
    } 
    
    state[start] = localState;
}

// P becomes O, R becomes P
__global__ void gen_perm(unsigned int * __restrict__ R, int * __restrict__ P, 
                const int n) {

    grid_group g    = this_grid();
    const int start = g.thread_rank();
    const int incr  = g.size();

    for (int i = start; i < n; i += incr) {
        R[P[i]] = i;
    } 
} 

__global__ void findH_thr_per_vertex(int * __restrict__ const H,
                curandStatePhilox4_32_10_t *state,
                const int * __restrict__ const O,
                const int * __restrict__ const num_edges, 
                const int * __restrict__ const adj, 
                const int * __restrict__ const eweights,
                const int n, const int m) {

    grid_group g = this_grid();
    const int start = g.thread_rank();
    const int incr  = g.size();

    curandStatePhilox4_32_10_t localState = state[start];
    // unsigned int x = curand(&localState);

    // This piece of code is to compute the max load imbalance with
    // the assumed striped distribution of vertices. This value might be much
    // lower than max degree / avg. degree. 
#if 1
    int load = 0;
    for (int i = start; i < n; i += incr) {
        load += (num_edges[i+1] - num_edges[i]);
    }
    g.sync();
    coalesced_group active = coalesced_threads();
    int max_load = reduce(active, load, greater<int>());
    if (start == 0) {
        printf("Max imbalance: %.3f\n", max_load / ((2.0 * m) / incr));
    }
    g.sync();
#endif

    unsigned int mask = INT_MAX;
    // 4 + 32 + 32 bytes
    for (int i = start; i < n; i += incr) {
        const int adj_start = num_edges[i];
        const int adj_end   = num_edges[i+1];
        const int degree = adj_end - adj_start;
        unsigned int x = (curand(&localState) & mask);
        const int offset = (x % degree); // modulo is expensive
#if GPU_DEBUG
        assert(offset < degree);
        assert(offset >= 0);
#endif
        int vm = adj[adj_start + offset]; // random read
#if 0
        int vm = adj[adj_start];
        int wm = eweights[adj_start];
        int j  = adj_start + 1;
        while (j < adj_end) {
            int v  = adj[j];
            int wv = eweights[j];
            if (wv > wm) {
                wv = wm;
                vm = v;
            }
            j++;
        }
#endif
        // applying permutation here itself
        H[i] = O[vm]; // these writes aren't getting coalesced
                      // maybe write to shared memory first
    }

    // save rng state
    state[start] = localState;

}

__global__ void HEC_mapping(int * __restrict__ M,
                int * __restrict__ d_nc,
                const int * __restrict__ const H,
                const int * __restrict__ const num_edges, 
                const int * __restrict__ const adj, 
                const int * __restrict__ const eweights,
                const int n) {

    grid_group g = this_grid();
    const int start = g.thread_rank();
    const int incr  = g.size();
    
    const int tid   = g.thread_rank();

    if (tid == 0) {
        ncoarse = 0;
        // nrand_accesses = 0;
        d_nc[0] = 0;
    }

    // cyclic distribution of threads to vertices
    // each thread processes n/nthreads vertices
    // 4 bytes
    for (int i = start; i < n; i += incr) {
        M[i] = INFTY;
    }
    
    g.sync();

    // 4 + 32 + 32 * mapped_frac bytes
    for (int u = start; u < n; u += incr) {
        int v = H[u];
#if GPU_DEBUG
        assert(v >= 0);
        assert(v < INFTY);
#endif
        if (M[v] == INFTY) // atomic write or extra random read?
            atomicCAS(&M[v], INFTY, v); // random atomic write
    }

    g.sync();

    // 4 + 32 * unmapped_frac bytes
    for (int u = start; u < n; u += incr) {
        if (M[u] == INFTY) {
            int v = H[u]; // random read
            M[u] = M[v];
        }
    }
    
    g.sync();

    // finish up
    // 4 * mapped_frac + 32 * unmapped_frac + 32 * nrand_accesses (between 1-3) bytes
    for (int u = start; u < n; u += incr) {
        int p = u;
        int x = M[p];
        if (p != x) {
            while (p != (x = M[p])) { // random read
                p = M[x];             // random read
                // atomicAdd(&nrand_accesses, 2);
            }
            // atomicAdd(&nrand_accesses, 1);
            M[u] = p; // write may not be coalesced
        }
    }
    g.sync();

    // determine ncoarse
    // 4 bytes
    for (int u = start; u < n; u += incr) {
        if (M[u] == u) {
            atomicAdd(&ncoarse, 1);
        }
    }
    
    g.sync();

    // Just a check 
#if GPU_DEBUG
    for (int u = start; u < n; u += incr) {
        assert(M[u] != INFTY);
    }
#endif

    if (start == 0) {
        *d_nc = ncoarse;
    }
    
    g.sync();

}

int main(int argc, char **argv) {

    if (argc != 2) {
        fprintf(stderr, "Usage: %s csr_filename\n", argv[0]);
        return EXIT_FAILURE;
    }
    char *filename = argv[1];

    int dev = 0;
    cudaSetDevice(dev);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    if ((deviceProp.major == 9999) && (deviceProp.minor == 9999)) {
        fprintf(stderr, "Error: GPU not detected. Exiting ...\n");
        return EXIT_FAILURE;
    } else {
        fprintf(stdout, "GPU: %s with compute capability %d.%d\n", 
                        deviceProp.name, deviceProp.major, deviceProp.minor);
    }
    const int numSM = deviceProp.multiProcessorCount;
    const int maxThreadsPerSM = deviceProp.maxThreadsPerMultiProcessor;
    fprintf(stdout, "Concurrency: %d (number of SMs: %d, max threads per SM: %d)\n", 
                    numSM * maxThreadsPerSM, numSM, maxThreadsPerSM);
    fprintf(stdout, "Memory bandwidth: %.3lf GB/s (clock rate %.3f MHz, bus width %d bits)\n",
                    2 * deviceProp.memoryClockRate * 1e-6f * deviceProp.memoryBusWidth / 8,
                    deviceProp.memoryClockRate * 1e-3f, deviceProp.memoryBusWidth);
    fprintf(stdout, "Global memory size: %.3f GB\n", deviceProp.totalGlobalMem/1e9); 
    fprintf(stdout, "L2 cache size: %d bytes\n", deviceProp.l2CacheSize);
    fprintf(stdout, "ECC enabled: %d\n", deviceProp.ECCEnabled);
    fprintf(stdout, "Total registers per block: %d\n", deviceProp.regsPerBlock);
    fprintf(stdout, "Total shared memory per block: %zu bytes\n", deviceProp.sharedMemPerBlock);

    int numBlocksPerSmRecommended = 0;
    int numThreads = NUM_THREADS_PER_BLOCK;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSmRecommended, triad, numThreads, dev);

    int numBlocks = (numSM * maxThreadsPerSM) / numThreads;
    fprintf(stdout, "Using a block size of %d, corresponds to %d blocks in all\n", 
                    numThreads, numBlocks);
    fprintf(stdout, "Recommended number of blocks per SM: %d, planned: %d\n", 
                    numBlocksPerSmRecommended, numBlocks/numSM);
    if (numBlocksPerSmRecommended < (numBlocks/numSM)) {
        fprintf(stderr, "Warning: changing number of blocks based on recommendation.\n");
        numBlocks = numSM * numBlocksPerSmRecommended;
    }
    fprintf(stdout, "Registers per thread: %d, shared mem per thread %d bytes\n", 
                    deviceProp.regsPerBlock / numThreads, deviceProp.sharedMemPerBlock / numThreads);

    if (!deviceProp.cooperativeLaunch) {
        fprintf(stderr, "Error: Device does not support cooperative launch. Exiting ...\n");
        return EXIT_FAILURE;
    }

    /* Run a bandwidth benchmark as a sanity check */
    int *d_C, *d_A, *d_B;
    int array_size = 50000000;
    assert(cudaMalloc(&d_C, array_size * sizeof(int)) == cudaSuccess);
    assert(cudaMalloc(&d_A, array_size * sizeof(int)) == cudaSuccess);
    assert(cudaMalloc(&d_B, array_size * sizeof(int)) == cudaSuccess);
 
    int *A = (int *) malloc(array_size * sizeof(int));
    assert(A != NULL);
    for (int i = 0; i < array_size; i++) {
        A[i] = 0;
    }
    assert(cudaMemcpy(d_C, A, array_size * sizeof(int), 
                      cudaMemcpyHostToDevice) == cudaSuccess);
    
    for (int i = 0; i < array_size; i++) {
        A[i] = i;
    }
    assert(cudaMemcpy(d_A, A, array_size * sizeof(int), 
                      cudaMemcpyHostToDevice) == cudaSuccess);
    
    for (int i = 0; i < array_size; i++) {
        A[i] = 1;
    }
    assert(cudaMemcpy(d_B, A, array_size * sizeof(int), 
                      cudaMemcpyHostToDevice) == cudaSuccess);
 
    dim3 dimBlock(numThreads, 1, 1);
    dim3 dimGrid(numBlocks, 1, 1);

    int numTrials = 10;
    
    int i = 0;
    while (i < numTrials) { 
        gpuTimer timer;
        timer.startTimer();
        void *kernelArgs[] = {&d_C, &d_A, &d_B, &array_size};
        cudaError_t err = cudaLaunchCooperativeKernel(((void *) triad), dimGrid, dimBlock, kernelArgs);
        assert(err == 0);
        // the old way of launching kernels 
        // triad<<<numBlocks, numThreads>>>(d_C, d_A, d_B, array_size);
        float elt = timer.stopTimer();
        fprintf(stdout, "Triad time: %.3f ms, bandwidth: %.3f GB/s\n", 
                    elt * 1e3, (3.0 * array_size * sizeof(int) * 1e-9) / elt);
        i++;
    }
    assert(cudaMemcpy(A, d_C, array_size * sizeof(int), 
                      cudaMemcpyDeviceToHost) == cudaSuccess);

    cudaFree(d_C);
    cudaFree(d_A);
    cudaFree(d_B);
    free(A);

    graph_t g;
    load_graph(&g, filename);

    fprintf(stdout, "Estimated mem. use: %.3f MB\n", 
                    ((4.0 * g.n + 16.0 * g.m) + 4.0 * g.n) * 1e-6);
    if ((8.0 * g.n + 16.0 * g.m) > (deviceProp.totalGlobalMem * 1.0)) {
        fprintf(stderr, "Error: Insufficient device memory. Exiting ...\n");
        return EXIT_FAILURE;
    } 
    
    graph_t d_g;
    copy_graph_to_device(&d_g, &g);  

    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSmRecommended, rng_init, 
                                                  numThreads, dev);
    fprintf(stdout, "Recommended number of blocks for rng_init kernel: %d, planned: %d\n", 
                    numBlocksPerSmRecommended, numBlocks/numSM);
    if (numBlocksPerSmRecommended < (numBlocks/numSM)) {
        fprintf(stderr, "Warning: changing number of blocks based on recommendation.\n");
        numBlocks = numSM * numBlocksPerSmRecommended;
    }
    dim3 dimGrid2(numBlocks, 1, 1);

    curandStatePhilox4_32_10_t *devPHILOXStates;
    assert(cudaMalloc((void **) &devPHILOXStates, 
                      numThreads * numBlocks * sizeof(curandStatePhilox4_32_10_t)) == cudaSuccess);
    void *kernelArgs[] = {&devPHILOXStates};
    cudaError_t err = cudaLaunchCooperativeKernel(
                      ((void *) rng_init), 
                      dimGrid2, dimBlock, kernelArgs);
    assert(err == 0);

    unsigned int *d_R;
    assert(cudaMalloc(&d_R, g.n * sizeof(unsigned int)) == cudaSuccess);
    int *d_P;
    assert(cudaMalloc(&d_P, g.n * sizeof(int)) == cudaSuccess);

    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSmRecommended, gen_rng_for_perm, 
                                                  numThreads, dev);
    fprintf(stdout, "Recommended number of blocks for gen_rng kernel: %d, planned: %d\n", 
                    numBlocksPerSmRecommended, numBlocks/numSM);
    assert(numBlocksPerSmRecommended >= (numBlocks/numSM));
   
    i = 0;

    while (i < numTrials) { 
        gpuTimer timer;
        timer.startTimer();
    
        void *kernelArgs[] = {&d_R, &d_P, &(d_g.n), &devPHILOXStates};
        err = cudaLaunchCooperativeKernel(
                      (void *) gen_rng_for_perm,
                      dimGrid2, dimBlock, kernelArgs);
        assert(err == 0);

        float elt = timer.stopTimer();
        fprintf(stdout, "Gen-rng time: %.3f ms, throughput: %.3f ME/s or %.3f MV/s, BW: %.3f GB/s\n", 
                    elt * 1e3, (d_g.m * 1e-6) / elt, (d_g.n * 1e-6) / elt, (8.0 * d_g.n * 1e-9) / elt);
        i++;
    }

    unsigned int *d_R_out;
    assert(cudaMalloc(&d_R_out, g.n * sizeof(unsigned int)) == cudaSuccess);
    int *d_P_out;
    assert(cudaMalloc(&d_P_out, g.n * sizeof(int)) == cudaSuccess);

    size_t  temp_storage_bytes  = 0;
    void    *d_temp_storage     = NULL;
    CubDebugExit(DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_R, d_R_out, 
                            d_P, d_P_out, d_g.n));
    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));
    
    i = 0; 
    while (i < numTrials) { 
        gpuTimer timer;
        timer.startTimer();
        
        CubDebugExit(DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, 
                                d_R, d_R_out, d_P, d_P_out, d_g.n));
        float elt = timer.stopTimer();
        fprintf(stdout, "CUB sort time: %.3f ms, throughput: %.3f Gpairs/s, BW: %.3f GB/s\n", 
                    elt * 1e3, (d_g.n * 1e-9) / elt,  (8 * d_g.n * 1e-9) / elt);
        i++;
    }
    
    CubDebugExit(g_allocator.DeviceFree(d_temp_storage)); 
    cudaFree(d_R); cudaFree(d_P);

    i = 0; 
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSmRecommended, gen_perm, 
                                                  numThreads, dev);
    fprintf(stdout, "Recommended number of blocks for gen_perm kernel: %d, planned: %d\n", 
                    numBlocksPerSmRecommended, numBlocks/numSM);
    if (numBlocksPerSmRecommended != (numBlocks/numSM)) {
        fprintf(stderr, "Warning: changing number of blocks based on recommendation.\n");
        numBlocks = numSM * numBlocksPerSmRecommended;
    }
    dim3 dimGrid3(numBlocks, 1, 1);

    while (i < numTrials) { 
        gpuTimer timer;
        timer.startTimer();

        void *kernelArgs[] = {&d_R_out, &d_P_out, &(d_g.n)};
        cudaError_t err = cudaLaunchCooperativeKernel(((void *) gen_perm), 
                        dimGrid3, dimBlock, kernelArgs);
        assert(err == 0);

        float elt = timer.stopTimer();
        fprintf(stdout, "gen_perm time: %.3f ms, BW: %.3f GB/s\n", 
                    elt * 1e3, (36 * d_g.n * 1e-9) / elt);
        i++;
    }
 

    int *H = (int *) malloc(g.n * sizeof(int));
    assert(H != NULL);
    for (int i = 0; i < g.n; i++) {
        H[i] = INFTY;
    }

    int *d_H;
    assert(cudaMalloc(&d_H, g.n * sizeof(int)) == cudaSuccess);
    assert(cudaMemcpy(d_H, H, g.n * sizeof(int), 
                      cudaMemcpyHostToDevice) == cudaSuccess);

    i = 0;

    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSmRecommended, findH_thr_per_vertex, 
                                                  numThreads, dev);
    fprintf(stdout, "Recommended number of blocks for findH kernel: %d, planned: %d\n", 
                    numBlocksPerSmRecommended, numBlocks/numSM);
    assert(numBlocksPerSmRecommended >= (numBlocks/numSM));
    
    while (i < numTrials) { 
        gpuTimer timer;
        timer.startTimer();

        void *kernelArgs[] = {&d_H, &devPHILOXStates, &d_P_out, &(d_g.num_edges), &(d_g.adj), 
                              &(d_g.eweights), &(d_g.n), &(d_g.m)};
        cudaError_t err = cudaLaunchCooperativeKernel(((void *) findH_thr_per_vertex), 
                        dimGrid2, dimBlock, kernelArgs);
        assert(err == 0);

        float elt = timer.stopTimer();
        fprintf(stdout, "Find H (per vertex) time: %.3f ms, throughput: %.3f ME/s or %.3f MV/s, BW: %.3f GB/s\n", 
                    elt * 1e3, (d_g.m * 1e-6) / elt, (d_g.n * 1e-6) / elt, (68.0 * d_g.n * 1e-9) / elt);
        i++;
    }
  
    i = 0; 
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSmRecommended, HEC_mapping, 
                                                  numThreads, dev);
    fprintf(stdout, "Recommended number of blocks for HEC mapping kernel: %d, planned: %d\n", 
                    numBlocksPerSmRecommended, numBlocks/numSM);
    if (numBlocksPerSmRecommended != (numBlocks/numSM)) {
        fprintf(stderr, "Warning: changing number of blocks based on recommendation.\n");
        numBlocks = numSM * numBlocksPerSmRecommended;
    }
    // dimGrid3(numBlocks, 1, 1);

    int *d_M;
    int *d_nc;
    assert(cudaMalloc(&d_M, g.n * sizeof(int)) == cudaSuccess);
    assert(cudaMalloc(&d_nc, sizeof(int)) == cudaSuccess);

    while (i < numTrials) { 
        gpuTimer timer;
        timer.startTimer();

        void *kernelArgs[] = {&d_M, &d_nc, 
                              &d_H, &(d_g.num_edges), &(d_g.adj), 
                              &(d_g.eweights), &(d_g.n)};
        cudaError_t err = cudaLaunchCooperativeKernel(((void *) HEC_mapping), 
                        dimGrid3, dimBlock, kernelArgs);
        assert(err == 0);

        float elt = timer.stopTimer();
        fprintf(stdout, "HEC time: %.3f ms, throughput: %.3f ME/s or %.3f MV/s, BW: %.3f GB/s\n", 
                    elt * 1e3, (d_g.m * 1e-6) / elt, (d_g.n * 1e-6) / elt, (208 * d_g.n * 1e-9) / elt);
        i++;

        int nc = 0;
        assert(cudaMemcpy(&nc, d_nc, sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess);
        fprintf(stdout, "ncoarse %d\n", nc);
    }


    /* Graph construction */
#if 0
    graph_t d_gc;
    int d_gc.n = nc;

    i = 0;
    while (i < numTrials) { 
        gpuTimer timer;
        timer.startTimer();

        void *kernelArgs[] = {&(d_gc.num_edges), &(d_gc.adj), &(d_gc.eweights),
                              &d_M, &nc, 
                              &(d_g.num_edges), &(d_g.adj), &(d_g.eweights),
                              &(d_g.n), &(d_g.m)}; 
        cudaError_t err = cudaLaunchCooperativeKernel(((void *) construct_coarsegraph_s1), 
                        dimGrid3, dimBlock, kernelArgs);
        assert(err == 0);

        float elt = timer.stopTimer();
        fprintf(stdout, "Graph construct time: %.3f ms, throughput: %.3f ME/s or %.3f MV/s, BW: %.3f GB/s\n", 
                    elt * 1e3, (d_g.m * 1e-6) / elt, (d_g.n * 1e-6) / elt, (208 * d_g.n * 1e-9) / elt);
        i++;

        int nc = 0;
        assert(cudaMemcpy(&nc, d_nc, sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess);
        fprintf(stdout, "ncoarse %d\n", nc);
    }
#endif


    free_graph_from_device(&d_g);
    cudaFree(d_R_out);
    cudaFree(d_P_out);
    cudaFree(d_H);
    cudaFree(d_M);
    cudaFree(d_nc);

    free_graph(&g);
    free(H);

    return EXIT_SUCCESS;
}

