#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cooperative_groups.h>
#include <curand_kernel.h>

namespace cg = cooperative_groups;
using namespace cg;

#define GPU_DEBUG 0

#define NUM_THREADS_PER_BLOCK 128
#define INFTY INT_MAX

static __device__ int ncoarse;

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
    for (int i = start; i < array_size; i += incr) { 
        C[i] = A[i] + B[i];
    }
}

__global__ void findH_thr_per_vertex(int * __restrict__ const H,
                curandStatePhilox4_32_10_t *state,
                const int * __restrict__ const num_edges, 
                const int * __restrict__ const adj, 
                const int * __restrict__ const eweights,
                const int n, const int m) {

    grid_group g = this_grid();
    const int start = g.thread_rank();
    const int incr  = g.size();

    curandStatePhilox4_32_10_t localState = state[start];
    // unsigned int x = curand(&localState);

    unsigned int mask = INT_MAX;
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
        int vm = adj[adj_start + offset];
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
        H[i] = vm; 
    }
}

__global__ void rng_init(curandStatePhilox4_32_10_t *state) {

    grid_group g = this_grid();
    const int tid = g.thread_rank();
    curand_init(1234, tid, 0, &state[tid]);
}

__global__ void HEC_mapping(int * __restrict__ M,
                int * __restrict__ X,
                int * __restrict__ Y,
                int * __restrict__ d_nc,
                const int * __restrict__ const H,
                const int * __restrict__ const num_edges, 
                const int * __restrict__ const adj, 
                const int * __restrict__ const eweights,
                const int n) {

    grid_group g = this_grid();
    const int start = g.thread_rank();
    const int incr  = g.size();

    if (start == 0) {
        ncoarse = 0;
    }

    for (int i = start; i < n; i += incr) {
        M[i] = INFTY;
        X[i] = INFTY;
        Y[i] = 0;
    }

    if (start == 0) {
        d_nc[0] = 0;
    }

    g.sync();

    for (int u = start; u < n; u += incr) {
        int v = H[u];
#if GPU_DEBUG
        assert(v >= 0);
        assert(v < INFTY);
#endif
        atomicCAS(&X[v], INFTY, u);
    }

    g.sync();

    for (int v = start; v < n; v += incr) {
        int u = X[v];
        if (u != INFTY) {
            if ((atomicAdd(&Y[u], 1) == 0) && (atomicAdd(&Y[v], 1) == 0)) {
                int nc = atomicAdd(&ncoarse, 1);
                M[u] = nc;
                M[v] = nc;
            }
        } 
    }

    g.sync();

    for (int u = start; u < n; u += incr) {
        int v = H[u];
        if (M[u] == INFTY) {
            int cv = M[v];
            if (cv != INFTY) {
                M[u] = cv;
            } else { /* we give up :( */
                int nc = atomicAdd(&ncoarse, 1); // we should also track the "new" verts
                M[u] = nc; // this should probably be AtomicCAS
                M[v] = nc; // and so does this one
            }
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

        void *kernelArgs[] = {&d_H, &devPHILOXStates, &(d_g.num_edges), &(d_g.adj), 
                              &(d_g.eweights), &(d_g.n), &(d_g.m)};
        cudaError_t err = cudaLaunchCooperativeKernel(((void *) findH_thr_per_vertex), 
                        dimGrid2, dimBlock, kernelArgs);
        assert(err == 0);

        float elt = timer.stopTimer();
        fprintf(stdout, "Find H (per vertex) time: %.3f ms, throughput: %.3f ME/s or %.3f MV/s\n", 
                    elt * 1e3, (d_g.m * 1e-6) / elt, (d_g.n * 1e-6) / elt);
        i++;
    }
  
    i = 0; 
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSmRecommended, HEC_mapping, 
                                                  numThreads, dev);
    fprintf(stdout, "Recommended number of blocks for HEC mapping kernel: %d, planned: %d\n", 
                    numBlocksPerSmRecommended, numBlocks/numSM);
    assert(numBlocksPerSmRecommended >= (numBlocks/numSM));
    
    int *d_M;
    int *d_X;
    int *d_Y;
    int *d_nc;
    assert(cudaMalloc(&d_M, g.n * sizeof(int)) == cudaSuccess);
    assert(cudaMalloc(&d_X, g.n * sizeof(int)) == cudaSuccess);
    assert(cudaMalloc(&d_Y, g.n * sizeof(int)) == cudaSuccess);
    assert(cudaMalloc(&d_nc, sizeof(int)) == cudaSuccess);

    while (i < numTrials) { 
        gpuTimer timer;
        timer.startTimer();

        void *kernelArgs[] = {&d_M, &d_X, &d_Y, &d_nc, 
                              &d_H, &(d_g.num_edges), &(d_g.adj), 
                              &(d_g.eweights), &(d_g.n)};
        cudaError_t err = cudaLaunchCooperativeKernel(((void *) HEC_mapping), 
                        dimGrid2, dimBlock, kernelArgs);
        assert(err == 0);

        float elt = timer.stopTimer();
        fprintf(stdout, "HEC time: %.3f ms, throughput: %.3f ME/s or %.3f MV/s\n", 
                    elt * 1e3, (d_g.m * 1e-6) / elt, (d_g.n * 1e-6) / elt);
        i++;

        int nc = 0;
        assert(cudaMemcpy(&nc, d_nc, sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess);
        fprintf(stdout, "ncoarse %d\n", nc);
    }
 
    free_graph_from_device(&d_g);
    cudaFree(d_H);

    free_graph(&g);
    free(H);

    return EXIT_SUCCESS;
}

