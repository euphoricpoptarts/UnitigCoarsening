#pragma once

namespace sgpar {

#ifdef __cplusplus
#define SGPAR_API 
#endif // __cplusplus

#define SGPAR_USE_ASSERT
#ifdef SGPAR_USE_ASSERT
#ifndef SGPAR_ASSERT
#include <assert.h>
#define SGPAR_ASSERT(expr) assert(expr)
#endif
#else
#define SGPAR_ASSERT(expr) 
#endif

    /**********************************************************
     *  PCG Random Number Generator
     **********************************************************
     */

     // *Really* minimal PCG32 code / (c) 2014 M.E. O'Neill / pcg-random.org
     // Licensed under Apache License 2.0 (NO WARRANTY, etc. see website)

    typedef struct { uint64_t state;  uint64_t inc; } sgp_pcg32_random_t;

    uint32_t sgp_pcg32_random_r(sgp_pcg32_random_t* rng) {
        uint64_t oldstate = rng->state;
        // Advance internal state
        rng->state = oldstate * 6364136223846793005ULL + (rng->inc | 1);
        // Calculate output function (XSH RR), uses old state for max ILP
        uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
        uint32_t rot = oldstate >> 59u;
        return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
    }

    /**********************************************************
     * Internal
     **********************************************************
     */
#if defined(SGPAR_HUGEGRAPHS)
    typedef uint64_t sgp_vid_t;
#define SGP_INFTY UINT64_MAX
    typedef uint64_t sgp_eid_t;
#elif defined(SGPAR_LARGEGRAPHS)
    typedef uint32_t sgp_vid_t;
#define SGP_INFTY UINT32_MAX
    typedef uint64_t sgp_eid_t;
#else
    typedef uint32_t sgp_vid_t;
#define SGP_INFTY UINT32_MAX
    typedef uint32_t sgp_eid_t;
#endif
    typedef double sgp_real_t;
    typedef sgp_real_t sgp_wgt_t;

#ifndef SGPAR_COARSENING_VTX_CUTOFF
#define SGPAR_COARSENING_VTX_CUTOFF 50
#endif

#ifndef SGPAR_COARSENING_MAXLEVELS
#define SGPAR_COARSENING_MAXLEVELS 100
#endif

    typedef std::atomic<sgp_vid_t> atom_vid_t;
    typedef std::atomic<sgp_eid_t> atom_eid_t;

    static sgp_real_t SGPAR_POWERITER_TOL = 1e-10;
    static sgp_real_t MAX_COARSEN_RATIO = 0.9;

    //100 trillion
#define SGPAR_POWERITER_ITER 100000000000000

    typedef struct {
        sgp_vid_t   nvertices;
        sgp_eid_t   nedges;
        sgp_eid_t* source_offsets;
        sgp_vid_t* edges_per_source;
        sgp_vid_t* destination_indices;
        sgp_wgt_t* weighted_degree;
        sgp_wgt_t* eweights;
    } sgp_graph_t;

#define CHECK_SGPAR(func)                                                      \
{                                                                              \
    int status = (func);                                                       \
    if (status != 0) {                                                         \
        printf("sgpar Error: return value %d at line %d. Exiting ... \n",      \
               status, __LINE__);                                              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_RETSTAT(func)                                                    \
{                                                                              \
    int status = (func);                                                       \
    if (status != 0) {                                                         \
        printf("Error: return value %d at line %d. Exiting ...\n",             \
               status, __LINE__);                                              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

    SGPAR_API double sgp_timer() {
#ifdef _OPENMP
        return omp_get_wtime();
#else
        struct timeval tp;
        gettimeofday(&tp, NULL);
        return (double)(tp.tv_sec + ((1e-6) * tp.tv_usec));
#endif
    }

    namespace sgpar_kokkos {
        typedef Kokkos::OpenMP Device;
        using matrix_type = typename KokkosSparse::CrsMatrix<sgp_wgt_t, sgp_eid_t, Device, void, sgp_eid_t>;
        using graph_type = typename matrix_type::staticcrsgraph_type;
    }
}