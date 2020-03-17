/** \file    sgpar.h
 *  \brief   Multilevel spectral graph partitioning
 *  \authors Kamesh Madduri, Shad Kirmani, and Michael Gilbert
 *  \date    September 2019
 *  \license MIT License 
 */

#ifndef SGPAR_H_
#define SGPAR_H_

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef _KOKKOS
#include <Kokkos_Core.hpp>
#endif

#ifdef __cplusplus
#include <atomic>
// #define USE_GNU_PARALLELMODE
#ifdef USE_GNU_PARALLELMODE
#include <parallel/algorithm> // for parallel sort
#else 
#include <algorithm>          // for STL sort
#endif

#ifdef EXPERIMENT
#include "ExperimentLoggerUtil.cpp"
#endif
#else
#include <stdatomic.h>
#endif

#ifdef __cplusplus
namespace sgpar {
//extern "C" {
#endif

#ifdef __cplusplus
#define SGPAR_API 
#endif // __cplusplus

#ifndef __cplusplus
#ifdef SGPAR_STATIC
#define SGPAR_API static
#else
#define SGPAR_API extern
#endif // __SGPAR_STATIC
#endif

#define SGPAR_USE_ASSERT
#ifdef SGPAR_USE_ASSERT
#ifndef SGPAR_ASSERT
#include <assert.h>
#define SGPAR_ASSERT(expr) assert(expr)
#endif
#else
#define SGPAR_ASSERT(expr) 
#endif

//typedef struct { uint64_t niters; int max_iter_reached; long edge_cut; long swaps; } sgp_refine_stats;

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
    rng->state = oldstate * 6364136223846793005ULL + (rng->inc|1);
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
typedef sgp_vid_t sgp_wgt_t;
typedef double sgp_real_t;

#ifndef SGPAR_COARSENING_VTX_CUTOFF
#define SGPAR_COARSENING_VTX_CUTOFF 50
#endif

#ifndef SGPAR_COARSENING_MAXLEVELS
#define SGPAR_COARSENING_MAXLEVELS 100
#endif

#ifdef __cplusplus
typedef std::atomic<sgp_vid_t> atom_vid_t;
#endif

static sgp_real_t SGPAR_POWERITER_TOL = 1e-10;

//100 trillion
#define SGPAR_POWERITER_ITER 100000000000000

typedef struct {
    int64_t   nvertices;   
    int64_t   nedges;     
    sgp_eid_t *source_offsets;
    sgp_vid_t *destination_indices;
    sgp_wgt_t *weighted_degree;
    sgp_wgt_t *eweights;
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

SGPAR_API int change_tol(sgp_real_t new_tol){
    SGPAR_POWERITER_TOL = new_tol;

    return EXIT_SUCCESS;
}

SGPAR_API double sgp_timer() {
#ifdef _OPENMP
    return omp_get_wtime();
#else
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double) (tp.tv_sec + ((1e-6)*tp.tv_usec));
#endif
}

SGPAR_API int sgp_coarsen_heavy_edge_matching(sgp_vid_t* vcmap,
                                              sgp_vid_t* nvertices_coarse_ptr,
                                              const sgp_graph_t g,
                                              const int coarsening_level,
                                              sgp_pcg32_random_t* rng) {

	sgp_vid_t n = g.nvertices;

	sgp_vid_t* vperm = (sgp_vid_t*)malloc(n * sizeof(sgp_vid_t));
	SGPAR_ASSERT(vperm != NULL);

	for (sgp_vid_t i = 0; i < n; i++) {
		vcmap[i] = SGP_INFTY;
		vperm[i] = i;
	}


	for (sgp_vid_t i = n - 1; i > 0; i--) {
		sgp_vid_t v_i = vperm[i];
#ifndef SGPAR_HUGEGRAPHS
		uint32_t j = (sgp_pcg32_random_r(rng)) % (i + 1);
#else
		uint64_t j1 = (sgp_pcg32_random_r(rng)) % (i + 1);
		uint64_t j2 = (sgp_pcg32_random_r(rng)) % (i + 1);
		uint64_t j = ((j1 << 32) + j2) % (i + 1);
#endif 
		sgp_vid_t v_j = vperm[j];
		vperm[i] = v_j;
		vperm[j] = v_i;
	}

	sgp_vid_t nvertices_coarse = 0;
	
    if(coarsening_level == 1){
        //match each vertex with its first unmatched neighbor
        for (sgp_vid_t i = 0; i < n; i++) {
            sgp_vid_t u = vperm[i];
            if (vcmap[u] == SGP_INFTY) {
                sgp_vid_t match = u;

                for (sgp_eid_t j = g.source_offsets[u] + 1;
                    j < g.source_offsets[u + 1]; j++) {
                    sgp_vid_t v = g.destination_indices[j];
                    //v must be unmatched to be considered
                    if (vcmap[v] == SGP_INFTY) {
                        j = g.source_offsets[u + 1];//break the loop
                        match = v;
                    }

                }
                sgp_vid_t coarse_vtx = nvertices_coarse++;
                vcmap[u] = coarse_vtx;
                vcmap[match] = coarse_vtx;//u and match are the same when matching with self
            }
        }
    }
    else {        
        //match the vertices, in random order, with the vertex on their heaviest adjacent edge
        //if no unmatched adjacent vertex exists, match with self
        for (sgp_vid_t i = 0; i < n; i++) {
            sgp_vid_t u = vperm[i];
            if (vcmap[u] == SGP_INFTY) {
                sgp_vid_t match = u;
                sgp_wgt_t max_ewt = 0;

                for (sgp_eid_t j = g.source_offsets[u] + 1;
                    j < g.source_offsets[u + 1]; j++) {
                    sgp_vid_t v = g.destination_indices[j];
                    //v must be unmatched to be considered
                    if (max_ewt < g.eweights[j] && vcmap[v] == SGP_INFTY) {
                        max_ewt = g.eweights[j];
                        match = v;
                    }

                }
                sgp_vid_t coarse_vtx = nvertices_coarse++;
                vcmap[u] = coarse_vtx;
                vcmap[match] = coarse_vtx;//u and match are the same when matching with self
            }
        }
    }

	free(vperm);

	*nvertices_coarse_ptr = nvertices_coarse;

	return EXIT_SUCCESS;

}

#ifndef _KOKKOS
SGPAR_API int sgp_coarsen_HEC(sgp_vid_t *vcmap, 
                              sgp_vid_t *nvertices_coarse_ptr, 
                              const sgp_graph_t g, 
                              const int coarsening_level,
                              sgp_pcg32_random_t *rng) {
    sgp_vid_t n = g.nvertices;

    sgp_vid_t *vperm = (sgp_vid_t *) malloc(n * sizeof(sgp_vid_t));
    SGPAR_ASSERT(vperm != NULL);

    for (sgp_vid_t i=0; i<n; i++) {
        vcmap[i] = SGP_INFTY;
        vperm[i] = i;
    }


    for (sgp_vid_t i=n-1; i>0; i--) {
        sgp_vid_t v_i = vperm[i];
#ifndef SGPAR_HUGEGRAPHS
        uint32_t j = (sgp_pcg32_random_r(rng)) % (i+1);
#else
        uint64_t j1 = (sgp_pcg32_random_r(rng)) % (i+1);
        uint64_t j2 = (sgp_pcg32_random_r(rng)) % (i+1);
        uint64_t j  = ((j1<<32) + j2) % (i+1);
#endif 
        sgp_vid_t v_j = vperm[j];
        vperm[i] = v_j;
        vperm[j] = v_i;
    }

    sgp_vid_t *hn = (sgp_vid_t *) malloc(n * sizeof(sgp_vid_t));
    SGPAR_ASSERT(hn != NULL);

	omp_lock_t * v_locks = (omp_lock_t*) malloc(n * sizeof(omp_lock_t));
	SGPAR_ASSERT(v_locks != NULL);

#ifdef __cplusplus
	std::atomic<sgp_vid_t> nvertices_coarse(0);
#else
    _Atomic sgp_vid_t nvertices_coarse = 0;
#endif

#pragma omp parallel
{

	int tid = omp_get_thread_num();
	sgp_pcg32_random_t t_rng;
	t_rng.state = rng->state + tid;
	t_rng.inc = rng->inc;

	for (sgp_vid_t i = 0; i < n; i++) {
		hn[i] = SGP_INFTY;
	}

#pragma omp barrier

	if (coarsening_level == 1) {
#pragma omp for
		for (sgp_vid_t i = 0; i < n; i++) {
			sgp_vid_t adj_size = g.source_offsets[i + 1] - g.source_offsets[i];
			sgp_vid_t offset = (sgp_pcg32_random_r(&t_rng)) % adj_size;
			// sgp_vid_t offset = 0;
			hn[i] = g.destination_indices[g.source_offsets[i] + offset];
		}
	}
	else {
#pragma omp for
		for (sgp_vid_t i = 0; i < n; i++) {
			sgp_vid_t hn_i = g.destination_indices[g.source_offsets[i]];
			sgp_wgt_t max_ewt = g.eweights[g.source_offsets[i]];

			for (sgp_eid_t j = g.source_offsets[i] + 1;
				j < g.source_offsets[i + 1]; j++) {
				if (max_ewt < g.eweights[j]) {
					max_ewt = g.eweights[j];
					hn_i = g.destination_indices[j];
				}

			}
			hn[i] = hn_i;
		}
	}

#pragma omp for
    for(sgp_vid_t i = 0; i < n; i++){
        omp_init_lock(v_locks + i);
    }

#pragma omp for
	for (sgp_vid_t i = 0; i < n; i++) {
		sgp_vid_t u = vperm[i];
		sgp_vid_t v = hn[u];

		sgp_vid_t less = u, more = v;
		if (v < u) {
			less = v;
			more = u;
		}

		omp_set_lock(v_locks + less);
		omp_set_lock(v_locks + more);
		if (vcmap[u] == SGP_INFTY) {
			if (vcmap[v] == SGP_INFTY) {
				vcmap[v] = nvertices_coarse++;
			}
			vcmap[u] = vcmap[v];
		}
		omp_unset_lock(v_locks + more);
		omp_unset_lock(v_locks + less);
	}

#pragma omp for
    for(sgp_vid_t i = 0; i < n; i++){
        omp_destroy_lock(v_locks + i);
    }
}
    
    free(hn);
    free(vperm);
	free(v_locks);

    *nvertices_coarse_ptr = nvertices_coarse;
    
    return EXIT_SUCCESS;
}
#else
SGPAR_API int sgp_coarsen_HEC(sgp_vid_t *vcmap, 
                              sgp_vid_t *nvertices_coarse_ptr, 
                              const sgp_graph_t g, 
                              const int coarsening_level,
                              sgp_pcg32_random_t *rng) {
    Kokkos::initialize();
    {

    sgp_vid_t n = g.nvertices;

    sgp_vid_t *vperm = (sgp_vid_t *) malloc(n * sizeof(sgp_vid_t));
    SGPAR_ASSERT(vperm != NULL);

    sgp_vid_t *hn = (sgp_vid_t *) malloc(n * sizeof(sgp_vid_t));
    SGPAR_ASSERT(hn != NULL);

    Kokkos::parallel_for( n, KOKKOS_LAMBDA ( int i ) {
        vcmap[i] = SGP_INFTY;
        vperm[i] = i;
        hn[i] = SGP_INFTY;
    });


    for (sgp_vid_t i=n-1; i>0; i--) {
        sgp_vid_t v_i = vperm[i];
#ifndef SGPAR_HUGEGRAPHS
        uint32_t j = (sgp_pcg32_random_r(rng)) % (i+1);
#else
        uint64_t j1 = (sgp_pcg32_random_r(rng)) % (i+1);
        uint64_t j2 = (sgp_pcg32_random_r(rng)) % (i+1);
        uint64_t j  = ((j1<<32) + j2) % (i+1);
#endif 
        sgp_vid_t v_j = vperm[j];
        vperm[i] = v_j;
        vperm[j] = v_i;
    }

    if (coarsening_level == 1) {
        for (sgp_vid_t i=0; i<n; i++) {
            sgp_vid_t adj_size = g.source_offsets[i+1]-g.source_offsets[i];
            sgp_vid_t offset = (sgp_pcg32_random_r(rng)) % adj_size;
            // sgp_vid_t offset = 0;
            hn[i] = g.destination_indices[g.source_offsets[i]+offset];
        }
    } else {
        Kokkos::parallel_for( n, KOKKOS_LAMBDA ( int i ) {
            sgp_vid_t hn_i = g.destination_indices[g.source_offsets[i]];
            sgp_wgt_t max_ewt = g.eweights[g.source_offsets[i]];

            for (sgp_eid_t j=g.source_offsets[i]+1; 
                           j<g.source_offsets[i+1]; j++) {
                if (max_ewt < g.eweights[j]) {
                    max_ewt = g.eweights[j];
                    hn_i = g.destination_indices[j];
                }

            }
            hn[i] = hn_i;         
        });
    }

    sgp_vid_t nvertices_coarse = 0;
   
    for (sgp_vid_t i=0; i<n; i++) {
        sgp_vid_t u = vperm[i];
        sgp_vid_t v = hn[u];
        if (vcmap[u] == SGP_INFTY) {
            if (vcmap[v] == SGP_INFTY) {
                vcmap[v] = nvertices_coarse++;
            }
            vcmap[u] = vcmap[v];
        }
    } 
    
    free(hn);
    free(vperm);

    *nvertices_coarse_ptr = nvertices_coarse;
    }
    Kokkos::finalize();

    
    return EXIT_SUCCESS;
}
#endif

typedef struct {
    sgp_vid_t u;
    sgp_vid_t v;
    sgp_vid_t w;
} edge_triple_t;
#ifdef __cplusplus

inline static bool uvw_cmpfn_inc(const edge_triple_t& a, 
                                 const edge_triple_t& b) {
    if (a.u != b.u) {
        return (a.u < b.u); // sort by u, increasing order
    } else {
        if (a.v != b.v) {
            return (a.v < b.v); // sort by v, increasing order
        } else {
            return (a.w > b.w); // sort by w, increasing order
        }
    }
}
#else
static int uvw_cmpfn_inc(const void *a, const void *b) {
    sgp_vid_t *av = ((sgp_vid_t *) a);
    sgp_vid_t *bv = ((sgp_vid_t *) b);
    if (av[0] > bv[0]) {
        return 1;
    }
    if (av[0] < bv[0]) {
        return -1;
    }
    if (*av == *bv) {
        if (av[1] > bv[1])
            return 1;
        if (av[1] < bv[1])
            return -1;
        if (av[1] == bv[1]) {
            if (av[2] < bv[2])
                return 1;
            if (av[2] > bv[2]) 
                return -1;
        }
    }
    return 0;
}
#endif

typedef struct {
    sgp_real_t ev;
    sgp_vid_t  u;
} sgp_vv_pair_t;

#ifdef __cplusplus
inline static bool vu_cmpfn_inc(const sgp_vv_pair_t& a, 
                                const sgp_vv_pair_t& b) {
    if (a.ev != b.ev) {
        return (a.ev < b.ev); // sort by ev
    } else {
        return (a.u < b.u); // sort by u, increasing order
    }
}
#else
static int vu_cmpfn_inc(const void *a, const void *b) {
    sgp_vv_pair_t *av = ((sgp_vv_pair_t *) a);
    sgp_vv_pair_t *bv = ((sgp_vv_pair_t *) b);
    if ((*av).ev > (*bv).ev) {
        return 1;
    }
    if ((*av).ev < (*bv).ev) {
        return -1;
    }
    if ((*av).ev == (*bv).ev) {
        if ((*av).u > (*bv).u)
            return 1;
        else
            return -1;
    }
    return 0;
}
#endif

//assumption: source_offsets[rangeBegin] <= target < source_offsets[rangeEnd] 
//
static sgp_vid_t binary_search_find_source_index(sgp_eid_t *source_offsets, sgp_vid_t rangeBegin, sgp_vid_t rangeEnd, sgp_eid_t target){
    if(rangeBegin + 1 == rangeEnd){
        return rangeBegin;
    }
    int rangeMiddle = (rangeBegin + rangeEnd) >> 1;
    if(source_offsets[rangeMiddle] <= target){
        return binary_search_find_source_index(source_offsets, rangeMiddle, rangeEnd, target);
    } else {
        return binary_search_find_source_index(source_offsets, rangeBegin, rangeMiddle, target);
    }
}

static sgp_eid_t binary_search_find_first_self_loop(edge_triple_t *edges, sgp_eid_t rangeBegin, sgp_eid_t rangeEnd){
    if(rangeBegin + 1 == rangeEnd){
        return rangeEnd;
    }
    int rangeMiddle = (rangeBegin + rangeEnd) >> 1;
    if(edges[rangeMiddle].u != SGP_INFTY){
        return binary_search_find_first_self_loop(edges, rangeMiddle, rangeEnd);
    } else {
        return binary_search_find_first_self_loop(edges, rangeBegin, rangeMiddle);
    }
}

#ifdef _KOKKOS
SGPAR_API int sgp_build_coarse_graph(sgp_graph_t *gc, 
                                     sgp_vid_t *vcmap, 
                                     const sgp_graph_t g, 
                                     const int coarsening_level, 
                                     double *sort_time) {
    sgp_vid_t n  = g.nvertices;
    sgp_eid_t nEdges = g.source_offsets[n];

    sgp_vid_t *edges_uvw;
    edges_uvw = (sgp_vid_t *) malloc(3*nEdges*sizeof(sgp_vid_t));
    SGPAR_ASSERT(edges_uvw != NULL);

Kokkos::initialize();
{

    Kokkos::parallel_for( nEdges, KOKKOS_LAMBDA (sgp_eid_t i) {
        int source_index = binary_search_find_source_index(g.source_offsets, 0, n, i);
        sgp_vid_t u = vcmap[source_index];
        sgp_vid_t v = vcmap[g.destination_indices[i]];
        
        if(u==v){
            //do this to filter self-loops to end of array after sorting    
            edges_uvw[3*i] = SGP_INFTY;
            edges_uvw[3*i+1] = SGP_INFTY;
        } else {
            edges_uvw[3*i] = u;
            edges_uvw[3*i+1] = v;
        }
        if (coarsening_level != 1) {
            edges_uvw[3*i+2] = g.eweights[i];
        } else {
            edges_uvw[3*i+2] = 1;
        }
    });

}
Kokkos::finalize();

    double elt = sgp_timer();
#ifdef __cplusplus
#ifdef USE_GNU_PARALLELMODE
    __gnu_parallel::sort(((edge_triple_t *) edges_uvw), 
                         ((edge_triple_t *) edges_uvw)+nEdges, uvw_cmpfn_inc,
                        __gnu_parallel::quicksort_tag());
#else
    std::sort(((edge_triple_t *) edges_uvw), 
              ((edge_triple_t *) edges_uvw)+nEdges,
              uvw_cmpfn_inc);
#endif
#else
    qsort(edges_uvw, nEdges, 3*sizeof(sgp_vid_t), uvw_cmpfn_inc);
#endif
    *sort_time += (sgp_timer() - elt);
    nEdges = binary_search_find_first_self_loop(((edge_triple_t *) edges_uvw), 0, nEdges);
    sgp_vid_t nc = gc->nvertices;
    sgp_vid_t *gc_degree = (sgp_vid_t *) malloc(nc*sizeof(sgp_vid_t));
    SGPAR_ASSERT(gc_degree != NULL);

    for (sgp_vid_t i=0; i<nc; i++) {
        gc_degree[i] = 0;
    }

    gc_degree[0]++;
    for (sgp_vid_t i=1; i<nEdges; i++) {
        sgp_vid_t prev_u = edges_uvw[3*(i-1)];
        sgp_vid_t prev_v = edges_uvw[3*(i-1)+1];
        sgp_vid_t curr_u = edges_uvw[3*i];
        sgp_vid_t curr_v = edges_uvw[3*i+1];
        if ((curr_u != prev_u) || (curr_v != prev_v)) {
            gc_degree[curr_u]++;
        }
    }

    sgp_eid_t *gc_source_offsets = (sgp_eid_t *) 
                                   malloc((gc->nvertices+1)*sizeof(sgp_eid_t));
    SGPAR_ASSERT(gc_source_offsets != NULL);

    gc_source_offsets[0] = 0;
    for (sgp_vid_t i=0; i<nc; i++) {
        gc_source_offsets[i+1] = gc_source_offsets[i] + gc_degree[i]; 
    }
    sgp_eid_t gc_nedges = gc_source_offsets[nc]/2;

    sgp_vid_t *gc_destination_indices = (sgp_vid_t *) 
                                   malloc(2*gc_nedges*sizeof(sgp_eid_t));
    SGPAR_ASSERT(gc_destination_indices != NULL);

    sgp_wgt_t *gc_eweights = (sgp_wgt_t *) 
                                   malloc(2*gc_nedges*sizeof(sgp_wgt_t));
    SGPAR_ASSERT(gc_eweights != NULL);

    for (sgp_vid_t i=0; i<nc; i++) {
        gc_degree[i] = 0;
    }

    gc_degree[0] = 1;
    gc_destination_indices[0] = edges_uvw[1];
    gc_eweights[0] = edges_uvw[2];
    for (sgp_eid_t i=1; i<nEdges; i++) { 
        sgp_vid_t curr_u = edges_uvw[3*i];
        sgp_vid_t curr_v = edges_uvw[3*i+1];

        sgp_vid_t prev_u = edges_uvw[3*(i-1)];
        sgp_vid_t prev_v = edges_uvw[3*(i-1)+1];
        sgp_eid_t eloc   = gc_source_offsets[curr_u] + gc_degree[curr_u];
        if ((curr_u != prev_u) || (curr_v != prev_v)) {
            gc_destination_indices[eloc] = curr_v;
            gc_eweights[eloc] = edges_uvw[3*i+2];
            gc_degree[curr_u]++;
        } else {
            gc_eweights[eloc-1] += edges_uvw[3*i+2];
        }
    }

    gc->nedges = gc_nedges;
    gc->destination_indices = gc_destination_indices;
    gc->source_offsets = gc_source_offsets;
    gc->eweights = gc_eweights;

    gc->weighted_degree = (sgp_wgt_t *) malloc(nc * sizeof(sgp_wgt_t));
    SGPAR_ASSERT(gc->weighted_degree != NULL);

    sgp_vid_t gcn = gc->nvertices;
    for (sgp_vid_t i=0; i<gcn; i++) {
        sgp_wgt_t degree_wt_i = 0;
        for (sgp_eid_t j=gc->source_offsets[i]; j<gc->source_offsets[i+1]; j++) {
            degree_wt_i += gc->eweights[j];
        }
        gc->weighted_degree[i] = degree_wt_i;
    }


    free(edges_uvw);
    free(gc_degree);
 
    return EXIT_SUCCESS;
}
#else

void parallel_prefix_sum(sgp_eid_t* gc_source_offsets, sgp_vid_t nc, int t_id, int total_threads) {

	//tree-reduction upwards first (largest index contains sum of whole array)
	sgp_vid_t multiplier = 1, prev_multiplier = 1;
	while (multiplier < nc) {
		multiplier <<= 1;
		sgp_vid_t pos = 0;
		//prevent unsigned rollover
		if (nc >= t_id * multiplier) {
			//standard reduction would have sum of whole array in lowest index
			//this makes it easier to compute the indices we need to add
			pos = nc - t_id * multiplier;
		}
#pragma omp barrier
		//strictly greater because gc_source_offsets[0] is always zero
		while (pos > prev_multiplier) {
			gc_source_offsets[pos] = gc_source_offsets[pos] + gc_source_offsets[pos - prev_multiplier];
			//prevent unsigned rollover
			if (pos >= multiplier * total_threads) {
				pos -= multiplier * total_threads;
			}
			else {
				pos = 0;
			}
		}
		prev_multiplier = multiplier;
	}

	//compute left-sums from the root of the tree downwards
	multiplier >>= 1;
	sgp_vid_t next_multiplier = multiplier >> 1;
	while (next_multiplier > 0) {
		sgp_vid_t pos = 0;
		if (nc > (next_multiplier + t_id * multiplier)) {
			pos = nc - (next_multiplier + t_id * multiplier);
		}
		//strictly greater because gc_source_offsets[0] is always zero
#pragma omp barrier
		while (pos > next_multiplier) {
			gc_source_offsets[pos] = gc_source_offsets[pos] + gc_source_offsets[pos - next_multiplier];
			//prevent unsigned rollover
			if (pos >= multiplier * total_threads) {
				pos -= multiplier * total_threads;
			}
			else {
				pos = 0;
			}
		}
		multiplier = next_multiplier;
		next_multiplier >>= 1;
	}
#pragma omp barrier
}

SGPAR_API int sgp_build_coarse_graph(sgp_graph_t *gc, 
                                     sgp_vid_t *vcmap, 
                                     const sgp_graph_t g, 
                                     const int coarsening_level, 
                                     double *sort_time) {
    sgp_vid_t n  = g.nvertices;
    sgp_eid_t nEdges = g.source_offsets[n];

    sgp_vid_t *edges_uvw;
    edges_uvw = (sgp_vid_t *) malloc(3*nEdges*sizeof(sgp_vid_t));
    SGPAR_ASSERT(edges_uvw != NULL);

#pragma omp parallel
{

    int thread_initialized = 0;
    sgp_vid_t u;
    int source_index = 0;

#pragma omp for
	//map fine edges to coarse edges
    for (sgp_eid_t i=0; i<nEdges; i++) {
        if(!thread_initialized){
            source_index = binary_search_find_source_index(g.source_offsets, 0, n, i);
            u = vcmap[source_index];
            thread_initialized = 1;
        } else if(g.source_offsets[source_index + 1] == i) {
            source_index++;
            u = vcmap[source_index];
        }
        sgp_vid_t v = vcmap[g.destination_indices[i]];
        
        if(u==v){
            //do this to filter self-loops to end of array after sorting    
            edges_uvw[3*i] = SGP_INFTY;
            edges_uvw[3*i+1] = SGP_INFTY;
        } else {
            edges_uvw[3*i] = u;
            edges_uvw[3*i+1] = v;
        }
        if (coarsening_level != 1) {
            edges_uvw[3*i+2] = g.eweights[i];
        } else {
            edges_uvw[3*i+2] = 1;
        }
    }

}

    double elt = sgp_timer();
#ifdef __cplusplus
#ifdef USE_GNU_PARALLELMODE
    printf("GNU parallel sort\n");
    __gnu_parallel::sort(((edge_triple_t *) edges_uvw), 
                         ((edge_triple_t *) edges_uvw)+nEdges, uvw_cmpfn_inc,
                        __gnu_parallel::quicksort_tag());
#else
    printf("std sort\n");
    std::sort(((edge_triple_t *) edges_uvw), 
              ((edge_triple_t *) edges_uvw)+nEdges,
              uvw_cmpfn_inc);
#endif
#else
    qsort(edges_uvw, nEdges, 3*sizeof(sgp_vid_t), uvw_cmpfn_inc);
#endif
    *sort_time += (sgp_timer() - elt);
    nEdges = binary_search_find_first_self_loop(((edge_triple_t *) edges_uvw), 0, nEdges);
    sgp_vid_t nc = gc->nvertices;

#ifdef __cplusplus
	//std::atomic<sgp_vid_t> gc_degree[nc] = {};
    atom_vid_t * gc_degree = (atom_vid_t *) malloc(nc * sizeof(atom_vid_t));
#else
	_Atomic sgp_vid_t* gc_degree = (_Atomic sgp_vid_t*) malloc(nc * sizeof(_Atomic sgp_vid_t));
#endif
    SGPAR_ASSERT(gc_degree != NULL);
    for (sgp_vid_t i=0; i<nc; i++) {
        gc_degree[i] = 0;
    }

    gc_degree[0]++;

    //have to ensure these are shared variables, so declare them outside parallel region
    //see malloc initialization of these arrays for explanation
    sgp_vid_t* gc_destination_indices;
    sgp_wgt_t * gc_eweights;

    //I guess because of the (nc + 1), the size of the array is not a shared variable
    //so it won't be a shared array if declared in the parallel region
    sgp_eid_t* gc_source_offsets = (sgp_eid_t*)	malloc((nc + 1) * sizeof(sgp_eid_t));
    SGPAR_ASSERT(gc_source_offsets != NULL);
    gc_source_offsets[0] = 0;

#pragma omp parallel
{

	sgp_vid_t total_threads = omp_get_num_threads();
	sgp_vid_t t_id = omp_get_thread_num();

    //adding total_threads + 1 has the effect of ceil((nEdges - 1) / total_threads)
	sgp_eid_t width = (nEdges - 1 + total_threads - 1) / total_threads;
	sgp_eid_t start_e = 1 + width * t_id;
	sgp_eid_t end_e = start_e + width;
	if (end_e > nEdges) {
		end_e = nEdges;
	}

	//count unique coarse edges
#pragma omp for
	for (sgp_vid_t i = 1; i < nEdges; i++) {
		sgp_vid_t prev_u = edges_uvw[3 * (i - 1)];
		sgp_vid_t prev_v = edges_uvw[3 * (i - 1) + 1];
		sgp_vid_t curr_u = edges_uvw[3 * i];
		sgp_vid_t curr_v = edges_uvw[3 * i + 1];
		if ((curr_u != prev_u) || (curr_v != prev_v)) {
			gc_degree[curr_u]++;
		}
	}

#pragma omp barrier

	//copy into source offsets
	gc_source_offsets[0] = 0;
#pragma omp for
	for (sgp_vid_t i = 0; i < nc; i++) {
		gc_source_offsets[i + 1] = gc_degree[i];
	}

	//prefix sum for source_offsets
	parallel_prefix_sum(gc_source_offsets, nc, t_id, total_threads);

	sgp_eid_t gc_nedges = gc_source_offsets[nc] / 2;
    
#pragma omp single
{
    //gc_nedges is not a shared variable so this must be done by one thread
    //therefore they are not automatically shared, which is why they are declared outside the parallel region
    gc_destination_indices = (sgp_vid_t*)
        malloc(2 * gc_nedges * sizeof(sgp_eid_t));
    gc_eweights = (sgp_wgt_t*)
        malloc(2 * gc_nedges * sizeof(sgp_wgt_t));
}

    SGPAR_ASSERT(gc_destination_indices != NULL);
    SGPAR_ASSERT(gc_eweights != NULL);

#pragma omp barrier

#pragma omp for
	for (sgp_vid_t i = 0; i < nc; i++) {
		gc_degree[i] = 0;
	}

#pragma omp single
	{
		gc_degree[0] = 1;
		gc_destination_indices[0] = edges_uvw[1];
		gc_eweights[0] = edges_uvw[2];
	}

	//don't adjust the start edge for thread 0
	if (t_id > 0) {
		sgp_vid_t curr_u = edges_uvw[3 * start_e];
		sgp_vid_t prev_u = edges_uvw[3 * (start_e - 1)];
		while (curr_u == prev_u && start_e < end_e) {
			start_e++;
			curr_u = edges_uvw[3 * start_e];
			prev_u = edges_uvw[3 * (start_e - 1)];
		}
	}

	if (start_e < end_e && end_e < nEdges) {
		sgp_vid_t curr_u = edges_uvw[3 * end_e];
		sgp_vid_t prev_u = edges_uvw[3 * (end_e - 1)];
		while (curr_u == prev_u && end_e < nEdges) {
			end_e++;
			if (end_e < nEdges) {
				curr_u = edges_uvw[3 * end_e];
				prev_u = edges_uvw[3 * (end_e - 1)];
			}
		}
	}

	//combine weights of like coarse edges and write to destination_indices
	for (sgp_eid_t i = start_e; i < end_e; i++) {
		sgp_vid_t curr_u = edges_uvw[3 * i];
		sgp_vid_t curr_v = edges_uvw[3 * i + 1];

		sgp_vid_t prev_u = edges_uvw[3 * (i - 1)];
		sgp_vid_t prev_v = edges_uvw[3 * (i - 1) + 1];
		sgp_eid_t eloc = gc_source_offsets[curr_u] + gc_degree[curr_u];
		if ((curr_u != prev_u) || (curr_v != prev_v)) {
			gc_destination_indices[eloc] = curr_v;
			gc_eweights[eloc] = edges_uvw[3 * i + 2];
			//only one thread should ever be working with a given curr_u, so there should be no race condition
			gc_degree[curr_u]++;
		}
		else {
			gc_eweights[eloc - 1] += edges_uvw[3 * i + 2];
		}
	}

#pragma omp single
	{
		gc->nedges = gc_nedges;
		gc->destination_indices = gc_destination_indices;
		gc->source_offsets = gc_source_offsets;
		gc->eweights = gc_eweights;
        gc->weighted_degree = (sgp_wgt_t*)malloc(nc * sizeof(sgp_wgt_t));
        SGPAR_ASSERT(gc->weighted_degree != NULL);
	}

	sgp_vid_t gcn = gc->nvertices;
#pragma omp for
	for (sgp_vid_t i = 0; i < gcn; i++) {
		sgp_wgt_t degree_wt_i = 0;
		for (sgp_eid_t j = gc->source_offsets[i]; j < gc->source_offsets[i + 1]; j++) {
			degree_wt_i += gc->eweights[j];
		}
		gc->weighted_degree[i] = degree_wt_i;
	}
}


    free(edges_uvw);
#ifdef __cplusplus
    //gc_degree is on the stack (I think) if using c++
    free(gc_degree);
#else
    free(gc_degree);
#endif
 
    return EXIT_SUCCESS;
}
#endif 

SGPAR_API int sgp_coarsen_one_level(sgp_graph_t *gc, sgp_vid_t *vcmap, 
                                    const sgp_graph_t g, 
                                    const int coarsening_level, 
                                    const int coarsening_alg,
                                    sgp_pcg32_random_t *rng, 
                                    double *sort_time_ptr) {
   
    
    if (coarsening_alg == 0) {
        sgp_vid_t nvertices_coarse;
        sgp_coarsen_HEC(vcmap, &nvertices_coarse, g, coarsening_level, rng);
        gc->nvertices = nvertices_coarse;
	}
	else if (coarsening_alg == 1) {
		sgp_vid_t nvertices_coarse;
		sgp_coarsen_heavy_edge_matching(vcmap, &nvertices_coarse, g, coarsening_level, rng);
		gc->nvertices = nvertices_coarse;
	}

    sgp_build_coarse_graph(gc, vcmap, g, coarsening_level, sort_time_ptr);

    return EXIT_SUCCESS;
}


SGPAR_API int sgp_vec_normalize(sgp_real_t *u, int64_t n) {

    assert(u != NULL);
    sgp_real_t squared_sum = 0;

    for (int64_t i=0; i<n; i++) {
        squared_sum += u[i]*u[i];
    }
    sgp_real_t sum_inv = 1/sqrt(squared_sum);

    for (int64_t i=0; i<n; i++) {
        u[i] = u[i]*sum_inv;
    }
    return EXIT_SUCCESS;
}

#ifdef _KOKKOS
SGPAR_API int sgp_vec_normalize_kokkos(sgp_real_t *u, int64_t n) {

    assert(u != NULL);
    sgp_real_t squared_sum = 0;

    Kokkos::parallel_reduce(n, KOKKOS_LAMBDA(const int64_t& i, sgp_real_t& thread_squared_sum) {
        thread_squared_sum += u[i]*u[i];
    }, squared_sum);
    sgp_real_t sum_inv = 1/sqrt(squared_sum);

    Kokkos::parallel_for(n, KOKKOS_LAMBDA(int64_t i) {
        u[i] = u[i]*sum_inv;
    });
    return EXIT_SUCCESS;
}

SGPAR_API int sgp_vec_dotproduct_kokkos(sgp_real_t *dot_prod_ptr, 
                                 sgp_real_t *u1, sgp_real_t *u2, int64_t n) {

    sgp_real_t dot_prod = 0;
    
    Kokkos::parallel_reduce(n, KOKKOS_LAMBDA(const int64_t& i, sgp_real_t& thread_dot_prod) {
        thread_dot_prod += u1[i]*u2[i];
    }, dot_prod);
    *dot_prod_ptr = dot_prod;
    return EXIT_SUCCESS;
}


SGPAR_API int sgp_vec_orthogonalize_kokkos(sgp_real_t *u1, sgp_real_t *u2, int64_t n) {

    sgp_real_t mult1;
    sgp_vec_dotproduct_kokkos(&mult1, u1, u2, n);

    Kokkos::parallel_for(n, KOKKOS_LAMBDA(int64_t i) {
        u1[i] -= mult1*u2[i];
    });
    return EXIT_SUCCESS;
}

SGPAR_API int sgp_vec_D_orthogonalize_kokkos(sgp_real_t *u1, sgp_real_t *u2, 
                        sgp_wgt_t *D,  int64_t n) {

    //u1[i] = u1[i] - (dot(u1, D*u2)/dot(u2, D*u2)) * u2[i]

    sgp_real_t mult1;
    sgp_vec_dotproduct_kokkos(&mult1, u1, u2, n);

    sgp_real_t mult_numer = 0.0;
    sgp_real_t mult_denom = 0.0;

    Kokkos::parallel_reduce(n, KOKKOS_LAMBDA(const int64_t& i, sgp_real_t& thread_mult_numer) {
        thread_mult_numer += u1[i]*D[i]*u2[i];
    }, mult_numer);
    Kokkos::parallel_reduce(n, KOKKOS_LAMBDA(const int64_t& i, sgp_real_t& thread_mult_denom) {
        thread_mult_denom += u2[i]*D[i]*u2[i];
    }, mult_denom);

    Kokkos::parallel_for(n, KOKKOS_LAMBDA(int64_t i) {
        u1[i] -= mult_numer*u2[i]/mult_denom;
    });
    return EXIT_SUCCESS;
}

SGPAR_API void sgp_power_iter_eigenvalue_log(sgp_real_t *u, sgp_graph_t g){
    sgp_real_t eigenval = 0;
    sgp_real_t eigenval_max = 0;
    sgp_real_t eigenval_min = 2;
    for (sgp_vid_t i=0; i<(g.nvertices); i++) {
        sgp_vid_t weighted_degree = g.source_offsets[i+1]-g.source_offsets[i];
        sgp_real_t u_i = weighted_degree*u[i];
        sgp_real_t matvec_i = 0;
        for (sgp_eid_t j=g.source_offsets[i]; 
                       j<g.source_offsets[i+1]; j++) {
            matvec_i += u[g.destination_indices[j]];
        }
        u_i -= matvec_i;
        sgp_real_t eigenval_est = u_i/u[i];
        if (eigenval_est < eigenval_min) {
            eigenval_min = eigenval_est;
        }
        if (eigenval_est > eigenval_max) {
            eigenval_max = eigenval_est;
        }
        eigenval += (u_i*u_i)*1e9;
    }

    printf("eigenvalue = %1.9lf (%1.9lf %1.9lf), "
                    "edge cut lb %5.0lf "
                    "gap ratio %.0lf\n", 
                    eigenval*1e-9, 
                    eigenval_min, eigenval_max,
                    eigenval*1e-9*(g.nvertices)/4,
                    ceil(1.0/(1.0-eigenval*1e-9)));
}

SGPAR_API int sgp_power_iter(sgp_real_t *u, sgp_graph_t g, int normLap, int final
#ifdef EXPERIMENT
    , ExperimentLoggerUtil& experiment
#endif
) {

    sgp_vid_t n = g.nvertices;

Kokkos::initialize();
{
    sgp_real_t *vec1 = (sgp_real_t *) malloc(n*sizeof(sgp_real_t));
    SGPAR_ASSERT(vec1 != NULL);
    for (sgp_vid_t i=0; i<n; i++) {
        vec1[i] = 1.0;
    }

    sgp_wgt_t gb;
    if(!normLap){
        if(!final){
            gb = 2*g.weighted_degree[0];
            for (sgp_vid_t i=1; i<n; i++) {
                if(gb < 2*g.weighted_degree[i]) {
                    gb = 2*g.weighted_degree[i];
                }
            }
        } else {
            sgp_wgt_t gb = 2*(g.source_offsets[1]-g.source_offsets[0]);
            for (sgp_vid_t i=1; i<n; i++) {
                if (gb < 2*(g.source_offsets[i+1]-g.source_offsets[i])) {
                    gb = 2*(g.source_offsets[i+1]-g.source_offsets[i]);
                }
            }
        }
    }

    sgp_wgt_t *weighted_degree;
    
    if(normLap && final){
        weighted_degree = (sgp_wgt_t *) malloc(n*sizeof(sgp_wgt_t));
        assert(weighted_degree != NULL);
        for (sgp_vid_t i=0; i<n; i++) {
            weighted_degree[i] = g.source_offsets[i+1] - g.source_offsets[i];
        }
    }

#if 0
    sgp_real_t mult = 0;
    sgp_vec_dotproduct_kokkos(&mult, vec1, u, n);
    for (sgp_vid_t i=0; i<n; i++) {
        u[i] -= mult*u[i]/n;
    }
    sgp_vec_normalize_kokkos(u, n);
#endif

    sgp_vec_normalize_kokkos(vec1, n); 
    if(!normLap){
        sgp_vec_orthogonalize_kokkos(u, vec1, n);
    } else {
        sgp_vec_D_orthogonalize_kokkos(u, vec1, g.weighted_degree, n);
    }
    sgp_vec_normalize_kokkos(u, n);

    sgp_real_t *v = (sgp_real_t *) malloc(n*sizeof(sgp_real_t));
    SGPAR_ASSERT(v != NULL);
    for (sgp_vid_t i=0; i<n; i++) {
        v[i] = u[i];
    }

    sgp_real_t tol = SGPAR_POWERITER_TOL;
    uint64_t niter = 0;
    uint64_t iter_max = (uint64_t) SGPAR_POWERITER_ITER / (uint64_t) n;
    sgp_real_t dotprod = 0, lastDotprod = 1;
    while ((fabs(dotprod) < (1-tol)) && (niter < iter_max)) {

        // u = v
        Kokkos::parallel_for(n, KOKKOS_LAMBDA(sgp_vid_t i) {
            u[i] = v[i];
        });

        // v = Lu
        Kokkos::parallel_for(n, KOKKOS_LAMBDA(sgp_vid_t i) {
            // sgp_real_t v_i = g.weighted_degree[i]*u[i];
            sgp_real_t weighted_degree_inv, v_i;
            if(!normLap){
                if(!final){
                    v_i = (gb-g.weighted_degree[i])*u[i];
                } else {
                    sgp_vid_t weighted_degree = g.source_offsets[i+1]-g.source_offsets[i];
                    v_i = (gb-weighted_degree)*u[i];
                }
            } else {
                weighted_degree_inv = 1.0/g.weighted_degree[i];
                v_i = 0.5*u[i];
            }
            sgp_real_t matvec_i = 0;
            for (sgp_eid_t j=g.source_offsets[i]; 
                           j<g.source_offsets[i+1]; j++) {
                if(!final){
                    matvec_i += u[g.destination_indices[j]]*g.eweights[j];
                } else {
                    matvec_i += u[g.destination_indices[j]];
                }
            }
            // v_i -= matvec_i;
            if(!normLap){
                v_i += matvec_i;
            } else {
                v_i += 0.5*matvec_i*weighted_degree_inv;
            }
            v[i] = v_i;
        });

        if(!normLap){
            sgp_vec_orthogonalize_kokkos(v, vec1, n);
        }
        sgp_vec_normalize_kokkos(v, n);
        lastDotprod = dotprod;
        sgp_vec_dotproduct_kokkos(&dotprod, u, v, n);
        niter++;
    }
    int max_iter_reached = 0;
    if (niter >= iter_max) {
        printf("exceeded max iter count, ");
        max_iter_reached = 1;
    }
    printf("number of iterations: %lu\n", niter);

#ifdef EXPERIMENT
    experiment.addCoarseLevel(niter, max_iter_reached, 0);
#endif

    if(!normLap && final){
        sgp_power_iter_eigenvalue_log(u, g);
    }

    free(vec1);
    free(v);
    if(normLap && final){
        free(weighted_degree);
    }

}
Kokkos::finalize();

    return EXIT_SUCCESS;
}
#elif defined(MP_REFINE)
SGPAR_API int sgp_vec_normalize_omp(sgp_real_t *u, int64_t n) {

    assert(u != NULL);
    static sgp_real_t squared_sum = 0;

#pragma omp single
    squared_sum = 0;

#pragma omp for reduction(+:squared_sum)
    for (int64_t i=0; i<n; i++) {
        squared_sum += u[i]*u[i];
    }
    sgp_real_t sum_inv = 1/sqrt(squared_sum);

    #pragma omp single 
    {
        //printf("squared_sum %3.3f\n", squared_sum);
    }

#pragma omp for
    for (int64_t i=0; i<n; i++) {
        u[i] = u[i]*sum_inv;
    }
    return EXIT_SUCCESS;
}

SGPAR_API int sgp_vec_dotproduct_omp(sgp_real_t *dot_prod_ptr, 
                                 sgp_real_t *u1, sgp_real_t *u2, int64_t n) {

    static sgp_real_t dot_prod = 0;

#pragma omp single
    dot_prod = 0;
    
#pragma omp for reduction(+:dot_prod)
    for (int64_t i=0; i<n; i++) {
        dot_prod += u1[i]*u2[i];
    }
    *dot_prod_ptr = dot_prod;

    return EXIT_SUCCESS;
}


SGPAR_API int sgp_vec_orthogonalize_omp(sgp_real_t *u1, sgp_real_t *u2, int64_t n) {

    sgp_real_t mult1;
    sgp_vec_dotproduct_omp(&mult1, u1, u2, n);

#pragma omp for
    for (int64_t i=0; i<n; i++) {
        u1[i] -= mult1*u2[i];
    }
    return EXIT_SUCCESS;
}

SGPAR_API int sgp_vec_D_orthogonalize_omp(sgp_real_t *u1, sgp_real_t *u2, 
                        sgp_wgt_t *D,  int64_t n) {

    //u1[i] = u1[i] - (dot(u1, D*u2)/dot(u2, D*u2)) * u2[i]

    sgp_real_t mult1;
    sgp_vec_dotproduct_omp(&mult1, u1, u2, n);

    static sgp_real_t mult_numer = 0;
    static sgp_real_t mult_denom = 0;

#pragma omp single
{
    mult_numer = 0;
    mult_denom = 0;
}

#pragma omp for reduction(+:mult_numer, mult_denom)
    for (int64_t i=0; i<n; i++) {
        mult_numer += u1[i]*D[i]*u2[i];
        mult_denom += u2[i]*D[i]*u2[i];
    }

#pragma omp for
    for (int64_t i=0; i<n; i++) {
        u1[i] -= mult_numer*u2[i]/mult_denom;
    }

    return EXIT_SUCCESS;
}

SGPAR_API void sgp_power_iter_eigenvalue_log(sgp_real_t *u, sgp_graph_t g){
    sgp_real_t eigenval = 0;
    sgp_real_t eigenval_max = 0;
    sgp_real_t eigenval_min = 2;
    for (sgp_vid_t i=0; i<(g.nvertices); i++) {
        sgp_vid_t weighted_degree = g.source_offsets[i+1]-g.source_offsets[i];
        sgp_real_t u_i = weighted_degree*u[i];
        sgp_real_t matvec_i = 0;
        for (sgp_eid_t j=g.source_offsets[i]; 
                       j<g.source_offsets[i+1]; j++) {
            matvec_i += u[g.destination_indices[j]];
        }
        u_i -= matvec_i;
        sgp_real_t eigenval_est = u_i/u[i];
        if (eigenval_est < eigenval_min) {
            eigenval_min = eigenval_est;
        }
        if (eigenval_est > eigenval_max) {
            eigenval_max = eigenval_est;
        }
        eigenval += (u_i*u_i)*1e9;
    }

    printf("eigenvalue = %1.9lf (%1.9lf %1.9lf), "
                    "edge cut lb %5.0lf "
                    "gap ratio %.0lf\n", 
                    eigenval*1e-9, 
                    eigenval_min, eigenval_max,
                    eigenval*1e-9*(g.nvertices)/4,
                    ceil(1.0/(1.0-eigenval*1e-9)));
}

SGPAR_API int sgp_power_iter(sgp_real_t *u, sgp_graph_t g, const int normLap, const int final
#ifdef EXPERIMENT
    , ExperimentLoggerUtil& experiment
#endif
) {

    sgp_vid_t n = g.nvertices;

    sgp_real_t *vec1 = (sgp_real_t *) malloc(n*sizeof(sgp_real_t));
    SGPAR_ASSERT(vec1 != NULL);

    sgp_wgt_t *weighted_degree;
    
    if(normLap && final){
        weighted_degree = (sgp_wgt_t *) malloc(n*sizeof(sgp_wgt_t));
        assert(weighted_degree != NULL);
        for (sgp_vid_t i=0; i<n; i++) {
            weighted_degree[i] = g.source_offsets[i+1] - g.source_offsets[i];
        }
    }

    sgp_wgt_t gb;
    if(!normLap){
        if(!final){
            gb = 2*g.weighted_degree[0];
            for (sgp_vid_t i=1; i<n; i++) {
                if(gb < 2*g.weighted_degree[i]) {
                    gb = 2*g.weighted_degree[i];
                }
            }
        } else {
            gb = 2*(g.source_offsets[1]-g.source_offsets[0]);
            for (sgp_vid_t i=1; i<n; i++) {
                if (gb < 2*(g.source_offsets[i+1]-g.source_offsets[i])) {
                    gb = 2*(g.source_offsets[i+1]-g.source_offsets[i]);
                }
            }
        }
    }


    uint64_t g_niter = 0;
    uint64_t iter_max = (uint64_t)SGPAR_POWERITER_ITER / (uint64_t)n;

    sgp_real_t* v = (sgp_real_t*)malloc(n * sizeof(sgp_real_t));
    SGPAR_ASSERT(v != NULL);
#pragma omp parallel shared(u)
{

    uint64_t niter = 0;
#pragma omp for
    for (sgp_vid_t i=0; i<n; i++) {
        vec1[i] = 1.0;
    }

#if 0
    sgp_real_t mult = 0;
    sgp_vec_dotproduct_omp(&mult, vec1, u, n);

#pragma omp for
    for (sgp_vid_t i=0; i<n; i++) {
        u[i] -= mult*u[i]/n;
    }
    sgp_vec_normalize_omp(u, n);
#endif

    sgp_vec_normalize_omp(vec1, n); 
    if(!normLap){
        sgp_vec_orthogonalize_omp(u, vec1, n);
    } else {
        sgp_vec_D_orthogonalize_omp(u, vec1, g.weighted_degree, n);
    }
    sgp_vec_normalize_omp(u, n);

#pragma omp for
    for (sgp_vid_t i=0; i<n; i++) {
        v[i] = u[i];
    }

    sgp_real_t tol = SGPAR_POWERITER_TOL;
    sgp_real_t dotprod = 0, lastDotprod = 1;
    while (fabs(dotprod - lastDotprod) > tol && (niter < iter_max)) {

        // u = v
#pragma omp for
        for (sgp_vid_t i=0; i<n; i++) {
            u[i] = v[i];
        }

        // v = Lu
#pragma omp for
        for (sgp_vid_t i=0; i<n; i++) {
            // sgp_real_t v_i = g.weighted_degree[i]*u[i];
            sgp_real_t weighted_degree_inv, v_i;
            if (normLap) {
                if (final) {
                    weighted_degree_inv = 1.0 / weighted_degree[i];
                    v_i = 0.5 * u[i];
                }
                else {
                    weighted_degree_inv = 1.0 / g.weighted_degree[i];
                    v_i = 0.5 * u[i];
                }
            }
            else {
                if (final) {
                    sgp_vid_t weighted_degree = g.source_offsets[i + 1] - g.source_offsets[i];
                    v_i = (gb - weighted_degree) * u[i];
                }
                else {
                    v_i = (gb - g.weighted_degree[i]) * u[i];
                }
            }
            sgp_real_t matvec_i = 0;
            for (sgp_eid_t j=g.source_offsets[i]; 
                           j<g.source_offsets[i+1]; j++) {
                if (final) {
                    matvec_i += u[g.destination_indices[j]];
                }
                else {
                    matvec_i += u[g.destination_indices[j]] * g.eweights[j];
                }
            }
            if (normLap) {
                v_i += 0.5 * matvec_i * weighted_degree_inv;
            }
            else {
                v_i += matvec_i;
            }
            v[i] = v_i;
        }

        if(!normLap){
            sgp_vec_orthogonalize_omp(v, vec1, n);
        }
        sgp_vec_normalize_omp(v, n);
		lastDotprod = dotprod;
        sgp_vec_dotproduct_omp(&dotprod, u, v, n);
        niter++;
    }

#pragma omp single
    {
        g_niter = niter;
    }
}
    free(v);

    int max_iter_reached = 0;
    if (g_niter >= iter_max) {
        printf("exceeded max iter count, ");
        max_iter_reached = 1;
    }
    printf("number of iterations: %lu\n", g_niter);
#ifdef EXPERIMENT
    experiment.addCoarseLevel(niter, max_iter_reached, 0);
#endif
    if(!normLap && final){
        sgp_power_iter_eigenvalue_log(u, g);
    }

    free(vec1);
    if(normLap && final){
        free(weighted_degree);
    }
    return EXIT_SUCCESS;
}
#else
SGPAR_API int sgp_vec_normalize_omp(sgp_real_t* u, int64_t n) {

    assert(u != NULL);
    static sgp_real_t squared_sum = 0;

//#pragma omp single
    squared_sum = 0;

//#pragma omp for reduction(+:squared_sum)
    for (int64_t i = 0; i < n; i++) {
        squared_sum += u[i] * u[i];
    }
    sgp_real_t sum_inv = 1 / sqrt(squared_sum);

//#pragma omp single 
    {
        //printf("squared_sum %3.3f\n", squared_sum);
    }

//#pragma omp for
    for (int64_t i = 0; i < n; i++) {
        u[i] = u[i] * sum_inv;
    }
    return EXIT_SUCCESS;
}

SGPAR_API int sgp_vec_dotproduct_omp(sgp_real_t* dot_prod_ptr,
    sgp_real_t* u1, sgp_real_t* u2, int64_t n) {

    static sgp_real_t dot_prod = 0;

//#pragma omp single
    dot_prod = 0;

//#pragma omp for reduction(+:dot_prod)
    for (int64_t i = 0; i < n; i++) {
        dot_prod += u1[i] * u2[i];
    }
    *dot_prod_ptr = dot_prod;

    return EXIT_SUCCESS;
}


SGPAR_API int sgp_vec_orthogonalize_omp(sgp_real_t* u1, sgp_real_t* u2, int64_t n) {

    sgp_real_t mult1;
    sgp_vec_dotproduct_omp(&mult1, u1, u2, n);

//#pragma omp for
    for (int64_t i = 0; i < n; i++) {
        u1[i] -= mult1 * u2[i];
    }
    return EXIT_SUCCESS;
}

SGPAR_API int sgp_vec_D_orthogonalize_omp(sgp_real_t* u1, sgp_real_t* u2,
    sgp_wgt_t* D, int64_t n) {

    //u1[i] = u1[i] - (dot(u1, D*u2)/dot(u2, D*u2)) * u2[i]

    sgp_real_t mult1;
    sgp_vec_dotproduct_omp(&mult1, u1, u2, n);

    static sgp_real_t mult_numer = 0;
    static sgp_real_t mult_denom = 0;

//#pragma omp single
    {
        mult_numer = 0;
        mult_denom = 0;
    }

//#pragma omp for reduction(+:mult_numer, mult_denom)
    for (int64_t i = 0; i < n; i++) {
        mult_numer += u1[i] * D[i] * u2[i];
        mult_denom += u2[i] * D[i] * u2[i];
    }

//#pragma omp for
    for (int64_t i = 0; i < n; i++) {
        u1[i] -= mult_numer * u2[i] / mult_denom;
    }

    return EXIT_SUCCESS;
}

SGPAR_API void sgp_power_iter_eigenvalue_log(sgp_real_t* u, sgp_graph_t g) {
    sgp_real_t eigenval = 0;
    sgp_real_t eigenval_max = 0;
    sgp_real_t eigenval_min = 2;
    for (sgp_vid_t i = 0; i < (g.nvertices); i++) {
        sgp_vid_t weighted_degree = g.source_offsets[i + 1] - g.source_offsets[i];
        sgp_real_t u_i = weighted_degree * u[i];
        sgp_real_t matvec_i = 0;
        for (sgp_eid_t j = g.source_offsets[i];
            j < g.source_offsets[i + 1]; j++) {
            matvec_i += u[g.destination_indices[j]];
        }
        u_i -= matvec_i;
        sgp_real_t eigenval_est = u_i / u[i];
        if (eigenval_est < eigenval_min) {
            eigenval_min = eigenval_est;
        }
        if (eigenval_est > eigenval_max) {
            eigenval_max = eigenval_est;
        }
        eigenval += (u_i * u_i) * 1e9;
    }

    printf("eigenvalue = %1.9lf (%1.9lf %1.9lf), "
        "edge cut lb %5.0lf "
        "gap ratio %.0lf\n",
        eigenval * 1e-9,
        eigenval_min, eigenval_max,
        eigenval * 1e-9 * (g.nvertices) / 4,
        ceil(1.0 / (1.0 - eigenval * 1e-9)));
}

SGPAR_API int sgp_power_iter(sgp_real_t* u, sgp_graph_t g, int normLap, int final
#ifdef EXPERIMENT
    , ExperimentLoggerUtil& experiment
#endif
                            ) {

    sgp_vid_t n = g.nvertices;

    sgp_real_t* vec1 = (sgp_real_t*)malloc(n * sizeof(sgp_real_t));
    SGPAR_ASSERT(vec1 != NULL);

    sgp_wgt_t* weighted_degree;

    if (normLap && final) {
        weighted_degree = (sgp_wgt_t*)malloc(n * sizeof(sgp_wgt_t));
        assert(weighted_degree != NULL);
        for (sgp_vid_t i = 0; i < n; i++) {
            weighted_degree[i] = g.source_offsets[i + 1] - g.source_offsets[i];
        }
    }

    sgp_wgt_t gb;
    if (!normLap) {
        if (!final) {
            gb = 2 * g.weighted_degree[0];
            for (sgp_vid_t i = 1; i < n; i++) {
                if (gb < 2 * g.weighted_degree[i]) {
                    gb = 2 * g.weighted_degree[i];
                }
            }
        }
        else {
            gb = 2 * (g.source_offsets[1] - g.source_offsets[0]);
            for (sgp_vid_t i = 1; i < n; i++) {
                if (gb < 2 * (g.source_offsets[i + 1] - g.source_offsets[i])) {
                    gb = 2 * (g.source_offsets[i + 1] - g.source_offsets[i]);
                }
            }
        }
    }

    uint64_t niter = 0;
    uint64_t iter_max = (uint64_t)SGPAR_POWERITER_ITER / (uint64_t)n;

//#pragma omp parallel shared(u)
  //  {

//#pragma omp for
        for (sgp_vid_t i = 0; i < n; i++) {
            vec1[i] = 1.0;
        }

#if 0
        sgp_real_t mult = 0;
        sgp_vec_dotproduct_omp(&mult, vec1, u, n);

#pragma omp for
        for (sgp_vid_t i = 0; i < n; i++) {
            u[i] -= mult * u[i] / n;
        }
        sgp_vec_normalize_omp(u, n);
#endif

        sgp_vec_normalize_omp(vec1, n);
        if (!normLap) {
            sgp_vec_orthogonalize_omp(u, vec1, n);
        }
        else {
            sgp_vec_D_orthogonalize_omp(u, vec1, g.weighted_degree, n);
        }
        sgp_vec_normalize_omp(u, n);

        sgp_real_t* v = (sgp_real_t*)malloc(n * sizeof(sgp_real_t));
        SGPAR_ASSERT(v != NULL);

//#pragma omp for
        for (sgp_vid_t i = 0; i < n; i++) {
            v[i] = u[i];
        }

        sgp_real_t tol = SGPAR_POWERITER_TOL;
        sgp_real_t dotprod = 0, lastDotprod = 1;
        while (fabs(dotprod - lastDotprod) > tol && (niter < iter_max)) {

            // u = v
//#pragma omp for
            for (sgp_vid_t i = 0; i < n; i++) {
                u[i] = v[i];
            }

            // v = Lu
//#pragma omp for
            for (sgp_vid_t i = 0; i < n; i++) {
                // sgp_real_t v_i = g.weighted_degree[i]*u[i];
                sgp_real_t weighted_degree_inv, v_i;
                if (!normLap) {
                    if (!final) {
                        v_i = (gb - g.weighted_degree[i]) * u[i];
                    }
                    else {
                        sgp_vid_t weighted_degree = g.source_offsets[i + 1] - g.source_offsets[i];
                        v_i = (gb - weighted_degree) * u[i];
                    }
                }
                else {
                    weighted_degree_inv = 1.0 / g.weighted_degree[i];
                    v_i = 0.5 * u[i];
                }
                sgp_real_t matvec_i = 0;
                for (sgp_eid_t j = g.source_offsets[i];
                    j < g.source_offsets[i + 1]; j++) {
                    if (!final) {
                        matvec_i += u[g.destination_indices[j]] * g.eweights[j];
                    }
                    else {
                        matvec_i += u[g.destination_indices[j]];
                    }
                }
                // v_i -= matvec_i;
                if (!normLap) {
                    v_i += matvec_i;
                }
                else {
                    v_i += 0.5 * matvec_i * weighted_degree_inv;
                }
                v[i] = v_i;
            }

            if (!normLap) {
                sgp_vec_orthogonalize_omp(v, vec1, n);
            }
            sgp_vec_normalize_omp(v, n);
            lastDotprod = dotprod;
            sgp_vec_dotproduct_omp(&dotprod, u, v, n);
            niter++;
        }

        if (omp_get_thread_num() == 0) {
            if (niter == iter_max) {
                printf("exceeded max iter count, ");
            }
            printf("number of iterations: %d\n", niter);
        }
        free(v);
    //}

    int max_iter_reached = 0;
    if (niter >= iter_max) {
        printf("exceeded max iter count, ");
        max_iter_reached = 1;
    }
    printf("number of iterations: %lu\n", niter);
#ifdef EXPERIMENT
        experiment.addCoarseLevel(niter, max_iter_reached, 0);
#endif
    if (!normLap && final) {
        sgp_power_iter_eigenvalue_log(u, g);
    }

    free(vec1);
    if (normLap && final) {
        free(weighted_degree);
    }
    return EXIT_SUCCESS;
}
#endif

SGPAR_API int sgp_compute_partition(sgp_vid_t *part, sgp_vid_t num_partitions, 
                                    long *edgecut, int perc_imbalance_allowed, 
                                    int local_search_alg,
                                    sgp_real_t *evec,
                                    sgp_graph_t g) {

    sgp_vid_t n = g.nvertices;
    sgp_vv_pair_t *vu_pair;
    vu_pair = (sgp_vv_pair_t *) malloc(n * sizeof(sgp_vv_pair_t));
    assert(vu_pair != NULL);

    for (sgp_vid_t i = 0; i<n; i++) {
        vu_pair[i].ev = evec[i];
        vu_pair[i].u  = i;
    }   
#ifdef __cplusplus
#ifdef USE_GNU_PARALLELMODE
    __gnu_parallel::sort(((sgp_vv_pair_t *) vu_pair), 
                         ((sgp_vv_pair_t *) vu_pair)+n,
                         vu_cmpfn_inc,
                        __gnu_parallel::quicksort_tag());
#else
    std::sort(((sgp_vv_pair_t *) vu_pair), 
              ((sgp_vv_pair_t *) vu_pair)+n,
              vu_cmpfn_inc);
#endif
#else
    qsort(vu_pair, n, sizeof(sgp_vv_pair_t), vu_cmpfn_inc);
#endif

    // Currently support only bipartitioning
    assert(num_partitions == 2);

    long max_part_size = ceil(n/((double) num_partitions));

    // allow some imbalance
    sgp_vid_t imbr = floor(max_part_size*(1.0 + perc_imbalance_allowed/100.0));
    sgp_vid_t imbl = n - imbr;
    for (sgp_vid_t i=0; i<imbl; i++) {
        part[vu_pair[i].u] = 0;
    }
    for (sgp_vid_t i=imbl; i<n; i++) {
        part[vu_pair[i].u] = 1;
    }

    long edgecut_curr = 0;
    for (sgp_vid_t i=0; i<n; i++) {
        sgp_vid_t part_i = part[i];
        for (sgp_eid_t j=g.source_offsets[i]; j<g.source_offsets[i+1]; j++) {
            sgp_vid_t v = g.destination_indices[j];
            if (part[v] != part_i) {
                edgecut_curr++;
            }
        }
    }

    long edgecut_min = edgecut_curr;
    long curr_split = imbl;

    for (sgp_vid_t i=imbl; i<imbr; i++) {
        /* add vert at position i to comm 0 */
        sgp_vid_t u = vu_pair[i].u;
        long ec_update = 0;
        for (sgp_eid_t j=g.source_offsets[u]; j<g.source_offsets[u+1]; j++) {
            sgp_vid_t v = g.destination_indices[j];
            if (part[v] == 1) {
                ec_update++;
            } else {
                ec_update--;
            }
        }
        edgecut_curr = edgecut_curr + 2*ec_update;

        if (edgecut_curr <= edgecut_min) {
            part[u] = 0;
            edgecut_min = edgecut_curr;
            curr_split = i+1;
            /*
            curr_split = n - i - 1;
            if ((n - i - 1) < (i+1)) {
                curr_split = i + 1;
            }
            */
        } 
    }

    free(vu_pair);

    printf("After bipartitioning, the partitions have %ld and %ld vertices, "
           "and the edge cut is %ld.\n", 
           curr_split, n-curr_split, edgecut_min/2);
    *edgecut = edgecut_min/2;

    if (local_search_alg == 0) 
        return 0;

    int64_t n_left = curr_split-n/3;
    if (n_left < 0) {
        n_left = 0;
    }
    int64_t n_right = curr_split+n/3;
    if (n_right > ((int64_t) n)) {
        n_right = n;
    }
    int num_swaps = 0;
    int64_t ec_change = 0;
    int max_swaps = n/3;
    while (num_swaps < max_swaps) {
        int64_t ec_dec_max = 0;
        sgp_vid_t move_right_vert = SGP_INFTY;
        for (long i=n_left; i<curr_split; i++) {
            sgp_vid_t part_i = part[i];
            
            int64_t ec_dec_i = 0;
            for (sgp_eid_t j=g.source_offsets[i]; 
                           j<g.source_offsets[i+1]; j++) {
                sgp_vid_t v = g.destination_indices[j];
                if (part[v] != part_i) {
                    ec_dec_i++;
                } else {
                    ec_dec_i--;
                }
            }
            if (ec_dec_i > ec_dec_max) {
                ec_dec_max = ec_dec_i;
                move_right_vert = i; 
            }
        }
        if (move_right_vert == SGP_INFTY) {
            // printf("Exiting before swap\n");
            break;
        }
        // printf("ec dec is %ld, moving vert %lu to the right\n",
        //                ec_dec_max, (uint64_t) move_right_vert);
        if (part[move_right_vert] == 0) {
            part[move_right_vert] = 1;
        } else {
            part[move_right_vert] = 0;
        }
        ec_change += ec_dec_max;
        int64_t ec_dec_max_prev = ec_dec_max;

        ec_dec_max = 0;
        sgp_vid_t move_left_vert = SGP_INFTY;
        for (long i=curr_split; i<n_right; i++) {
            sgp_vid_t part_i = part[i];
            int64_t ec_dec_i = 0;
            for (sgp_eid_t j=g.source_offsets[i]; 
                           j<g.source_offsets[i+1]; j++) {
                sgp_vid_t v = g.destination_indices[j];
                if (part[v] != part_i) {
                    ec_dec_i++;
                } else {
                    ec_dec_i--;
                }
            }
            if (ec_dec_i > ec_dec_max) {
                ec_dec_max = ec_dec_i;
                move_left_vert = i; 
            }
        }
        if (move_left_vert == SGP_INFTY) {
            /* Roll back prev swap and exit */
            if (part[move_right_vert] == 0) {
                part[move_right_vert] = 1;
            } else {
                part[move_right_vert] = 0;
            }
            ec_change -= ec_dec_max_prev;
            // printf("Incomplete swap, exiting\n");
            break;
        }
        // printf("ec dec is %ld, moving vert %lu to the left\n",
        //                ec_dec_max, (uint64_t) move_left_vert);
        if (part[move_left_vert] == 0) {
            part[move_left_vert] = 1;
        } else {
            part[move_left_vert] = 0;
        }
        ec_change += ec_dec_max;

        num_swaps++;
    }
    printf("Total change: %ld, swaps %d, new edgecut %ld\n", 
                    ec_change, num_swaps, edgecut_min/2-ec_change);

    edgecut_curr = 0;
    for (sgp_vid_t i=0; i<n; i++) {
        sgp_vid_t part_i = part[i];
        for (sgp_eid_t j=g.source_offsets[i]; j<g.source_offsets[i+1]; j++) {
            sgp_vid_t v = g.destination_indices[j];
            if (part[v] != part_i) {
                edgecut_curr++;
            }
        }
    }
    // fprintf(stderr, "computed %ld, est %ld\n", edgecut_curr/2,
    //                    edgecut_min/2-ec_change);
    assert(edgecut_curr/2 == (edgecut_min/2-ec_change));
    *edgecut = edgecut_curr/2;

#if 0
    sgp_real_t bin_width = 0.005;
    int64_t num_bins = ((int64_t) (2.0/bin_width) + 1);
    int64_t *bin_counts = (int64_t *) malloc(num_bins * sizeof(int64_t));
    for (int64_t i=0; i<num_bins; i++) {
        bin_counts[i] = 0; 
    }
    for (int64_t i=0; i<((int64_t)n); i++) {
        int64_t bin_num = ((int64_t) floor((1+evec[i])/bin_width));
        bin_counts[bin_num]++;
    }

    int64_t cumulative_bin_perc = 0;
    for (int64_t i=0; i<num_bins; i++) {
        int64_t bin_perc = (100*bin_counts[i])/n;
        if (bin_perc > 0) {
            cumulative_bin_perc += bin_perc;
            printf("bin %ld, perc %ld\n", i, bin_perc);
        }
    }
    printf("cumulative bin percentage: %ld\n", cumulative_bin_perc);

    free(bin_counts);
#endif

    return EXIT_SUCCESS;
}    

SGPAR_API int sgp_improve_partition(sgp_vid_t *part, sgp_vid_t num_partitions, 
                                    long *edgecut, int perc_imbalance_allowed, 
                                    sgp_real_t *evec,
                                    sgp_graph_t g) {

    return EXIT_SUCCESS;
}    


/**********************************************************
 * API
 **********************************************************
 */

SGPAR_API int sgp_load_graph(sgp_graph_t *g, char *csr_filename);
SGPAR_API int sgp_free_graph(sgp_graph_t *g);
SGPAR_API int sgp_load_partition(sgp_vid_t *part, int size, char *part_filename);
SGPAR_API int compute_partition_edit_distance(const sgp_vid_t* part1, const sgp_vid_t* part2, int size, unsigned int *diff);
SGPAR_API int sgp_partition_graph(sgp_vid_t *part, 
                                  const sgp_vid_t num_partitions,
                                  long *edge_cut,
                                  const int coarsening_alg, 
                                  const int refine_alg,
                                  const int local_search_alg, 
                                  const int perc_imbalance_allowed,
                                  const sgp_graph_t g,
#ifdef EXPERIMENT
    ExperimentLoggerUtil& experiment,
#endif
                                  const sgp_vid_t * best_part,
                                  const int compare_part,
                                  sgp_pcg32_random_t* rng);


#ifdef __cplusplus
}
#endif
#endif // SGPAR_H_

/**********************************************************
 * Implementation 
 **********************************************************
 */
#ifdef SGPAR_IMPLEMENTATION

#ifdef __cplusplus
namespace sgpar {
#endif

SGPAR_API int sgp_load_graph(sgp_graph_t *g, char *csr_filename) {

    FILE *infp = fopen(csr_filename, "rb");
    if (infp == NULL) {
        printf("Error: Could not open input file. Exiting ...\n");
        return EXIT_FAILURE;
    }
    long n, m;
    long unused_vals[4];
    fread(&n, sizeof(long), 1, infp);
    fread(&m, sizeof(long), 1, infp);
    fread(unused_vals, sizeof(long), 4, infp);
    g->nvertices = n;
    g->nedges = m/2;
    g->source_offsets = (sgp_eid_t *) malloc((g->nvertices+1)*sizeof(sgp_eid_t));
    SGPAR_ASSERT(g->source_offsets != NULL);
    g->destination_indices = (sgp_vid_t *) malloc(2*g->nedges*sizeof(sgp_vid_t));
    SGPAR_ASSERT(g->destination_indices != NULL);
    size_t nitems_read = fread(g->source_offsets, sizeof(sgp_eid_t), g->nvertices+1, infp);
    SGPAR_ASSERT(nitems_read == ((size_t) g->nvertices+1));
    nitems_read = fread(g->destination_indices, sizeof(sgp_vid_t), 2*g->nedges, infp);
    SGPAR_ASSERT(nitems_read == ((size_t) 2*g->nedges));
    CHECK_RETSTAT( fclose(infp) );
    g->eweights = NULL;
    g->weighted_degree = NULL;
    return EXIT_SUCCESS;
}

SGPAR_API int sgp_free_graph(sgp_graph_t *g) {

    if (g->source_offsets != NULL) {
        free(g->source_offsets);
        g->source_offsets = NULL;
    }
    
    if (g->destination_indices != NULL) {
        free(g->destination_indices);
        g->destination_indices = NULL;
    }

    if (g->eweights != NULL) {
        free(g->eweights);
        g->eweights = NULL;
    }

    if (g->weighted_degree != NULL) {
        free(g->weighted_degree);
        g->weighted_degree = NULL;
    }

    return EXIT_SUCCESS;
}

SGPAR_API int sgp_load_partition(sgp_vid_t *part, int size, char *part_filename){
    FILE *infp = fopen(part_filename, "rb");
    if (infp == NULL) {
        printf("Error: Could not open partition file %s. Exiting ...\n", part_filename);
        return EXIT_FAILURE;
    }

    for(int i = 0; i < size; i++){
        if(fscanf(infp, "%i", part + i) == 0){
            return EXIT_FAILURE;
        }
    }

    return EXIT_SUCCESS;
}

//partitions assumed to same vertex labellings
SGPAR_API int compute_partition_edit_distance(const sgp_vid_t* part1, const sgp_vid_t* part2, int size, unsigned int *diff){

    int d = 0; //difference if partition labelling is same
    int d2 = 0; //difference if partition labelling is swapped
    for(int i = 0; i < size; i++){
        if(part1[i] != part2[i]){
            d++;
        } else {
            d2++;
        }
    }

    if(d < d2){
        *diff = d/2;
    } else {
        *diff = d2/2;
    }
    return EXIT_SUCCESS;
}

SGPAR_API int sgp_partition_graph(sgp_vid_t *part, 
                                  const sgp_vid_t num_partitions,
                                  long *edge_cut,
                                  const int coarsening_alg, 
                                  const int refine_alg,
                                  const int local_search_alg, 
                                  const int perc_imbalance_allowed,
                                  const sgp_graph_t g,
#ifdef EXPERIMENT
                                  ExperimentLoggerUtil& experiment,
#endif
                                  const sgp_vid_t * best_part,
                                  const int compare_part,
                                  sgp_pcg32_random_t* rng) {

    printf("sgpar settings: %d %lu %.16f\n", 
                    SGPAR_COARSENING_VTX_CUTOFF, 
                    (uint64_t) SGPAR_POWERITER_ITER,
                    SGPAR_POWERITER_TOL);

    int coarsening_level = 0;
    sgp_graph_t g_all[SGPAR_COARSENING_MAXLEVELS];
    sgp_vid_t *vcmap[SGPAR_COARSENING_MAXLEVELS];
    
    for (int i=0; i<SGPAR_COARSENING_MAXLEVELS; i++) {
        g_all[i].nvertices = 0;
        g_all[i].source_offsets = NULL;
        g_all[i].destination_indices = NULL;
        g_all[i].eweights = NULL;
    }
    g_all[0].nvertices = g.nvertices; g_all[0].nedges = g.nedges;
    g_all[0].source_offsets = g.source_offsets;
    g_all[0].destination_indices = g.destination_indices;

    double start_time = sgp_timer();
    double coarsening_sort_time = 0;

    sgp_pcg32_random_t coarsen_rng;
    coarsen_rng.state = 0xfedcba9876543210;
    coarsen_rng.inc = 1;

    //generate all coarse graphs
    while ((coarsening_level < (SGPAR_COARSENING_MAXLEVELS-1)) && 
           (g_all[coarsening_level].nvertices > SGPAR_COARSENING_VTX_CUTOFF) &&
           (coarsening_alg != 5)) {
        coarsening_level++;
        vcmap[coarsening_level-1] = (sgp_vid_t *) 
                                    malloc(g_all[coarsening_level-1].nvertices
                                                 * sizeof(sgp_vid_t));
        SGPAR_ASSERT(vcmap[coarsening_level-1] != NULL);
        CHECK_SGPAR( sgp_coarsen_one_level(&g_all[coarsening_level],
                                            vcmap[coarsening_level-1],
                                            g_all[coarsening_level-1], 
                                            coarsening_level, coarsening_alg, 
                                            rng, &coarsening_sort_time) );

        printf("Coarsening graph at level %d\n", coarsening_level);
    }

    printf("Coarsest level: %d\n", coarsening_level);

    int num_coarsening_levels = coarsening_level+1;

    double fin_coarsening_time = sgp_timer();

    sgp_vid_t gc_nvertices = g_all[num_coarsening_levels-1].nvertices;
    sgp_real_t *eigenvec[SGPAR_COARSENING_MAXLEVELS];
    eigenvec[num_coarsening_levels-1] = (sgp_real_t *) 
                                        malloc(gc_nvertices*sizeof(sgp_real_t));
    SGPAR_ASSERT(eigenvec[num_coarsening_levels-1] != NULL);
    //randomly initialize guess eigenvector for coarsest graph
    for (sgp_vid_t i=0; i<gc_nvertices; i++) {
        eigenvec[num_coarsening_levels-1][i] = 
                            ((double) sgp_pcg32_random_r(rng))/UINT32_MAX;
    }

    sgp_vec_normalize(eigenvec[num_coarsening_levels-1], gc_nvertices);
    if (coarsening_alg != 5) { /* coarsening_alg = 5 is no coarsening */
        printf("Coarsening level %d, ", num_coarsening_levels-1);        
        if (refine_alg == 0) {
            CHECK_SGPAR( sgp_power_iter(eigenvec[num_coarsening_levels-1], 
                g_all[num_coarsening_levels-1], 0, 0
#ifdef EXPERIMENT
                , experiment 
#endif
            ) );
        } else {
            CHECK_SGPAR( sgp_power_iter(eigenvec[num_coarsening_levels-1], 
                   g_all[num_coarsening_levels-1], 1, 0
#ifdef EXPERIMENT
                , experiment
#endif
            ) );
        }
    }

#ifdef COARSE_EIGEN_EC
    sgp_real_t* prolonged_eigenvec[SGPAR_COARSENING_MAXLEVELS];
    for (int l=num_coarsening_levels-2; l>=0; l--) {
        sgp_vid_t gcl_n = g_all[l].nvertices;
        eigenvec[l] = (sgp_real_t *) malloc(gcl_n*sizeof(sgp_real_t));
        SGPAR_ASSERT(eigenvec[l] != NULL);

        prolonged_eigenvec[l + 1] = (sgp_real_t*)malloc(gcl_n * sizeof(sgp_real_t));
        SGPAR_ASSERT(prolonged_eigenvec[l+1] != NULL);
        //prolong eigenvector from coarser level to finer level
        for (sgp_vid_t i=0; i<gcl_n; i++) {
            eigenvec[l][i] = eigenvec[l+1][vcmap[l][i]];
            prolonged_eigenvec[l + 1][i] = eigenvec[l][i];
        }
        free(eigenvec[l+1]);

        //prolong l+1 eigenvector to finest level
        for (int l2 = l - 1; l2 >= 0; l2--) {
            sgp_real_t* prev_prolonged = prolonged_eigenvec[l + 1];
            sgp_vid_t gcl2_n = g_all[l2].nvertices;
            prolonged_eigenvec[l + 1] = (sgp_real_t*)malloc(gcl2_n * sizeof(sgp_real_t));
            SGPAR_ASSERT(prolonged_eigenvec[l + 1] != NULL);
            for (sgp_vid_t i = 0; i < gcl2_n; i++) {
                prolonged_eigenvec[l + 1][i] = prev_prolonged[vcmap[l2][i]];
            }
            free(prev_prolonged);
        }

        free(vcmap[l]);

        sgp_vec_normalize(eigenvec[l], gcl_n);
        if (l > 0) {
            printf("Coarsening level %d, ", l);
            if (refine_alg == 0) {
                sgp_power_iter(eigenvec[l], g_all[l], 0, 0
#ifdef EXPERIMENT
                    , experiment
#endif
                );
            } else {
                sgp_power_iter(eigenvec[l], g_all[l], 1, 0
#ifdef EXPERIMENT
                    , experiment
#endif
                );
            }
        }
    }
#else
    for (int l = num_coarsening_levels - 2; l >= 0; l--) {
        sgp_vid_t gcl_n = g_all[l].nvertices;
        eigenvec[l] = (sgp_real_t*)malloc(gcl_n * sizeof(sgp_real_t));
        SGPAR_ASSERT(eigenvec[l] != NULL);

        sgp_real_t * prolonged_eigenvec = (sgp_real_t*)malloc(gcl_n * sizeof(sgp_real_t));
        SGPAR_ASSERT(prolonged_eigenvec != NULL);
        //prolong eigenvector from coarser level to finer level
        for (sgp_vid_t i = 0; i < gcl_n; i++) {
            eigenvec[l][i] = eigenvec[l + 1][vcmap[l][i]];
            prolonged_eigenvec[i] = eigenvec[l][i];
        }
        free(eigenvec[l + 1]);
        free(vcmap[l]);

        sgp_vec_normalize(eigenvec[l], gcl_n);

        //don't do refinement for finest level here
        if (l > 0) {
            printf("Coarsening level %d, ", l);
            if (refine_alg == 0) {
                sgp_power_iter(eigenvec[l], g_all[l], 0, 0
#ifdef EXPERIMENT
                    , experiment
#endif
                );
            }
            else {
                sgp_power_iter(eigenvec[l], g_all[l], 1, 0
#ifdef EXPERIMENT
                    , experiment
#endif
                );
            }
        }
    }
#endif

    double fin_refine_time = sgp_timer();

    printf("Coarsening level %d, ", 0);
    if (refine_alg == 0) {
        CHECK_SGPAR( sgp_power_iter(eigenvec[0], g_all[0], 0, 1
#ifdef EXPERIMENT
            , experiment
#endif
        ) );
    } else {
        CHECK_SGPAR( sgp_power_iter(eigenvec[0], g_all[0], 1, 1
#ifdef EXPERIMENT
            , experiment
#endif
        ));
    }
    double fin_final_level_time = sgp_timer();

    printf("Total: %3.3lf s, coarsening %3.3lf %3.0lf%% "
                    "(sort %3.3lf %3.0lf%%), "
                    "refine %3.3lf s (%3.3lf s, %3.0lf%% + %3.3lf, %3.0lf%%)\n", 
                    fin_final_level_time-start_time,
                    fin_coarsening_time-start_time,
                    (fin_coarsening_time-start_time)*100/(fin_final_level_time-start_time),
                    coarsening_sort_time,
                    coarsening_sort_time*100/(fin_final_level_time-start_time),
                    fin_final_level_time-fin_coarsening_time,
                    fin_final_level_time-fin_refine_time,
                    100*(fin_final_level_time-fin_refine_time)/(fin_final_level_time-start_time),
                    fin_refine_time-fin_coarsening_time,
                    100*(fin_refine_time-fin_coarsening_time)/
                    (fin_final_level_time-start_time));

#ifdef EXPERIMENT
    experiment.setTotalDurationSeconds(fin_final_level_time - start_time);
    experiment.setCoarsenDurationSeconds(fin_coarsening_time - start_time);
    experiment.setRefineDurationSeconds(fin_refine_time - fin_coarsening_time);
    experiment.setCoarsenSortDurationSeconds(coarsening_sort_time);
#endif


    for (int i=1; i<num_coarsening_levels; i++) {
        sgp_free_graph(&g_all[i]);
    }

#ifdef COARSE_EIGEN_EC
        for (int l = num_coarsening_levels - 1; l >= 1; l--) {
            printf("Computing edge cut for eigenvector prolonged from coarse level %d\n", l);
            sgp_compute_partition(part, num_partitions, edge_cut,
                perc_imbalance_allowed,
                local_search_alg,
                prolonged_eigenvec[l], g);

            free(prolonged_eigenvec[l]);
            
            unsigned int part_diff = 0;
            if (compare_part) {
                CHECK_SGPAR(compute_partition_edit_distance(part, best_part, g.nvertices, &part_diff));
            }

            //fprintf(metricfp, " %lu %lu", *edge_cut, part_diff);
        }
#endif

    

    sgp_compute_partition(part, num_partitions, edge_cut,
        perc_imbalance_allowed,
        local_search_alg,
        eigenvec[0], g);

    unsigned int part_diff = 0;
    if (compare_part) {
        CHECK_SGPAR(compute_partition_edit_distance(part, best_part, g.nvertices, &part_diff));
    }

#ifdef EXPERIMENT
        experiment.setFinestEdgeCut(*edge_cut);
        experiment.setPartitionDiff(part_diff);
#endif

    sgp_improve_partition(part, num_partitions, edge_cut,
                           perc_imbalance_allowed,
                           eigenvec[0], g);
    free(eigenvec[0]);

    return EXIT_SUCCESS;
}

#ifdef __cplusplus
}
#endif // namespace sgpar

#endif // SGPAR_IMPLEMENTATION
