#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <list>

#ifdef _OPENMP
#include <omp.h>
#else
#include <sys/time.h>
#endif

#include <Kokkos_Core.hpp>
#include <Kokkos_Atomic.hpp>
#include <Kokkos_UnorderedMap.hpp>
#include "KokkosSparse_CrsMatrix.hpp"
#include "KokkosSparse_spmv.hpp"
#include "KokkosSparse_spgemm.hpp"
#include "KokkosGraph_Distance2Color.hpp"
#include "KokkosKernels_SparseUtils.hpp"

#include <unordered_map>
// #define USE_GNU_PARALLELMODE
#ifdef USE_GNU_PARALLELMODE
#include <parallel/algorithm> // for parallel sort
#else 
#include <algorithm>          // for STL sort
#endif

#ifdef EXPERIMENT
#include "ExperimentLoggerUtil.cpp"
#endif

#include "definitions_kokkos.h"

namespace sgpar {
namespace sgpar_kokkos {

SGPAR_API int sgp_coarsen_mis_2(matrix_type& interp,
    sgp_vid_t* nvertices_coarse_ptr,
    const matrix_type& g,
    const int coarsening_level,
    sgp_pcg32_random_t* rng,
    double* time_ptrs) {

    sgp_vid_t n = g.numRows();
    typedef KokkosKernels::Experimental::KokkosKernelsHandle
        <sgp_eid_t, sgp_vid_t, sgp_wgt_t,
        typename Device::execution_space, typename Device::memory_space, typename Device::memory_space > KernelHandle;

    KernelHandle kh;
    kh.set_team_work_size(16);
    kh.set_dynamic_scheduling(true);

    kh.create_distance2_graph_coloring_handle();
    KokkosGraph::Experimental::graph_color_distance2(&kh, n, g.graph.row_map, g.graph.entries);
    Kokkos::View<sgp_vid_t*> colors = kh.get_distance2_graph_coloring_handle()->get_vertex_colors();
    kh.destroy_distance2_graph_coloring_handle();

    Kokkos::View<sgp_vid_t> nvc("nvertices_coarse");
    Kokkos::View<sgp_vid_t*> vcmap("vcmap", n);
    
    sgp_vid_t first_color = 1;

    //create aggregates for color 1
    Kokkos::parallel_for(n, KOKKOS_LAMBDA(sgp_vid_t i){
        if (colors(i) == first_color) {
            vcmap(i) = Kokkos::atomic_fetch_add(&nvc(), 1);
        }
        else {
            vcmap(i) = SGP_INFTY;
        }
    });

    //add direct neighbors of color 1 to aggregates
    //could also do this by checking neighbors of each un-aggregated vertex
    Kokkos::parallel_for(n, KOKKOS_LAMBDA(sgp_vid_t i){
        if (colors(i) == first_color) {
            //could use a thread team here
            for (sgp_eid_t j = g.graph.row_map(i); j < g.graph.row_map(i + 1); j++) {
                sgp_vid_t v = g.graph.entries(j);
                vcmap(v) = vcmap(i);
            }
        }
    });

    //add distance-2 neighbors of color 1 to arbitrary neighboring aggregate
    Kokkos::parallel_for(n, KOKKOS_LAMBDA(sgp_vid_t i){
        if (vcmap(i) != SGP_INFTY) {
            //could use a thread team here
            for (sgp_eid_t j = g.graph.row_map(i); j < g.graph.row_map(i + 1); j++) {
                sgp_vid_t v = g.graph.entries(j);
                if (vcmap(v) == SGP_INFTY) {
                    vcmap(v) = vcmap(i);
                }
            }
        }
    });

    //create singleton aggregates of remaining unaggregated vertices
    Kokkos::parallel_for(n, KOKKOS_LAMBDA(sgp_vid_t i){
        if (vcmap(i) == SGP_INFTY) {
            vcmap(i) = Kokkos::atomic_fetch_add(&nvc(), 1);
        }
    });

    sgp_vid_t nc = 0;
    Kokkos::deep_copy(nc, nvc);
    *nvertices_coarse_ptr = nc;

    edge_view_t row_map("interpolate row map", n + 1);

    Kokkos::parallel_for(n + 1, KOKKOS_LAMBDA(sgp_vid_t u){
        row_map(u) = u;
    });

    vtx_view_t entries("interpolate entries", n);
    wgt_view_t values("interpolate values", n);
    //compute the interpolation weights
    Kokkos::parallel_for(n, KOKKOS_LAMBDA(sgp_vid_t u){
        entries(u) = vcmap(u);
        values(u) = 1.0;
    });

    graph_type graph(entries, row_map);
    interp = matrix_type("interpolate", nc, values, graph);

    return EXIT_SUCCESS;
}

SGPAR_API int sgp_coarsen_HEC(matrix_type& interp,
    sgp_vid_t* nvertices_coarse_ptr,
    const matrix_type& g,
    const int coarsening_level,
    sgp_pcg32_random_t* rng,
    double* time_ptrs) {

    sgp_vid_t n = g.numRows();

    vtx_view_t vperm("permutation", n);
    vtx_mirror_t perm_m = Kokkos::create_mirror(vperm);

    vtx_view_t hn("heavies", n);

    vtx_view_t vcmap("vcmap", n);

    Kokkos::parallel_for("initialize vcmap", n, KOKKOS_LAMBDA(sgp_vid_t i) {
        vcmap(i) = SGP_INFTY;
    });

    Kokkos::parallel_for("host initialize permutation", host_policy(0, n), KOKKOS_LAMBDA(sgp_vid_t i) {
        perm_m(i) = i;
    });

    Kokkos::Timer timer;
    for (sgp_vid_t i = n - 1; i > 0; i--) {
        sgp_vid_t v_i = perm_m(i);
#ifndef SGPAR_HUGEGRAPHS
        uint32_t j = (sgp_pcg32_random_r(rng)) % (i + 1);
#else
        uint64_t j1 = (sgp_pcg32_random_r(rng)) % (i + 1);
        uint64_t j2 = (sgp_pcg32_random_r(rng)) % (i + 1);
        uint64_t j = ((j1 << 32) + j2) % (i + 1);
#endif 
        sgp_vid_t v_j = perm_m(j);
        perm_m(i) = v_j;
        perm_m(j) = v_i;
    }
    Kokkos::deep_copy(vperm, perm_m);

    vtx_view_t reverse_map("reversed", n);
    Kokkos::parallel_for("construct reverse map", n, KOKKOS_LAMBDA(sgp_vid_t i) {
        reverse_map(vperm(i)) = i;
    });
    time_ptrs[6] += timer.seconds();

    timer.reset();
    if (coarsening_level == 1) {
        uint64_t state = rng->state;
        uint64_t inc = rng->inc;
        //all weights equal at this level so choose heaviest edge randomly
        Kokkos::parallel_for("Random HN", n, KOKKOS_LAMBDA(sgp_vid_t i) {
            sgp_vid_t adj_size = g.graph.row_map(i + 1) - g.graph.row_map(i);
            sgp_pcg32_random_t copy;
            copy.state = state + i;
            copy.inc = inc;
            sgp_vid_t offset = g.graph.row_map(i) + ((sgp_pcg32_random_r(&copy)) % adj_size);
            hn(i) = g.graph.entries(offset);
        });
    }
    else {
        Kokkos::parallel_for("Heaviest HN", n, KOKKOS_LAMBDA(sgp_vid_t i) {
            sgp_vid_t hn_i = g.graph.entries(g.graph.row_map(i));
            sgp_wgt_t max_ewt = g.values(g.graph.row_map(i));

            sgp_eid_t end_offset = g.graph.row_map(i + 1);// +g.edges_per_source[i];

            for (sgp_eid_t j = g.graph.row_map(i) + 1; j < end_offset; j++) {
                if (max_ewt < g.values(j)) {
                    max_ewt = g.values(j);
                    hn_i = g.graph.entries(j);
                }

            }
            hn(i) = hn_i;
        });
    }
    time_ptrs[7] += timer.seconds();
    vtx_view_t match("match", n);
    Kokkos::parallel_for(n, KOKKOS_LAMBDA(sgp_vid_t i){
        match(i) = SGP_INFTY;
    });

    sgp_vid_t perm_length = n;

    Kokkos::View<sgp_vid_t> nvertices_coarse("nvertices");

    //construct mapping using heaviest edges
    int swap = 1;
    timer.reset();
    while (perm_length > 0) {
        vtx_view_t next_perm("next perm", perm_length);
        Kokkos::View<sgp_vid_t> next_length("next_length");
        
        Kokkos::parallel_for(perm_length, KOKKOS_LAMBDA(sgp_vid_t i){
            sgp_vid_t u = vperm(i);
            sgp_vid_t v = hn(u);
            int condition = reverse_map(u) < reverse_map(v);
            //need to enforce an ordering condition to allow hard-stall conditions to be broken
            if (condition ^ swap) {
                if (Kokkos::atomic_compare_exchange_strong(&match(u), SGP_INFTY, v)) {
                    if (Kokkos::atomic_compare_exchange_strong(&match(v), SGP_INFTY, u)) {
                        sgp_vid_t cv = Kokkos::atomic_fetch_add(&nvertices_coarse(), 1);
                        vcmap(u) = cv;
                        vcmap(v) = cv;
                    }
                    else {
                        if (vcmap(v) != SGP_INFTY) {
                            vcmap(u) = vcmap(v);
                        }
                        else {
                            match(u) = SGP_INFTY;
                        }
                    }
                }
            }
        });
        Kokkos::fence();
        //add the ones that failed to be reprocessed next round
        //maybe count these then create next_perm to save memory?
        Kokkos::parallel_for(perm_length, KOKKOS_LAMBDA(sgp_vid_t i){
            sgp_vid_t u = vperm(i);
            if (vcmap(u) == SGP_INFTY) {
                sgp_vid_t add_next = Kokkos::atomic_fetch_add(&next_length(), 1);
                next_perm(add_next) = u;
            }
        });
        Kokkos::fence();
        swap = swap ^ 1;
        Kokkos::deep_copy(perm_length, next_length);
        vperm = next_perm;
    }
    time_ptrs[8] += timer.seconds();

    sgp_vid_t nc = 0;
    Kokkos::deep_copy(nc, nvertices_coarse);
    *nvertices_coarse_ptr = nc;

    edge_view_t row_map("interpolate row map", n + 1);

    Kokkos::parallel_for(n + 1, KOKKOS_LAMBDA(sgp_vid_t u){
        row_map(u) = u;
    });

    vtx_view_t entries("interpolate entries", n);
    wgt_view_t values("interpolate values", n);
    //compute the interpolation weights
    Kokkos::parallel_for(n, KOKKOS_LAMBDA(sgp_vid_t u){
        entries(u) = vcmap(u);
        values(u) = 1.0;
    });

    graph_type graph(entries, row_map);
    interp = matrix_type("interpolate", nc, values, graph);

    return EXIT_SUCCESS;
}

//assumes that matrix has one entry-per row, not valid for general matrices
SGPAR_API int compute_transpose(const matrix_type& mtx,
    matrix_type& transpose) {
    sgp_vid_t n = mtx.numRows();
    sgp_vid_t nc = mtx.numCols();

    vtx_view_t fine_per_coarse("fine_per_coarse", nc);
    //transpose interpolation matrix
    vtx_view_t adj_transpose("adj_transpose", n);
    wgt_view_t wgt_transpose("weights_transpose", n);
    edge_view_t row_map_transpose("rows_transpose", nc + 1);

    Kokkos::parallel_for(n, KOKKOS_LAMBDA(sgp_vid_t i) {
        sgp_vid_t v = mtx.graph.entries(i);
        Kokkos::atomic_increment(&fine_per_coarse(v));
    });
    Kokkos::parallel_scan(nc, KOKKOS_LAMBDA(const sgp_vid_t i,
        sgp_vid_t & update, const bool final) {
        // Load old value in case we update it before accumulating
        const sgp_vid_t val_i = fine_per_coarse(i);
        // For inclusive scan,
        // change the update value before updating array.
        update += val_i;
        if (final) {
            row_map_transpose(i + 1) = update; // only update array on final pass
        }
    });
    Kokkos::parallel_for(nc, KOKKOS_LAMBDA(sgp_vid_t i) {
        fine_per_coarse(i) = 0;
    });
    Kokkos::parallel_for(n, KOKKOS_LAMBDA(sgp_vid_t i) {
        sgp_vid_t v = mtx.graph.entries(i);
        sgp_eid_t offset = row_map_transpose(v) + Kokkos::atomic_fetch_add(&fine_per_coarse(v), 1);
        adj_transpose(offset) = i;
        wgt_transpose(offset) = 1;
    });

    graph_type transpose_graph(adj_transpose, row_map_transpose);
    transpose = matrix_type("transpose", n, wgt_transpose, transpose_graph);

    return EXIT_SUCCESS;
}

int dump_mtx(const matrix_type& mtx, char* filename, bool symmetric) {
    vtx_mirror_t entry_m = Kokkos::create_mirror(mtx.graph.entries);
    Kokkos::deep_copy(entry_m, mtx.graph.entries);
    edge_mirror_t row_m = Kokkos::create_mirror(mtx.graph.row_map);
    Kokkos::deep_copy(row_m, mtx.graph.row_map);
    wgt_mirror_t value_m = Kokkos::create_mirror(mtx.values);
    Kokkos::deep_copy(value_m, mtx.values);

    char* filename_row = new char[strlen(filename) + 5];
    char* filename_col = new char[strlen(filename) + 5];
    char* filename_vals = new char[strlen(filename) + 6];
    char* filename_descr = new char[strlen(filename) + 7];
    strcpy(filename_row, filename);
    strcpy(filename_col, filename);
    strcpy(filename_vals, filename);
    strcpy(filename_descr, filename);
    strcat(filename_row, "_row");
    strcat(filename_col, "_col");
    strcat(filename_vals, "_vals");
    strcat(filename_descr, "_descr");
    FILE* RowFile = fopen(filename_row, "w");
    FILE* ColFile = fopen(filename_col, "w");
    FILE* ValsFile = fopen(filename_vals, "w");
    FILE* DescrFile = fopen(filename_descr, "w");

    if (symmetric) {
        fprintf(DescrFile, "%% symmetric\n");
    }
    fprintf(DescrFile, "%i %i %i\n\n", mtx.numRows(), mtx.numCols(), mtx.nnz());
    fclose(DescrFile);
    free(filename_descr);

    fwrite(row_m.data(), sizeof(sgp_eid_t), mtx.numRows() + 1, RowFile);
    fwrite(entry_m.data(), sizeof(sgp_vid_t), mtx.nnz(), ColFile);
    fwrite(value_m.data(), sizeof(sgp_wgt_t), mtx.nnz(), ValsFile);
    fclose(RowFile);
    fclose(ColFile);
    fclose(ValsFile);
    free(filename_row);
    free(filename_col);
    free(filename_vals);

    return EXIT_SUCCESS;
}

SGPAR_API int sgp_build_coarse_graph_spgemm(matrix_type& gc,
    const matrix_type& interp_mtx,
    const matrix_type& g,
    const int coarsening_level,
    double* time_ptrs) {

    sgp_vid_t n = g.numRows();
    sgp_vid_t nc = interp_mtx.numCols();

    matrix_type interp_transpose;// = KokkosKernels::Impl::transpose_matrix(interp_mtx);
    compute_transpose(interp_mtx, interp_transpose);

    //dump_mtx(g, "dump/g_dump", true);
    //dump_mtx(interp_mtx, "dump/interp_dump", false);
    //dump_mtx(interp_transpose, "dump/interp_transpose_dump", false);
    typedef KokkosKernels::Experimental::KokkosKernelsHandle
        <sgp_eid_t, sgp_vid_t, sgp_wgt_t,
        typename Device::execution_space, typename Device::memory_space, typename Device::memory_space > KernelHandle;

    KernelHandle kh;
    kh.set_team_work_size(16);
    kh.set_dynamic_scheduling(true);

    // Select an spgemm algorithm, limited by configuration at compile-time and set via the handle
    // Some options: {SPGEMM_KK_MEMORY, SPGEMM_KK_SPEED, SPGEMM_KK_MEMSPEED, /*SPGEMM_CUSPARSE, */ SPGEMM_MKL}
    KokkosSparse::SPGEMMAlgorithm spgemm_algorithm = KokkosSparse::SPGEMM_KK_MEMORY;
    kh.create_spgemm_handle(spgemm_algorithm);

#ifdef TRANSPOSE_FIRST
    Kokkos::View<sgp_eid_t*> row_map_p1("rows_partial", nc + 1);
    KokkosSparse::Experimental::spgemm_symbolic(
        &kh,
        nc,
        n,
        n,
        interp_transpose.graph.row_map,
        interp_transpose.graph.entries,
        false,
        g.graph.row_map,
        g.graph.entries,
        false,
        row_map_p1
        );

    //partial-result matrix
    Kokkos::View<sgp_vid_t*> entries_p1("adjacencies_partial", kh.get_spgemm_handle()->get_c_nnz());
    Kokkos::View<sgp_wgt_t*> values_p1("weights_partial", kh.get_spgemm_handle()->get_c_nnz());

    KokkosSparse::Experimental::spgemm_numeric(
        &kh,
        nc,
        n,
        n,
        interp_transpose.graph.row_map,
        interp_transpose.graph.entries,
        interp_transpose.values,
        false,
        g.graph.row_map,
        g.graph.entries,
        g.values,
        false,
        row_map_p1,
        entries_p1,
        values_p1
        );


    Kokkos::View<sgp_eid_t*> row_map_coarse("rows_coarse", nc + 1);
    KokkosSparse::Experimental::spgemm_symbolic(
        &kh,
        nc,
        n,
        nc,
        row_map_p1,
        entries_p1,
        false,
        interp_mtx.graph.row_map,
        interp_mtx.graph.entries,
        false,
        row_map_coarse
        );
    //coarse-graph adjacency matrix
    Kokkos::View<sgp_vid_t*> adj_coarse("adjacencies_coarse", kh.get_spgemm_handle()->get_c_nnz());
    Kokkos::View<sgp_wgt_t*> wgt_coarse("weights_coarse", kh.get_spgemm_handle()->get_c_nnz());

    KokkosSparse::Experimental::spgemm_numeric(
        &kh,
        nc,
        n,
        nc,
        row_map_p1,
        entries_p1,
        values_p1,
        false,
        interp_mtx.graph.row_map,
        interp_mtx.graph.entries,
        interp_mtx.values,
        false,
        row_map_coarse,
        adj_coarse,
        wgt_coarse
        );
#else
    Kokkos::View<sgp_eid_t*> row_map_p1("rows_partial", n + 1);
    KokkosSparse::Experimental::spgemm_symbolic(
        &kh,
        n,
        n,
        nc,
        g.graph.row_map,
        g.graph.entries,
        false,
        interp_mtx.graph.row_map,
        interp_mtx.graph.entries,
        false,
        row_map_p1
        );

    //partial-result matrix
    Kokkos::View<sgp_vid_t*> entries_p1("adjacencies_partial", kh.get_spgemm_handle()->get_c_nnz());
    Kokkos::View<sgp_wgt_t*> values_p1("weights_partial", kh.get_spgemm_handle()->get_c_nnz());

    KokkosSparse::Experimental::spgemm_numeric(
        &kh,
        n,
        n,
        nc,
        g.graph.row_map,
        g.graph.entries,
        g.values,
        false,
        interp_mtx.graph.row_map,
        interp_mtx.graph.entries,
        interp_mtx.values,
        false,
        row_map_p1,
        entries_p1,
        values_p1
        );


    Kokkos::View<sgp_eid_t*> row_map_coarse("rows_coarse", nc + 1);
    KokkosSparse::Experimental::spgemm_symbolic(
        &kh,
        nc,
        n,
        nc,
        interp_transpose.graph.row_map,
        interp_transpose.graph.entries,
        false,
        row_map_p1,
        entries_p1,
        false,
        row_map_coarse
        );
    //coarse-graph adjacency matrix
    Kokkos::View<sgp_vid_t*> adj_coarse("adjacencies_coarse", kh.get_spgemm_handle()->get_c_nnz());
    Kokkos::View<sgp_wgt_t*> wgt_coarse("weights_coarse", kh.get_spgemm_handle()->get_c_nnz());

    KokkosSparse::Experimental::spgemm_numeric(
        &kh,
        nc,
        n,
        nc,
        interp_transpose.graph.row_map,
        interp_transpose.graph.entries,
        interp_transpose.values,
        false,
        row_map_p1,
        entries_p1,
        values_p1,
        false,
        row_map_coarse,
        adj_coarse,
        wgt_coarse
        );
#endif

    edge_view_t nonLoops("nonLoop", nc);

    //gonna reuse this to count non-self loop edges
    Kokkos::parallel_for(nc, KOKKOS_LAMBDA(sgp_vid_t i) {
        nonLoops(i) = 0;
    });

    Kokkos::parallel_for(nc, KOKKOS_LAMBDA(sgp_vid_t u) {
        for (sgp_eid_t j = row_map_coarse(u); j < row_map_coarse(u + 1); j++) {
            if (adj_coarse(j) != u) {
                nonLoops(u)++;
            }
        }
    });

    Kokkos::View<sgp_eid_t*> row_map_nonloop("nonloop row map", nc + 1);

    Kokkos::parallel_scan(nc, KOKKOS_LAMBDA(const sgp_vid_t i,
        sgp_eid_t & update, const bool final) {
        // Load old value in case we update it before accumulating
        const sgp_eid_t val_i = nonLoops(i);
        // For inclusive scan,
        // change the update value before updating array.
        update += val_i;
        if (final) {
            row_map_nonloop(i + 1) = update; // only update array on final pass
        }
    });

    Kokkos::View<sgp_eid_t*> rmn_subview = Kokkos::subview(row_map_nonloop, std::make_pair(nc, nc + 1));
    Kokkos::View<sgp_eid_t*>::HostMirror rmn_subview_m = Kokkos::create_mirror(rmn_subview);
    Kokkos::deep_copy(rmn_subview_m, rmn_subview);

    Kokkos::View<sgp_vid_t*> entries_nonloop("nonloop entries", rmn_subview_m(0));
    Kokkos::View<sgp_wgt_t*> values_nonloop("nonloop values", rmn_subview_m(0));

    Kokkos::parallel_for(nc, KOKKOS_LAMBDA(sgp_vid_t i) {
        nonLoops(i) = 0;
    });

    Kokkos::parallel_for(nc, KOKKOS_LAMBDA(sgp_vid_t u) {
        for (sgp_eid_t j = row_map_coarse(u); j < row_map_coarse(u + 1); j++) {
            if (adj_coarse(j) != u) {
                sgp_eid_t offset = row_map_nonloop(u) + nonLoops(u)++;
                entries_nonloop(offset) = adj_coarse(j);
                values_nonloop(offset) = wgt_coarse(j);
            }
        }
    });

    kh.destroy_spgemm_handle();

    graph_type gc_graph(entries_nonloop, row_map_nonloop);
    gc = matrix_type("gc", nc, values_nonloop, gc_graph);

    return EXIT_SUCCESS;
}

KOKKOS_INLINE_FUNCTION
void heap_deduplicate(const sgp_eid_t bottom, const sgp_eid_t top, vtx_view_t dest_by_source, wgt_view_t wgt_by_source, sgp_vid_t& edges_per_source) {

    sgp_vid_t size = top - bottom;
    sgp_eid_t offset = bottom;
    sgp_eid_t last_offset = offset;
    //max heapify (root at source_bucket_offset[u+1] - 1)
    for (sgp_vid_t i = size / 2; i > 0; i--) {
        sgp_eid_t heap_node = top - i, leftC = top - 2 * i, rightC = top - 1 - 2 * i;
        sgp_vid_t j = i;
        //heapify heap_node
        while ((2 * j <= size && dest_by_source(heap_node) < dest_by_source(leftC)) || (2 * j + 1 <= size && dest_by_source(heap_node) < dest_by_source(rightC))) {
            if (2 * j + 1 > size || dest_by_source(leftC) > dest_by_source(rightC)) {
                sgp_vid_t swap = dest_by_source(leftC);
                dest_by_source(leftC) = dest_by_source(heap_node);
                dest_by_source(heap_node) = swap;

                sgp_wgt_t w_swap = wgt_by_source(leftC);
                wgt_by_source(leftC) = wgt_by_source(heap_node);
                wgt_by_source(heap_node) = w_swap;
                j = 2 * j;
            }
            else {
                sgp_vid_t swap = dest_by_source(rightC);
                dest_by_source(rightC) = dest_by_source(heap_node);
                dest_by_source(heap_node) = swap;

                sgp_wgt_t w_swap = wgt_by_source(rightC);
                wgt_by_source(rightC) = wgt_by_source(heap_node);
                wgt_by_source(heap_node) = w_swap;
                j = 2 * j + 1;
            }
            heap_node = top - j, leftC = top - 2 * j, rightC = top - 1 - 2 * j;
        }
    }

    //heap sort
    for (sgp_eid_t i = bottom; i < top; i++) {

        sgp_vid_t top_swap = dest_by_source(top - 1);
        dest_by_source(top - 1) = dest_by_source(i);
        dest_by_source(i) = top_swap;

        sgp_wgt_t top_w_swap = wgt_by_source(top - 1);
        wgt_by_source(top - 1) = wgt_by_source(i);
        wgt_by_source(i) = top_w_swap;

        size--;

        sgp_vid_t j = 1;
        sgp_eid_t heap_node = top - j, leftC = top - 2 * j, rightC = top - 1 - 2 * j;
        //re-heapify root node
        while ((2 * j <= size && dest_by_source(heap_node) < dest_by_source(leftC)) || (2 * j + 1 <= size && dest_by_source(heap_node) < dest_by_source(rightC))) {
            if (2 * j + 1 > size || dest_by_source(leftC) > dest_by_source(rightC)) {
                sgp_vid_t swap = dest_by_source(leftC);
                dest_by_source(leftC) = dest_by_source(heap_node);
                dest_by_source(heap_node) = swap;

                sgp_wgt_t w_swap = wgt_by_source(leftC);
                wgt_by_source(leftC) = wgt_by_source(heap_node);
                wgt_by_source(heap_node) = w_swap;
                j = 2 * j;
            }
            else {
                sgp_vid_t swap = dest_by_source(rightC);
                dest_by_source(rightC) = dest_by_source(heap_node);
                dest_by_source(heap_node) = swap;

                sgp_wgt_t w_swap = wgt_by_source(rightC);
                wgt_by_source(rightC) = wgt_by_source(heap_node);
                wgt_by_source(heap_node) = w_swap;
                j = 2 * j + 1;
            }
            heap_node = top - j, leftC = top - 2 * j, rightC = top - 1 - 2 * j;
        }

        //sub-array is now sorted from bottom to i

        if (last_offset < offset) {
            if (dest_by_source(last_offset) == dest_by_source(i)) {
                wgt_by_source(last_offset) += wgt_by_source(i);
            }
            else {
                dest_by_source(offset) = dest_by_source(i);
                wgt_by_source(offset) = wgt_by_source(i);
                last_offset = offset;
                offset++;
            }
        }
        else {
            offset++;
        }
    }
    edges_per_source = offset - bottom;
}

SGPAR_API int sgp_build_coarse_graph_msd(matrix_type& gc,
    const matrix_type& vcmap,
    const matrix_type& g,
    const int coarsening_level,
    double* time_ptrs) {
    sgp_vid_t n = g.numRows();
    sgp_vid_t nc = vcmap.numCols();

    //radix sort source vertices, then sort edges
    Kokkos::View<const sgp_eid_t> rm_subview = Kokkos::subview(g.graph.row_map, n);
    sgp_eid_t size_rm = 0;
    Kokkos::deep_copy(size_rm, rm_subview);
    vtx_view_t mapped_edges("mapped edges", size_rm);

    edge_view_t source_bucket_offset("source_bucket_offsets", nc + 1);

    sgp_eid_t gc_nedges = 0;

    vtx_view_t edges_per_source("edges_per_source", nc);

    Kokkos::Timer timer;

    //count edges per vertex
    Kokkos::parallel_for(n, KOKKOS_LAMBDA(sgp_vid_t i) {
        sgp_vid_t u = vcmap.graph.entries(i);
        for (sgp_eid_t j = g.graph.row_map(i); j < g.graph.row_map(i + 1); j++) {
            sgp_vid_t v = vcmap.graph.entries(g.graph.entries(j));
            mapped_edges(j) = v;
            if (u != v) {
                Kokkos::atomic_increment(&edges_per_source(u));
            }
        }
    });

    time_ptrs[2] += timer.seconds();
    timer.reset();

    Kokkos::parallel_scan(nc, KOKKOS_LAMBDA(const sgp_vid_t i,
        sgp_eid_t & update, const bool final) {
        // Load old value in case we update it before accumulating
        const sgp_eid_t val_i = edges_per_source(i);
        // For inclusive scan,
        // change the update value before updating array.
        update += val_i;
        if (final) {
            source_bucket_offset(i + 1) = update; // only update array on final pass
        }
    });

    Kokkos::parallel_for(nc, KOKKOS_LAMBDA(sgp_vid_t i) {
        edges_per_source(i) = 0; // will use as counter again
    });

    time_ptrs[3] += timer.seconds();
    timer.reset();

    Kokkos::View<sgp_eid_t> sbo_subview = Kokkos::subview(source_bucket_offset, nc);
    sgp_eid_t size_sbo = 0;
    Kokkos::deep_copy(size_sbo, sbo_subview);

    vtx_view_t dest_by_source("dest_by_source", size_sbo);
    wgt_view_t wgt_by_source("wgt_by_source", size_sbo);

    Kokkos::parallel_for(n, KOKKOS_LAMBDA(sgp_vid_t i) {
        sgp_vid_t u = vcmap.graph.entries(i);
        for (sgp_eid_t j = g.graph.row_map(i); j < g.graph.row_map(i + 1); j++) {
            sgp_vid_t v = mapped_edges(j);
            if (u != v) {
                sgp_eid_t offset = Kokkos::atomic_fetch_add(&edges_per_source(u), 1);

                offset += source_bucket_offset(u);

                dest_by_source(offset) = v;
                wgt_by_source(offset) = g.values[j];
            }
        }
    });

    time_ptrs[4] += timer.seconds();
    timer.reset();

    //sort by dest and deduplicate
    Kokkos::parallel_reduce(nc, KOKKOS_LAMBDA(const sgp_vid_t u, sgp_eid_t & thread_sum) {
        sgp_eid_t bottom = source_bucket_offset(u);
        sgp_eid_t top = source_bucket_offset(u + 1);
#if 0
        sgp_eid_t next_offset = bottom;
        Kokkos::UnorderedMap<sgp_vid_t, sgp_eid_t> map(top - bottom);
        //hashing sort
        for (sgp_eid_t i = bottom; i < top; i++) {

            sgp_vid_t v = dest_by_source(i);

            if (map.exists(v)) {
                uint32_t key = map.find(v);
                sgp_eid_t idx = map.value_at(key);

                wgt_by_source(idx) += wgt_by_source(i);
            }
            else {
                map.insert( v, next_offset );
                dest_by_source(next_offset) = dest_by_source(i);
                wgt_by_source(next_offset) = wgt_by_source(i);
                next_offset++;
            }
        }

        edges_per_source(u) = next_offset - bottom;
#endif
        heap_deduplicate(bottom, top, dest_by_source, wgt_by_source, edges_per_source(u));
        thread_sum += edges_per_source(u);
    }, gc_nedges);

    time_ptrs[5] += timer.seconds();

    edge_view_t source_offsets("source_offsets", nc + 1);

    Kokkos::parallel_scan(nc, KOKKOS_LAMBDA(const sgp_vid_t i,
        sgp_eid_t & update, const bool final) {
        // Load old value in case we update it before accumulating
        const sgp_eid_t val_i = edges_per_source(i);
        // For inclusive scan,
        // change the update value before updating array.
        update += val_i;
        if (final) {
            source_offsets(i + 1) = update; // only update array on final pass
        }
    });

    vtx_view_t dest_idx("dest_idx", gc_nedges);
    wgt_view_t wgts("wgts", gc_nedges);

    Kokkos::parallel_for(nc, KOKKOS_LAMBDA(sgp_vid_t u) {
        sgp_eid_t end_offset = source_bucket_offset(u) + edges_per_source(u);
        sgp_vid_t dest_offset = source_offsets(u);
        for (sgp_eid_t j = source_bucket_offset(u); j < end_offset; j++) {
            dest_idx(dest_offset) = dest_by_source(j);
            wgts(dest_offset) = wgt_by_source(j);
            dest_offset++;
        }
    });

    graph_type gc_graph(dest_idx, source_offsets);
    gc = matrix_type("gc", nc, wgts, gc_graph);

    return EXIT_SUCCESS;
}

SGPAR_API int sgp_coarsen_one_level(matrix_type& gc, matrix_type& interpolation_graph,
    const matrix_type& g,
    const int coarsening_level,
    sgp_pcg32_random_t* rng,
    double* time_ptrs) {

    Kokkos::Timer timer;
    sgp_vid_t nvertices_coarse;
    sgp_coarsen_HEC(interpolation_graph, &nvertices_coarse, g, coarsening_level, rng, time_ptrs);
    time_ptrs[0] += timer.seconds();

    timer.reset();
    sgp_build_coarse_graph_spgemm(gc, interpolation_graph, g, coarsening_level, time_ptrs);
    time_ptrs[1] += timer.seconds();

    return EXIT_SUCCESS;
}

SGPAR_API int sgp_generate_coarse_graphs(const sgp_graph_t* fine_g, std::list<matrix_type>& coarse_graphs, std::list<matrix_type>& interp_mtxs, sgp_pcg32_random_t* rng, double* time_ptrs) {

    Kokkos::Timer timer;
    sgp_vid_t fine_n = fine_g->nvertices;
    edge_view_t row_map("row map", fine_n + 1);
    edge_mirror_t row_mirror = Kokkos::create_mirror(row_map);
    vtx_view_t entries("entries", fine_g->source_offsets[fine_n]);
    vtx_mirror_t entries_mirror = Kokkos::create_mirror(entries);
    wgt_view_t values("values", fine_g->source_offsets[fine_n]);
    wgt_mirror_t values_mirror = Kokkos::create_mirror(values);

    Kokkos::parallel_for(host_policy(0, fine_n + 1), KOKKOS_LAMBDA(sgp_vid_t u) {
        row_mirror(u) = fine_g->source_offsets[u];
    });
    Kokkos::parallel_for(host_policy(0, fine_g->source_offsets[fine_n]), KOKKOS_LAMBDA(sgp_vid_t i) {
        entries_mirror(i) = fine_g->destination_indices[i];
        values_mirror(i) = 1.0;
    });

    Kokkos::deep_copy(row_map, row_mirror);
    Kokkos::deep_copy(entries, entries_mirror);
    Kokkos::deep_copy(values, values_mirror);

    graph_type fine_graph(entries, row_map);
    coarse_graphs.push_back(matrix_type("interpolate", fine_g->nvertices, values, fine_graph));

    printf("Fine graph copy to device time: %.8f\n", timer.seconds());

    int coarsening_level = 0;
    while (coarse_graphs.rbegin()->numRows() > SGPAR_COARSENING_VTX_CUTOFF) {
        printf("Calculating coarse graph %ld\n", coarse_graphs.size());

        coarse_graphs.push_back(matrix_type());
        interp_mtxs.push_back(matrix_type());
        auto end_pointer = coarse_graphs.rbegin();

        CHECK_SGPAR(sgp_coarsen_one_level(*coarse_graphs.rbegin(),
            *interp_mtxs.rbegin(),
            *(++coarse_graphs.rbegin()),
            ++coarsening_level,
            rng, time_ptrs));

#ifdef DEBUG
        sgp_real_t coarsen_ratio = (sgp_real_t) coarse_graphs.rbegin()->numRows() / (sgp_real_t) (++coarse_graphs.rbegin())->numRows();
        printf("Coarsening ratio: %.8f\n", coarsen_ratio);
#endif
    }

    //don't use the coarsest level if it has too few vertices
    if (coarse_graphs.rbegin()->numRows() < 30) {
        coarse_graphs.pop_back();
        interp_mtxs.pop_back();
        coarsening_level--;
    }

    return EXIT_SUCCESS;
}

}
}