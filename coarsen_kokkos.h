#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <list>
#include <ctime>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#else
#include <sys/time.h>
#endif

#include <unordered_map>
// #define USE_GNU_PARALLELMODE
#ifdef USE_GNU_PARALLELMODE
#include <parallel/algorithm> // for parallel sort
#else 
#include <algorithm>          // for STL sort
#endif

#include "ExperimentLoggerUtil.cpp"
#include "heuristics.h"
#include "definitions_kokkos.h"

namespace sgpar {
namespace sgpar_kokkos {



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
	vtx_view_t& c_vtx_w, const vtx_view_t f_vtx_w,
    const matrix_type& interp_mtx,
    const matrix_type& g,
    const int coarsening_level) {

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

	c_vtx_w = vtx_view_t("coarse vtx weights", interp_mtx.numCols());
    KokkosSparse::spmv("N", 1.0, interp_transpose, f_vtx_w, 0.0, c_vtx_w);

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

template<typename ExecutionSpace>
struct functorDedupeAfterSort
{
    typedef ExecutionSpace execution_space;

    edge_view_t row_map;
    vtx_view_t entries;
    wgt_view_t wgts;
    wgt_view_t wgtsOut;
    edge_view_t dedupe_edge_count;

    functorDedupeAfterSort(edge_view_t row_map,
        vtx_view_t entries,
        wgt_view_t wgts,
        wgt_view_t wgtsOut,
        edge_view_t dedupe_edge_count)
        : row_map(row_map)
        , entries(entries)
        , wgts(wgts)
        , wgtsOut(wgtsOut)
        , dedupe_edge_count(dedupe_edge_count) {}

    KOKKOS_INLINE_FUNCTION
        void operator()(const member& thread, sgp_eid_t& thread_sum) const
    {
        sgp_vid_t u = thread.league_rank();
        sgp_eid_t start = row_map(u);
        sgp_eid_t end = row_map(u + 1);
        Kokkos::parallel_scan(Kokkos::TeamThreadRange(thread, start, end), [=](const sgp_eid_t& i, sgp_eid_t& update, const bool final) {
            if (i == start) {
                update += 1;
            }
            else if (entries(i) != entries(i - 1)) {
                update += 1;
            }
            if (final) {
                entries(start + update - 1) = entries(i);
                Kokkos::atomic_add(&wgtsOut(start + update - 1), wgts(i));
                if (i + 1 == end) {
                    dedupe_edge_count(u) == update;
                }
            }
            });
        thread_sum += dedupe_edge_count(u);
    }

    KOKKOS_INLINE_FUNCTION
        void operator()(const sgp_vid_t& u, sgp_eid_t& thread_sum) const
    {
        sgp_vid_t offset = row_map(u);
        sgp_vid_t last = SGP_INFTY;
        for (sgp_eid_t i = row_map(u); i < row_map(u + 1); i++) {
            if (last != entries(i)) {
                entries(offset) = entries(i);
                wgtsOut(offset) = wgts(i);
                last = entries(offset);
                offset++;
            }
            else {
                wgtsOut(offset - 1) += wgts(i);
            }
        }
        dedupe_edge_count(u) = offset - row_map(u);
        thread_sum += offset - row_map(u);
    }
};

template<typename ExecutionSpace, typename uniform_memory_pool_t>
struct functorHashmapAccumulator
{
    typedef ExecutionSpace execution_space;

	vtx_view_t remaining;
    edge_view_t row_map;
    vtx_view_t entries;
    wgt_view_t wgts;
    edge_view_t dedupe_edge_count;
    uniform_memory_pool_t _memory_pool;
    const sgp_vid_t _hash_size;
    const sgp_vid_t _max_hash_entries;

    typedef Kokkos::Experimental::UniqueToken<execution_space, Kokkos::Experimental::UniqueTokenScope::Global> unique_token_t;
    unique_token_t tokens;

    functorHashmapAccumulator(edge_view_t row_map,
        vtx_view_t entries,
        wgt_view_t wgts,
        edge_view_t dedupe_edge_count,
        uniform_memory_pool_t memory_pool,
        const sgp_vid_t hash_size,
        const sgp_vid_t max_hash_entries,
		vtx_view_t remaining)
        : row_map(row_map)
        , entries(entries)
        , wgts(wgts)
        , dedupe_edge_count(dedupe_edge_count)
        , _memory_pool(memory_pool)
        , _hash_size(hash_size)
        , _max_hash_entries(max_hash_entries)
		, remaining(remaining)
        , tokens(ExecutionSpace()){}

	//reduces to find total number of rows that were too large
    KOKKOS_INLINE_FUNCTION
        void operator()(const sgp_vid_t idx_unrem, sgp_vid_t& thread_sum) const
    {
		sgp_vid_t idx = remaining(idx_unrem);
        typedef sgp_vid_t hash_size_type;
        typedef sgp_vid_t hash_key_type;
        typedef sgp_wgt_t hash_value_type;

		//can't do this row at current hashmap size
		if(row_map(idx + 1) - row_map(idx) >= _max_hash_entries){
			thread_sum++;
		   	return;
		}
        // Alternative to team_policy thread id
        auto tid = tokens.acquire();

        // Acquire a chunk from the memory pool using a spin-loop.
        volatile sgp_vid_t* ptr_temp = nullptr;
        while (nullptr == ptr_temp)
        {
            ptr_temp = (volatile sgp_vid_t*)(_memory_pool.allocate_chunk(tid));
        }
        sgp_vid_t* ptr_memory_pool_chunk = (sgp_vid_t*)(ptr_temp);

        KokkosKernels::Experimental::HashmapAccumulator<hash_size_type, hash_key_type, hash_value_type> hash_map;

        // Set pointer to hash indices
        sgp_vid_t* used_hash_indices = (sgp_vid_t*)(ptr_temp);
        ptr_temp += _hash_size;

        // Set pointer to hash begins
        hash_map.hash_begins = (sgp_vid_t*)(ptr_temp);
        ptr_temp += _hash_size;

        // Set pointer to hash nexts
        hash_map.hash_nexts = (sgp_vid_t*)(ptr_temp);

        // Set pointer to hash keys
        hash_map.keys = (sgp_vid_t*) entries.data() + row_map(idx);

        // Set pointer to hash values
        hash_map.values = (sgp_wgt_t*) wgts.data() + row_map(idx);

        // Set up limits in Hashmap_Accumulator
        hash_map.hash_key_size = _max_hash_entries;
        hash_map.max_value_size = _max_hash_entries;

        // hash function is hash_size-1 (note: hash_size must be a power of 2)
        sgp_vid_t hash_func_pow2 = _hash_size - 1;

        // These are updated by Hashmap_Accumulator insert functions.
        sgp_vid_t used_hash_size = 0;
        sgp_vid_t used_hash_count = 0;

        // Loop over stuff
        for (sgp_eid_t i = row_map(idx); i < row_map(idx + 1); i++)
        {
            sgp_vid_t key = entries(i);
            sgp_wgt_t value = wgts(i);

            // Compute the hash index using & instead of % (modulus is slower).
            sgp_vid_t hash = key & hash_func_pow2;

            int r = hash_map.sequential_insert_into_hash_mergeAdd_TrackHashes(hash,
                key,
                value,
                &used_hash_size,
                hash_map.max_value_size,
                &used_hash_count,
                used_hash_indices);

            // Check return code
            if (r)
            {
                // insert should return nonzero if the insert failed, but for sequential_insert_into_hash_TrackHashes
                // the 'full' case is currently ignored, so r will always be 0.
            }
        }

        //sgp_vid_t insert_at = row_map(idx);

        // Reset the Begins values to -1 before releasing the memory pool chunk.
        // If you don't do this the next thread that grabs this memory chunk will not work properly.
        for (sgp_vid_t i = 0; i < used_hash_count; i++)
        {
            sgp_vid_t dirty_hash = used_hash_indices[i];
            //entries(insert_at) = hash_map.keys[i];
            //wgts(insert_at) = hash_map.values[i];

            hash_map.hash_begins[dirty_hash] = -1;
            //insert_at++;
        }

        //used_hash_size gives the number of entries, used_hash_count gives the number of dirty hash values (I think)
        dedupe_edge_count(idx) = used_hash_size;//insert_at - row_map(idx);

        // Release the memory pool chunk back to the pool
        _memory_pool.release_chunk(ptr_memory_pool_chunk);

        // Release the UniqueToken
        tokens.release(tid);

    }   // operator()

};  // functorHashmapAccumulator

SGPAR_API int sgp_build_coarse_graph_msd(matrix_type& gc,
    vtx_view_t& c_vtx_w,
    const matrix_type& vcmap,
    const matrix_type& g,
    const vtx_view_t& f_vtx_w,
    const int coarsening_level,
    ExperimentLoggerUtil& experiment) {
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

    c_vtx_w = vtx_view_t("coarse vertex weights", nc);

    //count edges per vertex
    Kokkos::parallel_for(policy(n, Kokkos::AUTO), KOKKOS_LAMBDA(const member& thread) {
        sgp_vid_t u = vcmap.graph.entries(thread.league_rank());
        sgp_eid_t start = g.graph.row_map(thread.league_rank());
        sgp_eid_t end = g.graph.row_map(thread.league_rank() + 1);
        sgp_vid_t nonLoopEdgesTotal = 0;
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(thread, start, end), [=] (const sgp_eid_t idx, sgp_vid_t& local_sum) {
            sgp_vid_t v = vcmap.graph.entries(g.graph.entries(idx));
            mapped_edges(idx) = v;
            if (u != v) {
                local_sum++;
            }
        }, nonLoopEdgesTotal);
        Kokkos::single(Kokkos::PerTeam(thread), [=]() {
            Kokkos::atomic_add(&edges_per_source(u), nonLoopEdgesTotal);
            Kokkos::atomic_add(&c_vtx_w(u), f_vtx_w(thread.league_rank()));
        });
    });

    experiment.addMeasurement(ExperimentLoggerUtil::Measurement::Count, timer.seconds());
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

    experiment.addMeasurement(ExperimentLoggerUtil::Measurement::Prefix, timer.seconds());
    timer.reset();

    Kokkos::View<sgp_eid_t> sbo_subview = Kokkos::subview(source_bucket_offset, nc);
    sgp_eid_t size_sbo = 0;
    Kokkos::deep_copy(size_sbo, sbo_subview);

    vtx_view_t dest_by_source("dest_by_source", size_sbo);
    wgt_view_t wgt_by_source("wgt_by_source", size_sbo);

    Kokkos::parallel_for(policy(n, Kokkos::AUTO), KOKKOS_LAMBDA(const member& thread) {
        sgp_vid_t u = vcmap.graph.entries(thread.league_rank());
        sgp_eid_t start = g.graph.row_map(thread.league_rank());
        sgp_eid_t end = g.graph.row_map(thread.league_rank() + 1);
        Kokkos::parallel_for(Kokkos::TeamThreadRange(thread, start, end), [=](const sgp_eid_t idx) {
            sgp_vid_t v = mapped_edges(idx);
            if (u != v) {
                sgp_eid_t offset = Kokkos::atomic_fetch_add(&edges_per_source(u), 1);

                offset += source_bucket_offset(u);

                dest_by_source(offset) = v;
                wgt_by_source(offset) = g.values(idx);
            }
        });
    });

    experiment.addMeasurement(ExperimentLoggerUtil::Measurement::Bucket, timer.seconds());
    timer.reset();

#ifdef HASHMAP
    
	sgp_vid_t remaining_count = nc;
	vtx_view_t remaining("remaining vtx", nc);
	Kokkos::parallel_for(nc, KOKKOS_LAMBDA(const sgp_vid_t i){
		remaining(i) = i;
	});
	do{
	//figure out max size for hashmap
    sgp_vid_t avg_entries = 0;
    if(typeid(Kokkos::DefaultExecutionSpace::memory_space) != typeid(Kokkos::DefaultHostExecutionSpace::memory_space) && static_cast<double>(remaining_count) / static_cast<double>(nc) > 0.01){
		Kokkos::parallel_reduce("calc average", remaining_count, KOKKOS_LAMBDA(const sgp_vid_t i, sgp_vid_t& thread_sum){
			sgp_vid_t u = remaining(i);
        	sgp_vid_t degree = edges_per_source(u);
    		thread_sum += degree;
		}, avg_entries);
		//degrees are often skewed so we want to err on the side of bigger hashmaps
		avg_entries = avg_entries * 2 / remaining_count;
		if(avg_entries < 50) avg_entries = 50;
	} else {
	Kokkos::parallel_reduce("calc average", remaining_count, KOKKOS_LAMBDA(const sgp_vid_t i, sgp_vid_t& thread_max){
		sgp_vid_t u = remaining(i);
        sgp_vid_t degree = edges_per_source(u);
		if(degree > thread_max){
    		thread_max = degree;
		}
	}, Kokkos::Max<sgp_vid_t, Kokkos::HostSpace>(avg_entries));
	avg_entries++;
	}

    typedef typename KokkosKernels::Impl::UniformMemoryPool<Kokkos::DefaultExecutionSpace, sgp_vid_t> uniform_memory_pool_t;
    // Set the hash_size as the next power of 2 bigger than hash_size_hint.
    // - hash_size must be a power of two since we use & rather than % (which is slower) for
    // computing the hash value for HashmapAccumulator.
	sgp_vid_t max_entries = avg_entries;
    sgp_vid_t hash_size = 1;
    while (hash_size < max_entries) { hash_size *= 2; }

    // Create Uniform Initialized Memory Pool
    KokkosKernels::Impl::PoolType pool_type = KokkosKernels::Impl::ManyThread2OneChunk;

	if(typeid(Kokkos::DefaultExecutionSpace::memory_space) == typeid(Kokkos::DefaultHostExecutionSpace::memory_space)){
	//	pool_type = KokkosKernels::Impl::OneThread2OneChunk;
	}

    // Determine memory chunk size for UniformMemoryPool
    sgp_vid_t mem_chunk_size = hash_size;      // for hash indices
    mem_chunk_size += hash_size;            // for hash begins
    mem_chunk_size += max_entries;     // for hash nexts
    // Set a cap on # of chunks to 32.  In application something else should be done
    // here differently if we're OpenMP vs. GPU but for this example we can just cap
    // our number of chunks at 32.
    sgp_vid_t mem_chunk_count = Kokkos::DefaultExecutionSpace::concurrency();

	if(typeid(Kokkos::DefaultExecutionSpace::memory_space) != typeid(Kokkos::DefaultHostExecutionSpace::memory_space)){
		//walk back number of mem_chunks if necessary
		size_t mem_needed = static_cast<size_t>(mem_chunk_count) * static_cast<size_t>(mem_chunk_size) * sizeof(sgp_vid_t);
		size_t max_mem_allowed = 536870912;//1073741824;
		if(mem_needed > max_mem_allowed){
			size_t chunk_dif = mem_needed - max_mem_allowed;
			chunk_dif = chunk_dif / (static_cast<size_t>(mem_chunk_size) * sizeof(sgp_vid_t));
			chunk_dif++;
			mem_chunk_count -= chunk_dif;
		}
	}

    uniform_memory_pool_t memory_pool(mem_chunk_count, mem_chunk_size, -1, pool_type);

    functorHashmapAccumulator<Kokkos::DefaultExecutionSpace, uniform_memory_pool_t>
        hashmapAccumulator(source_bucket_offset, dest_by_source, wgt_by_source, edges_per_source, memory_pool, hash_size, max_entries, remaining);

	sgp_vid_t old_remaining_count = remaining_count;
    Kokkos::parallel_reduce("hashmap time", old_remaining_count, hashmapAccumulator, remaining_count);

	if(remaining_count > 0){
		vtx_view_t new_remaining("new remaining vtx", remaining_count);

		Kokkos::parallel_scan("move remaining vertices", old_remaining_count, KOKKOS_LAMBDA(const sgp_vid_t i, sgp_vid_t& update, const bool final){
			sgp_vid_t u = remaining(i);
			if(edges_per_source(u) >= max_entries){
				if(final){
					new_remaining(update) = u;
				}
				update++;
			}
		});

		remaining = new_remaining;
	}
	//printf("remaining count: %u\n", remaining_count);
	} while(remaining_count > 0);
#elif defined(RADIX)
    Kokkos::Timer radix;
    KokkosSparse::Experimental::SortEntriesFunctor<Kokkos::DefaultExecutionSpace, sgp_eid_t, sgp_vid_t, edge_view_t, vtx_view_t>
        sortEntries(source_bucket_offset, dest_by_source, wgt_by_source);
    Kokkos::parallel_for("radix sort time", policy(nc, Kokkos::AUTO), sortEntries);
    experiment.addMeasurement(ExperimentLoggerUtil::Measurement::RadixSort, radix.seconds());
    radix.reset();

    functorDedupeAfterSort<Kokkos::DefaultExecutionSpace>
        deduper(source_bucket_offset, dest_by_source, wgt_by_source, wgt_by_source, edges_per_source);
    Kokkos::parallel_reduce("deduplicated sorted", nc, deduper, gc_nedges);
    experiment.addMeasurement(ExperimentLoggerUtil::Measurement::RadixDedupe, radix.seconds());
    radix.reset();
#else
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
#endif

    experiment.addMeasurement(ExperimentLoggerUtil::Measurement::Dedupe, timer.seconds());
    timer.reset();

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

#ifdef HASHMAP
    Kokkos::View<sgp_eid_t> edge_total_subview = Kokkos::subview(source_offsets, nc);
    Kokkos::deep_copy(gc_nedges, edge_total_subview);
#endif

    vtx_view_t dest_idx("dest_idx", gc_nedges);
    wgt_view_t wgts("wgts", gc_nedges);

    Kokkos::parallel_for(policy(nc, Kokkos::AUTO), KOKKOS_LAMBDA(const member& thread) {
        sgp_vid_t u = thread.league_rank();
        sgp_eid_t start_origin = source_bucket_offset(u);
        sgp_eid_t start_dest = source_offsets(u);
        Kokkos::parallel_for(Kokkos::TeamThreadRange(thread, edges_per_source(u)), [=](const sgp_eid_t idx) {
            dest_idx(start_dest + idx) = dest_by_source(start_origin + idx);
            wgts(start_dest + idx) = wgt_by_source(start_origin + idx);
        });
    });

    graph_type gc_graph(dest_idx, source_offsets);
    gc = matrix_type("gc", nc, wgts, gc_graph);

    return EXIT_SUCCESS;
}

SGPAR_API int sgp_coarsen_one_level(matrix_type& gc, matrix_type& interpolation_graph, vtx_view_t& c_vtx_w,
    const matrix_type& g,
    const vtx_view_t& f_vtx_w,
    const int coarsening_level,
    sgp_pcg32_random_t* rng,
    ExperimentLoggerUtil& experiment) {

    Kokkos::Timer timer;
    sgp_vid_t nvertices_coarse;
#ifdef HEC
    sgp_coarsen_HEC(interpolation_graph, &nvertices_coarse, g, coarsening_level, rng, experiment);
#elif defined PUREMATCH || MTMETIS
    sgp_coarsen_match(interpolation_graph, &nvertices_coarse, g, coarsening_level, rng, experiment);
#else
    sgp_coarsen_mis_2(interpolation_graph, &nvertices_coarse, g, coarsening_level, rng, experiment);
#endif
	experiment.addMeasurement(ExperimentLoggerUtil::Measurement::Map, timer.seconds());

    timer.reset();
#ifdef SPGEMM
    sgp_build_coarse_graph_spgemm(gc, c_vtx_w, f_vtx_w, interpolation_graph, g, coarsening_level);
#else
    sgp_build_coarse_graph_msd(gc, c_vtx_w, interpolation_graph, g, f_vtx_w, coarsening_level, experiment);
#endif
    experiment.addMeasurement(ExperimentLoggerUtil::Measurement::Build, timer.seconds());
    timer.reset();

    return EXIT_SUCCESS;
}

SGPAR_API int sgp_generate_coarse_graphs(const sgp_graph_t* fine_g, std::list<matrix_type>& coarse_graphs, std::list<matrix_type>& interp_mtxs, std::list<vtx_view_t>& vtx_weights_list, sgp_pcg32_random_t* rng, ExperimentLoggerUtil& experiment) {

    Kokkos::Timer timer;
    sgp_vid_t fine_n = fine_g->nvertices;
    edge_view_t row_map("row map", fine_n + 1);
    edge_mirror_t row_mirror = Kokkos::create_mirror(row_map);
    vtx_view_t entries("entries", fine_g->source_offsets[fine_n]);
    vtx_mirror_t entries_mirror = Kokkos::create_mirror(entries);
    wgt_view_t values("values", fine_g->source_offsets[fine_n]);
    wgt_mirror_t values_mirror = Kokkos::create_mirror(values);
    vtx_view_t vtx_weights("vtx weights", fine_n);

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

    Kokkos::parallel_for(fine_n, KOKKOS_LAMBDA(const sgp_vid_t i){
        vtx_weights(i) = 1;
    });

    graph_type fine_graph(entries, row_map);
    coarse_graphs.push_back(matrix_type("interpolate", fine_g->nvertices, values, fine_graph));
    vtx_weights_list.push_back(vtx_weights);

    printf("Fine graph copy to device time: %.8f\n", timer.seconds());

    int coarsening_level = 0;
    while (coarse_graphs.rbegin()->numRows() > SGPAR_COARSENING_VTX_CUTOFF) {
        printf("Calculating coarse graph %ld\n", coarse_graphs.size());

        coarse_graphs.push_back(matrix_type());
        vtx_view_t coarse_vtx_weights;
        interp_mtxs.push_back(matrix_type());
        auto end_pointer = coarse_graphs.rbegin();

        CHECK_SGPAR(sgp_coarsen_one_level(*coarse_graphs.rbegin(),
            *interp_mtxs.rbegin(),
            coarse_vtx_weights,
            *(++coarse_graphs.rbegin()),
            *(vtx_weights_list.rbegin()),
            ++coarsening_level,
            rng, experiment));

        vtx_weights_list.push_back(coarse_vtx_weights);

		if(coarse_graphs.size() > 200) break;
#ifdef DEBUG
        sgp_real_t coarsen_ratio = (sgp_real_t) coarse_graphs.rbegin()->numRows() / (sgp_real_t) (++coarse_graphs.rbegin())->numRows();
        printf("Coarsening ratio: %.8f\n", coarsen_ratio);
#endif
    }

    //don't use the coarsest level if it has too few vertices
    if (coarse_graphs.rbegin()->numRows() < 10) {
        coarse_graphs.pop_back();
        interp_mtxs.pop_back();
        vtx_weights_list.pop_back();
        coarsening_level--;
    }

    return EXIT_SUCCESS;
}

}
}
