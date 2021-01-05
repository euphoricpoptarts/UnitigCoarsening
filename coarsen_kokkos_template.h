#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <list>
#include <ctime>
#include <limits>
#include <fstream>
#include <sstream>

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
#include "heuristics_template.h"

template<typename ordinal_t, typename edge_offset_t, typename scalar_t, class Device>
class coarse_builder {
public:
    using matrix_t = typename KokkosSparse::CrsMatrix<scalar_t, ordinal_t, Device, void, edge_offset_t>;
    using vtx_view_t = typename Kokkos::View<ordinal_t>;
    using graph_type = typename matrix_t::staticcrsgraph_type;

    // contains matrix and vertex weights corresponding to current level
    // interp matrix maps previous level to this level
    struct coarse_level_triple {
        matrix_t coarse_mtx;
        vtx_view_t coarse_vtx_wgts;
        matrix_t interp_mtx;
        int level;
    };

    enum Heuristic { HEC, HECv2, HECv3, Match, MtMetis, MIS2, GOSH, GOSHv2 };
    // default heuristic is HEC
    Heuristic h = HEC;
    coarsen_heuristics mapper;

private:
    using wgt_view_t = typename Kokkos::View<scalar_t>;
    using edge_view_t = typename Kokkos::View<edge_offset_t>;
    bool use_hashmap = false;
    const ordinal_t ORD_MAX = std::numeric_limits<ordinal_t>::max();

//assumes that matrix has one entry-per row, not valid for general matrices
int compute_transpose(const matrix_t& mtx,
    matrix_t& transpose) {
    ordinal_t n = mtx.numRows();
    ordinal_t nc = mtx.numCols();

    vtx_view_t fine_per_coarse("fine_per_coarse", nc);
    //transpose interpolation matrix
    vtx_view_t adj_transpose("adj_transpose", n);
    wgt_view_t wgt_transpose("weights_transpose", n);
    edge_view_t row_map_transpose("rows_transpose", nc + 1);

    Kokkos::parallel_for(n, KOKKOS_LAMBDA(ordinal_t i) {
        ordinal_t v = mtx.graph.entries(i);
        Kokkos::atomic_increment(&fine_per_coarse(v));
    });
    Kokkos::parallel_scan(nc, KOKKOS_LAMBDA(const ordinal_t i,
        ordinal_t & update, const bool final) {
        // Load old value in case we update it before accumulating
        const ordinal_t val_i = fine_per_coarse(i);
        // For inclusive scan,
        // change the update value before updating array.
        update += val_i;
        if (final) {
            row_map_transpose(i + 1) = update; // only update array on final pass
        }
    });
    Kokkos::parallel_for(nc, KOKKOS_LAMBDA(ordinal_t i) {
        fine_per_coarse(i) = 0;
    });
    Kokkos::parallel_for(n, KOKKOS_LAMBDA(ordinal_t i) {
        ordinal_t v = mtx.graph.entries(i);
        edge_offset_t offset = row_map_transpose(v) + Kokkos::atomic_fetch_add(&fine_per_coarse(v), 1);
        adj_transpose(offset) = i;
        wgt_transpose(offset) = 1;
    });

    graph_type transpose_graph(adj_transpose, row_map_transpose);
    transpose = matrix_t("transpose", n, wgt_transpose, transpose_graph);

    return EXIT_SUCCESS;
}

/*
int sgp_build_coarse_graph_spgemm(matrix_t& gc,
    vtx_view_t& c_vtx_w, const vtx_view_t f_vtx_w,
    const matrix_t& interp_mtx,
    const matrix_t& g,
    const int coarsening_level) {

    ordinal_t n = g.numRows();
    ordinal_t nc = interp_mtx.numCols();

    matrix_t interp_transpose;// = KokkosKernels::Impl::transpose_matrix(interp_mtx);
    compute_transpose(interp_mtx, interp_transpose);

    //write_g(g, "dump/g_dump.mtx", true);
    //write_g(interp_mtx, "dump/interp_dump.mtx", false);
    //write_g(interp_transpose, "dump/interp_transpose_dump.mtx", false);
    typedef KokkosKernels::Experimental::KokkosKernelsHandle
        <edge_offset_t, ordinal_t, scalar_t,
        typename Device::execution_space, typename Device::memory_space, typename Device::memory_space > KernelHandle;

    KernelHandle kh;
    kh.set_team_work_size(16);
    kh.set_dynamic_scheduling(true);

    // Select an spgemm algorithm, limited by configuration at compile-time and set via the handle
    KokkosSparse::SPGEMMAlgorithm spgemm_algorithm = KokkosSparse::SPGEMM_KK_MEMORY;
    kh.create_spgemm_handle(spgemm_algorithm);

#ifdef TRANSPOSE_FIRST
    edge_view_t row_map_p1("rows_partial", nc + 1);
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
    vtx_view_t entries_p1("adjacencies_partial", kh.get_spgemm_handle()->get_c_nnz());
    wgt_view_t values_p1("weights_partial", kh.get_spgemm_handle()->get_c_nnz());

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


    edge_view_t row_map_coarse("rows_coarse", nc + 1);
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
    vtx_view_t adj_coarse("adjacencies_coarse", kh.get_spgemm_handle()->get_c_nnz());
    wgt_view_t wgt_coarse("weights_coarse", kh.get_spgemm_handle()->get_c_nnz());

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
    edge_view_t row_map_p1("rows_partial", n + 1);
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
    vtx_view_t entries_p1("adjacencies_partial", kh.get_spgemm_handle()->get_c_nnz());
    wgt_view_t values_p1("weights_partial", kh.get_spgemm_handle()->get_c_nnz());

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


    edge_view_t row_map_coarse("rows_coarse", nc + 1);
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
    vtx_view_t adj_coarse("adjacencies_coarse", kh.get_spgemm_handle()->get_c_nnz());
    wgt_view_t wgt_coarse("weights_coarse", kh.get_spgemm_handle()->get_c_nnz());

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
    Kokkos::parallel_for(nc, KOKKOS_LAMBDA(ordinal_t i) {
        nonLoops(i) = 0;
    });

    Kokkos::parallel_for(nc, KOKKOS_LAMBDA(ordinal_t u) {
        for (edge_offset_t j = row_map_coarse(u); j < row_map_coarse(u + 1); j++) {
            if (adj_coarse(j) != u) {
                nonLoops(u)++;
            }
        }
    });

    edge_view_t row_map_nonloop("nonloop row map", nc + 1);

    Kokkos::parallel_scan(nc, KOKKOS_LAMBDA(const ordinal_t i,
        edge_offset_t & update, const bool final) {
        // Load old value in case we update it before accumulating
        const edge_offset_t val_i = nonLoops(i);
        // For inclusive scan,
        // change the update value before updating array.
        update += val_i;
        if (final) {
            row_map_nonloop(i + 1) = update; // only update array on final pass
        }
    });

    //fix this to use a 1D view
    Kokkos::View<edge_offset_t*> rmn_subview = Kokkos::subview(row_map_nonloop, std::make_pair(nc, nc + 1));
    Kokkos::View<edge_offset_t*>::HostMirror rmn_subview_m = Kokkos::create_mirror(rmn_subview);
    Kokkos::deep_copy(rmn_subview_m, rmn_subview);

    vtx_view_t entries_nonloop("nonloop entries", rmn_subview_m(0));
    wgt_view_t values_nonloop("nonloop values", rmn_subview_m(0));

    Kokkos::parallel_for(nc, KOKKOS_LAMBDA(ordinal_t i) {
        nonLoops(i) = 0;
    });

    Kokkos::parallel_for(nc, KOKKOS_LAMBDA(ordinal_t u) {
        for (edge_offset_t j = row_map_coarse(u); j < row_map_coarse(u + 1); j++) {
            if (adj_coarse(j) != u) {
                edge_offset_t offset = row_map_nonloop(u) + nonLoops(u)++;
                entries_nonloop(offset) = adj_coarse(j);
                values_nonloop(offset) = wgt_coarse(j);
            }
        }
    });

    kh.destroy_spgemm_handle();

    graph_type gc_graph(entries_nonloop, row_map_nonloop);
    gc = matrix_t("gc", nc, values_nonloop, gc_graph);

    c_vtx_w = vtx_view_t("coarse vtx weights", interp_mtx.numCols());
    KokkosSparse::spmv("N", 1.0, interp_transpose, f_vtx_w, 0.0, c_vtx_w);

    return EXIT_SUCCESS;
}
*/

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

/*    KOKKOS_INLINE_FUNCTION
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
*/
    KOKKOS_INLINE_FUNCTION
        void operator()(const ordinal_t& u, edge_offset_t& thread_sum) const
    {
        ordinal_t offset = row_map(u);
        ordinal_t last = ORD_MAX;
        for (edge_offset_t i = row_map(u); i < row_map(u + 1); i++) {
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
    const ordinal_t _hash_size;
    const ordinal_t _max_hash_entries;

    typedef Kokkos::Experimental::UniqueToken<execution_space, Kokkos::Experimental::UniqueTokenScope::Global> unique_token_t;
    unique_token_t tokens;

    functorHashmapAccumulator(edge_view_t row_map,
        vtx_view_t entries,
        wgt_view_t wgts,
        edge_view_t dedupe_edge_count,
        uniform_memory_pool_t memory_pool,
        const ordinal_t hash_size,
        const ordinal_t max_hash_entries,
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
        void operator()(const ordinal_t idx_unrem, ordinal_t& thread_sum) const
    {
        ordinal_t idx = remaining(idx_unrem);
        typedef ordinal_t hash_size_type;
        typedef ordinal_t hash_key_type;
        typedef scalar_t hash_value_type;

        //can't do this row at current hashmap size
        if(row_map(idx + 1) - row_map(idx) >= _max_hash_entries){
            thread_sum++;
            return;
        }
        // Alternative to team_policy thread id
        auto tid = tokens.acquire();

        // Acquire a chunk from the memory pool using a spin-loop.
        volatile ordinal_t* ptr_temp = nullptr;
        while (nullptr == ptr_temp)
        {
            ptr_temp = (volatile ordinal_t*)(_memory_pool.allocate_chunk(tid));
        }
        ordinal_t* ptr_memory_pool_chunk = (ordinal_t*)(ptr_temp);

        KokkosKernels::Experimental::HashmapAccumulator<hash_size_type, hash_key_type, hash_value_type> hash_map;

        // Set pointer to hash indices
        ordinal_t* used_hash_indices = (ordinal_t*)(ptr_temp);
        ptr_temp += _hash_size;

        // Set pointer to hash begins
        hash_map.hash_begins = (ordinal_t*)(ptr_temp);
        ptr_temp += _hash_size;

        // Set pointer to hash nexts
        hash_map.hash_nexts = (ordinal_t*)(ptr_temp);

        // Set pointer to hash keys
        hash_map.keys = (ordinal_t*) entries.data() + row_map(idx);

        // Set pointer to hash values
        hash_map.values = (scalar_t*) wgts.data() + row_map(idx);

        // Set up limits in Hashmap_Accumulator
        hash_map.hash_key_size = _max_hash_entries;
        hash_map.max_value_size = _max_hash_entries;

        // hash function is hash_size-1 (note: hash_size must be a power of 2)
        ordinal_t hash_func_pow2 = _hash_size - 1;

        // These are updated by Hashmap_Accumulator insert functions.
        ordinal_t used_hash_size = 0;
        ordinal_t used_hash_count = 0;

        // Loop over stuff
        for (edge_offset_t i = row_map(idx); i < row_map(idx + 1); i++)
        {
            ordinal_t key = entries(i);
            scalar_t value = wgts(i);

            // Compute the hash index using & instead of % (modulus is slower).
            ordinal_t hash = key & hash_func_pow2;

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
        for (ordinal_t i = 0; i < used_hash_count; i++)
        {
            ordinal_t dirty_hash = used_hash_indices[i];
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

void sgp_deduplicate_graph(const ordinal_t n, const ordinal_t nc,
    vtx_view_t edges_per_source, vtx_view_t dest_by_source, wgt_view_t wgt_by_source,
    const edge_view_t source_bucket_offset, ExperimentLoggerUtil& experiment, edge_offset_t& gc_nedges) {

    if (use_hashmap) {

        ordinal_t remaining_count = nc;
        vtx_view_t remaining("remaining vtx", nc);
        Kokkos::parallel_for(nc, KOKKOS_LAMBDA(const ordinal_t i){
            remaining(i) = i;
        });
        do {
            //figure out max size for hashmap
            ordinal_t avg_entries = 0;
            if (typeid(Kokkos::DefaultExecutionSpace::memory_space) != typeid(Kokkos::DefaultHostExecutionSpace::memory_space) && static_cast<double>(remaining_count) / static_cast<double>(nc) > 0.01) {
                Kokkos::parallel_reduce("calc average", remaining_count, KOKKOS_LAMBDA(const ordinal_t i, ordinal_t & thread_sum){
                    ordinal_t u = remaining(i);
                    ordinal_t degree = edges_per_source(u);
                    thread_sum += degree;
                }, avg_entries);
                //degrees are often skewed so we want to err on the side of bigger hashmaps
                avg_entries = avg_entries * 2 / remaining_count;
                if (avg_entries < 50) avg_entries = 50;
            }
            else {
                Kokkos::parallel_reduce("calc average", remaining_count, KOKKOS_LAMBDA(const ordinal_t i, ordinal_t & thread_max){
                    ordinal_t u = remaining(i);
                    ordinal_t degree = edges_per_source(u);
                    if (degree > thread_max) {
                        thread_max = degree;
                    }
                }, Kokkos::Max<ordinal_t, Kokkos::HostSpace>(avg_entries));
                avg_entries++;
            }

            typedef typename KokkosKernels::Impl::UniformMemoryPool<Kokkos::DefaultExecutionSpace, ordinal_t> uniform_memory_pool_t;
            // Set the hash_size as the next power of 2 bigger than hash_size_hint.
            // - hash_size must be a power of two since we use & rather than % (which is slower) for
            // computing the hash value for HashmapAccumulator.
            ordinal_t max_entries = avg_entries;
            ordinal_t hash_size = 1;
            while (hash_size < max_entries) { hash_size *= 2; }

            // Create Uniform Initialized Memory Pool
            KokkosKernels::Impl::PoolType pool_type = KokkosKernels::Impl::ManyThread2OneChunk;

            if (typeid(Kokkos::DefaultExecutionSpace::memory_space) == typeid(Kokkos::DefaultHostExecutionSpace::memory_space)) {
                //	pool_type = KokkosKernels::Impl::OneThread2OneChunk;
            }

            // Determine memory chunk size for UniformMemoryPool
            ordinal_t mem_chunk_size = hash_size;      // for hash indices
            mem_chunk_size += hash_size;            // for hash begins
            mem_chunk_size += max_entries;     // for hash nexts
            // Set a cap on # of chunks to 32.  In application something else should be done
            // here differently if we're OpenMP vs. GPU but for this example we can just cap
            // our number of chunks at 32.
            ordinal_t mem_chunk_count = Kokkos::DefaultExecutionSpace::concurrency();

            if (typeid(Kokkos::DefaultExecutionSpace::memory_space) != typeid(Kokkos::DefaultHostExecutionSpace::memory_space)) {
                //walk back number of mem_chunks if necessary
                size_t mem_needed = static_cast<size_t>(mem_chunk_count) * static_cast<size_t>(mem_chunk_size) * sizeof(ordinal_t);
                size_t max_mem_allowed = 536870912;//1073741824;
                if (mem_needed > max_mem_allowed) {
                    size_t chunk_dif = mem_needed - max_mem_allowed;
                    chunk_dif = chunk_dif / (static_cast<size_t>(mem_chunk_size) * sizeof(ordinal_t));
                    chunk_dif++;
                    mem_chunk_count -= chunk_dif;
                }
            }

            uniform_memory_pool_t memory_pool(mem_chunk_count, mem_chunk_size, -1, pool_type);

            functorHashmapAccumulator<Kokkos::DefaultExecutionSpace, uniform_memory_pool_t>
                hashmapAccumulator(source_bucket_offset, dest_by_source, wgt_by_source, edges_per_source, memory_pool, hash_size, max_entries, remaining);

            ordinal_t old_remaining_count = remaining_count;
            Kokkos::parallel_reduce("hashmap time", old_remaining_count, hashmapAccumulator, remaining_count);

            if (remaining_count > 0) {
                vtx_view_t new_remaining("new remaining vtx", remaining_count);

                Kokkos::parallel_scan("move remaining vertices", old_remaining_count, KOKKOS_LAMBDA(const ordinal_t i, ordinal_t & update, const bool final){
                    ordinal_t u = remaining(i);
                    if (edges_per_source(u) >= max_entries) {
                        if (final) {
                            new_remaining(update) = u;
                        }
                        update++;
                    }
                });

                remaining = new_remaining;
            }
            //printf("remaining count: %u\n", remaining_count);
        } while (remaining_count > 0);
    }
    else {
        Kokkos::Timer radix;
        KokkosSparse::Experimental::SortEntriesFunctor<Kokkos::DefaultExecutionSpace, edge_offset_t, ordinal_t, edge_view_t, vtx_view_t>
            sortEntries(source_bucket_offset, dest_by_source, wgt_by_source);
        Kokkos::parallel_for("radix sort time", policy(nc, Kokkos::AUTO), sortEntries);
        experiment.addMeasurement(ExperimentLoggerUtil::Measurement::RadixSort, radix.seconds());
        radix.reset();

        functorDedupeAfterSort<Kokkos::DefaultExecutionSpace>
            deduper(source_bucket_offset, dest_by_source, wgt_by_source, wgt_by_source, edges_per_source);
        Kokkos::parallel_reduce("deduplicated sorted", nc, deduper, gc_nedges);
        experiment.addMeasurement(ExperimentLoggerUtil::Measurement::RadixDedupe, radix.seconds());
        radix.reset();
    }

}

coarse_level_triple sgp_build_nonskew(const matrix_t g,
    const matrix_t vcmap,
    vtx_view_t mapped_edges,
    vtx_view_t edges_per_source,
    ExperimentLoggerUtil& experiment,
    Kokkos::Timer& timer) {

    ordinal_t n = g.numRows();
    ordinal_t nc = vcmap.numCols();
    edge_view_t source_bucket_offset("source_bucket_offsets", nc + 1);
    edge_offset_t gc_nedges = 0;

    experiment.addMeasurement(ExperimentLoggerUtil::Measurement::Count, timer.seconds());
    timer.reset();

    Kokkos::parallel_scan(nc, KOKKOS_LAMBDA(const ordinal_t i,
        edge_offset_t & update, const bool final) {
        // Load old value in case we update it before accumulating
        const edge_offset_t val_i = edges_per_source(i);
        // For inclusive scan,
        // change the update value before updating array.
        update += val_i;
        if (final) {
            source_bucket_offset(i + 1) = update; // only update array on final pass
        }
    });

    Kokkos::parallel_for(nc, KOKKOS_LAMBDA(ordinal_t i) {
        edges_per_source(i) = 0; // will use as counter again
    });

    experiment.addMeasurement(ExperimentLoggerUtil::Measurement::Prefix, timer.seconds());
    timer.reset();

    Kokkos::View<edge_offset_t> sbo_subview = Kokkos::subview(source_bucket_offset, nc);
    edge_offset_t size_sbo = 0;
    Kokkos::deep_copy(size_sbo, sbo_subview);

    vtx_view_t dest_by_source("dest_by_source", size_sbo);
    wgt_view_t wgt_by_source("wgt_by_source", size_sbo);

    Kokkos::parallel_for(policy(n, Kokkos::AUTO), KOKKOS_LAMBDA(const member & thread) {
        ordinal_t u = vcmap.graph.entries(thread.league_rank());
        edge_offset_t start = g.graph.row_map(thread.league_rank());
        edge_offset_t end = g.graph.row_map(thread.league_rank() + 1);
        Kokkos::parallel_for(Kokkos::TeamThreadRange(thread, start, end), [=](const edge_offset_t idx) {
            ordinal_t v = mapped_edges(idx);
            if (u != v) {
                edge_offset_t offset = Kokkos::atomic_fetch_add(&edges_per_source(u), 1);

                offset += source_bucket_offset(u);

                dest_by_source(offset) = v;
                wgt_by_source(offset) = g.values(idx);
            }
            });
    });

    experiment.addMeasurement(ExperimentLoggerUtil::Measurement::Bucket, timer.seconds());
    timer.reset();

    sgp_deduplicate_graph(n, nc,
        edges_per_source, dest_by_source, wgt_by_source,
        source_bucket_offset, experiment, gc_nedges);

    experiment.addMeasurement(ExperimentLoggerUtil::Measurement::Dedupe, timer.seconds());
    timer.reset();

    edge_view_t source_offsets("source_offsets", nc + 1);

    Kokkos::parallel_scan(nc, KOKKOS_LAMBDA(const ordinal_t i,
        edge_offset_t & update, const bool final) {
        // Load old value in case we update it before accumulating
        const edge_offset_t val_i = edges_per_source(i);
        // For inclusive scan,
        // change the update value before updating array.
        update += val_i;
        if (final) {
            source_offsets(i + 1) = update; // only update array on final pass
        }
    });

    Kokkos::View<edge_offset_t> edge_total_subview = Kokkos::subview(source_offsets, nc);
    Kokkos::deep_copy(gc_nedges, edge_total_subview);

    vtx_view_t dest_idx("dest_idx", gc_nedges);
    wgt_view_t wgts("wgts", gc_nedges);

    Kokkos::parallel_for(policy(nc, Kokkos::AUTO), KOKKOS_LAMBDA(const member & thread) {
        ordinal_t u = thread.league_rank();
        edge_offset_t start_origin = source_bucket_offset(u);
        edge_offset_t start_dest = source_offsets(u);
        Kokkos::parallel_for(Kokkos::TeamThreadRange(thread, edges_per_source(u)), [=](const edge_offset_t idx) {
            dest_idx(start_dest + idx) = dest_by_source(start_origin + idx);
            wgts(start_dest + idx) = wgt_by_source(start_origin + idx);
            });
    });

    graph_type gc_graph(dest_idx, source_offsets);
    gc = matrix_type("gc", nc, wgts, gc_graph);

    coarse_level_triple next_level;
    next_level.coarse_mtx = gc;
    return next_level;
}

coarse_level_triple sgp_build_skew(const matrix_t g,
    const matrix_t vcmap,
    vtx_view_t mapped_edges,
    vtx_view_t degree_initial,
    ExperimentLoggerUtil& experiment,
    Kokkos::Timer& timer) {

    ordinal_t n = g.numRows();
    ordinal_t nc = vcmap.numCols();
    edge_offset_t gc_nedges = 0;
    
    vtx_view_t dedupe_count("dedupe count", n);
    edge_view_t row_map_copy("row map copy", n + 1);

    //recount with edges only belonging to fine vertex of coarse vertex of smaller degree
    Kokkos::parallel_for("recount edges", policy(n, Kokkos::AUTO), KOKKOS_LAMBDA(const member & thread) {
        ordinal_t outer_idx = thread.league_rank();
        ordinal_t u = vcmap.graph.entries(outer_idx);
        edge_offset_t start = g.graph.row_map(outer_idx);
        edge_offset_t end = g.graph.row_map(outer_idx + 1);
        ordinal_t nonLoopEdgesTotal = 0;
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(thread, start, end), [=](const edge_offset_t idx, ordinal_t& local_sum) {
            ordinal_t v = mapped_edges(idx);
            bool degree_less = degree_initial(u) < degree_initial(v);
            bool degree_equal = degree_initial(u) == degree_initial(v);
            if (u != v && (degree_less || (degree_equal && u < v))) {
                local_sum++;
            }
            }, nonLoopEdgesTotal);
        Kokkos::single(Kokkos::PerTeam(thread), [=]() {
            dedupe_count(outer_idx) = nonLoopEdgesTotal;
        });
    });

    experiment.addMeasurement(ExperimentLoggerUtil::Measurement::Count, timer.seconds());
    timer.reset();

    Kokkos::parallel_scan(n, KOKKOS_LAMBDA(const ordinal_t i,
        edge_offset_t & update, const bool final) {
        // Load old value in case we update it before accumulating
        const edge_offset_t val_i = dedupe_count(i);
        // For inclusive scan,
        // change the update value before updating array.
        update += val_i;
        if (final) {
            row_map_copy(i + 1) = update; // only update array on final pass
        }
    });

    Kokkos::parallel_for(n, KOKKOS_LAMBDA(ordinal_t i) {
        dedupe_count(i) = 0; // will use as counter again
    });

    experiment.addMeasurement(ExperimentLoggerUtil::Measurement::Prefix, timer.seconds());
    timer.reset();

    Kokkos::View<edge_offset_t> fine_recount_subview = Kokkos::subview(row_map_copy, n);
    edge_offset_t fine_recount = 0;
    Kokkos::deep_copy(fine_recount, fine_recount_subview);

    vtx_view_t dest_fine("fine to coarse dests", fine_recount);
    wgt_view_t wgt_fine("fine to coarse wgts", fine_recount);

    Kokkos::parallel_for(policy(n, Kokkos::AUTO), KOKKOS_LAMBDA(const member & thread) {
        ordinal_t outer_idx = thread.league_rank();
        ordinal_t u = vcmap.graph.entries(outer_idx);
        edge_offset_t start = g.graph.row_map(outer_idx);
        edge_offset_t end = g.graph.row_map(outer_idx + 1);
        Kokkos::parallel_for(Kokkos::TeamThreadRange(thread, start, end), [=](const edge_offset_t idx) {
            ordinal_t v = mapped_edges(idx);
            bool degree_less = degree_initial(u) < degree_initial(v);
            bool degree_equal = degree_initial(u) == degree_initial(v);
            if (u != v && (degree_less || (degree_equal && u < v))) {
                edge_offset_t offset = Kokkos::atomic_fetch_add(&dedupe_count(outer_idx), 1);

                offset += row_map_copy(outer_idx);

                dest_fine(offset) = v;
                wgt_fine(offset) = g.values(idx);
            }
            });
    });
    //"delete" these views
    Kokkos::resize(mapped_edges, 0);
    
    //deduplicate coarse adjacencies within each fine row
    sgp_deduplicate_graph(n, n,
        dedupe_count, dest_fine, wgt_fine,
        row_map_copy, experiment, gc_nedges);

    edge_view_t source_bucket_offset("source_bucket_offsets", nc + 1);
    vtx_view_t edges_per_source("edges_per_source", nc);

    Kokkos::parallel_for("sum fine row sizes", n, KOKKOS_LAMBDA(const ordinal_t i){
        ordinal_t u = vcmap.graph.entries(i);
        Kokkos::atomic_fetch_add(&edges_per_source(u), dedupe_count(i));
    });
    Kokkos::parallel_scan(nc, KOKKOS_LAMBDA(const ordinal_t i,
        edge_offset_t & update, const bool final) {
        // Load old value in case we update it before accumulating
        const edge_offset_t val_i = edges_per_source(i);
        // For inclusive scan,
        // change the update value before updating array.
        update += val_i;
        if (final) {
            source_bucket_offset(i + 1) = update; // only update array on final pass
        }
    });
    Kokkos::parallel_for(nc, KOKKOS_LAMBDA(const ordinal_t i){
        edges_per_source(i) = 0;
    });
    vtx_view_t dest_by_source("dest by source", gc_nedges);
    wgt_view_t wgt_by_source("wgt by source", gc_nedges);
    Kokkos::parallel_for("combine deduped fine rows", policy(n, Kokkos::AUTO), KOKKOS_LAMBDA(const member & thread) {
        ordinal_t outer_idx = thread.league_rank();
        ordinal_t u = vcmap.graph.entries(outer_idx);
        edge_offset_t start = row_map_copy(outer_idx);
        edge_offset_t end = start + dedupe_count(outer_idx);
        Kokkos::parallel_for(Kokkos::TeamThreadRange(thread, start, end), [=](const edge_offset_t idx) {
            ordinal_t v = dest_fine(idx);
            bool degree_less = degree_initial(u) < degree_initial(v);
            bool degree_equal = degree_initial(u) == degree_initial(v);
            if (degree_less || (degree_equal && u < v)) {
                edge_offset_t offset = Kokkos::atomic_fetch_add(&edges_per_source(u), 1);

                offset += source_bucket_offset(u);

                dest_by_source(offset) = v;
                wgt_by_source(offset) = wgt_fine(idx);
            }
            });
    });
    gc_nedges = 0;
    Kokkos::resize(dest_fine, 0);
    Kokkos::resize(wgt_fine, 0);

    experiment.addMeasurement(ExperimentLoggerUtil::Measurement::Bucket, timer.seconds());
    timer.reset();

    sgp_deduplicate_graph(n, nc,
        edges_per_source, dest_by_source, wgt_by_source,
        source_bucket_offset, experiment, gc_nedges);

    experiment.addMeasurement(ExperimentLoggerUtil::Measurement::Dedupe, timer.seconds());
    timer.reset();

    //reused degree initial as degree final
    vtx_view_t degree_final = degree_initial;
    Kokkos::parallel_for(nc, KOKKOS_LAMBDA(const ordinal_t i){
        degree_final(i) = edges_per_source(i);
    });

    Kokkos::parallel_for(policy(nc, Kokkos::AUTO), KOKKOS_LAMBDA(const member & thread) {
        ordinal_t u = thread.league_rank();
        edge_offset_t start = source_bucket_offset(u);
        edge_offset_t end = start + edges_per_source(u);
        Kokkos::parallel_for(Kokkos::TeamThreadRange(thread, start, end), [=](const edge_offset_t idx) {
            ordinal_t v = dest_by_source(idx);
            //increment other vertex
            Kokkos::atomic_fetch_add(&degree_final(v), 1);
            });
    });

    edge_view_t source_offsets("source_offsets", nc + 1);

    Kokkos::parallel_scan(nc, KOKKOS_LAMBDA(const ordinal_t i,
        edge_offset_t & update, const bool final) {
        // Load old value in case we update it before accumulating
        const edge_offset_t val_i = degree_final(i);
        // For inclusive scan,
        // change the update value before updating array.
        update += val_i;
        if (final) {
            source_offsets(i + 1) = update; // only update array on final pass
            degree_final(i) = 0;
        }
    });

    Kokkos::View<edge_offset_t> edge_total_subview = Kokkos::subview(source_offsets, nc);
    Kokkos::deep_copy(gc_nedges, edge_total_subview);

    vtx_view_t dest_idx("dest_idx", gc_nedges);
    wgt_view_t wgts("wgts", gc_nedges);

    Kokkos::parallel_for(policy(nc, Kokkos::AUTO), KOKKOS_LAMBDA(const member & thread) {
        ordinal_t u = thread.league_rank();
        edge_offset_t u_origin = source_bucket_offset(u);
        edge_offset_t u_dest_offset = source_offsets(u);
        Kokkos::parallel_for(Kokkos::TeamThreadRange(thread, edges_per_source(u)), [=](const edge_offset_t u_idx) {
            ordinal_t v = dest_by_source(u_origin + u_idx);
            scalar_t wgt = wgt_by_source(u_origin + u_idx);
            edge_offset_t v_dest_offset = source_offsets(v);
            edge_offset_t v_dest = v_dest_offset + Kokkos::atomic_fetch_add(&degree_final(v), 1);
            edge_offset_t u_dest = u_dest_offset + Kokkos::atomic_fetch_add(&degree_final(u), 1);

            dest_idx(u_dest) = v;
            wgts(u_dest) = wgt;
            dest_idx(v_dest) = u;
            wgts(v_dest) = wgt;
            });
    });

    graph_type gc_graph(dest_idx, source_offsets);
    gc = matrix_type("gc", nc, wgts, gc_graph);

    experiment.addMeasurement(ExperimentLoggerUtil::Measurement::WriteGraph, timer.seconds());
    timer.reset();

    coarse_level_triple next_level;
    next_level.coarse_mtx = gc;
    return next_level;
}

coarse_level_triple sgp_build_coarse_graph(const coarse_level_triple level,
    const matrix_t vcmap,
    ExperimentLoggerUtil& experiment) {

    matrix_t g = level.coarse_mtx;
    ordinal_t n = g.numRows();
    ordinal_t nc = vcmap.numCols();

    //radix sort source vertices, then sort edges
    Kokkos::View<const edge_offset_t> rm_subview = Kokkos::subview(g.graph.row_map, n);
    edge_offset_t size_rm = 0;
    Kokkos::deep_copy(size_rm, rm_subview);
    vtx_view_t mapped_edges("mapped edges", size_rm);

    Kokkos::Timer timer;

    vtx_view_t degree_initial("edges_per_source", nc);
    c_vtx_w = vtx_view_t("coarse vertex weights", nc);

    //count edges per vertex
    Kokkos::parallel_for(policy(n, Kokkos::AUTO), KOKKOS_LAMBDA(const member& thread) {
        ordinal_t u = vcmap.graph.entries(thread.league_rank());
        edge_offset_t start = g.graph.row_map(thread.league_rank());
        edge_offset_t end = g.graph.row_map(thread.league_rank() + 1);
        ordinal_t nonLoopEdgesTotal = 0;
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(thread, start, end), [=] (const edge_offset_t idx, ordinal_t& local_sum) {
            ordinal_t v = vcmap.graph.entries(g.graph.entries(idx));
            mapped_edges(idx) = v;
            if (u != v) {
                local_sum++;
            }
        }, nonLoopEdgesTotal);
        Kokkos::single(Kokkos::PerTeam(thread), [=]() {
            Kokkos::atomic_add(&degree_initial(u), nonLoopEdgesTotal);
            Kokkos::atomic_add(&c_vtx_w(u), f_vtx_w(thread.league_rank()));
        });
    });

    edge_offset_t total_unduped = 0;
    ordinal_t max_unduped = 0;

    Kokkos::parallel_reduce("find max", nc, KOKKOS_LAMBDA(const ordinal_t i, ordinal_t& l_max){
        if (l_max <= degree_initial(i)) {
            l_max = degree_initial(i);
        }
    }, Kokkos::Max<ordinal_t, Kokkos::HostSpace>(max_unduped));

    Kokkos::parallel_reduce("find total", nc, KOKKOS_LAMBDA(const ordinal_t i, edge_offset_t& sum){
        sum += degree_initial(i);
    }, total_unduped);

    edge_offset_t avg_unduped = total_unduped / nc;

    coarse_level_triple next_level;
    //only do if graph is sufficiently irregular
    //don't do optimizations if running on CPU (the default host space)
    if (avg_unduped > 50 && (max_unduped / 10) > avg_unduped && typeid(Kokkos::DefaultExecutionSpace::memory_space) != typeid(Kokkos::DefaultHostExecutionSpace::memory_space)) {
        next_level = sgp_build_skew(g, vcmap, mapped_edges, degree_initial, experiment, timer);
    }
    else {
        next_level = sgp_build_nonskew(g, vcmap, mapped_edges, degree_initial, experiment, timer);
    }

    next_level.coarse_vtx_wgts = f_vtx_w;
    next_level.level = level.level + 1;
    next_level.interp_mtx = vcmap;
    return next_level;
}

coarse_level_triple sgp_coarsen_one_level(const coarse_level_triple level,
    ExperimentLoggerUtil& experiment) {

    Kokkos::Timer timer;
    ordinal_t nvertices_coarse;
    matrix_t interpolation_graph;

    switch (h) {
        case HEC:
        case HECv2:
        case HECv3:
            mapper.sgp_coarsen_HEC(interpolation_graph, &nvertices_coarse, g, coarsening_level, rng, experiment);
            break;
        case Match:
        case MtMetis:
            mapper.sgp_coarsen_match(interpolation_graph, &nvertices_coarse, g, coarsening_level, rng, experiment);
            break;
        case MIS2:
            mapper.sgp_coarsen_mis_2(interpolation_graph, &nvertices_coarse, g, coarsening_level, rng, experiment);
            break;
        case GOSHv2:
            mapper.sgp_coarsen_GOSH_v2(interpolation_graph, &nvertices_coarse, g, coarsening_level, rng, experiment);
            break;
        case GOSH:
            mapper.sgp_coarsen_GOSH(interpolation_graph, &nvertices_coarse, g, coarsening_level, rng, experiment);
            break;
    }
    experiment.addMeasurement(ExperimentLoggerUtil::Measurement::Map, timer.seconds());

    timer.reset();
    coarse_level_triple next_level = sgp_build_coarse_graph(level, interpolation_graph, experiment);
    experiment.addMeasurement(ExperimentLoggerUtil::Measurement::Build, timer.seconds());
    timer.reset();

    return next_level;
}

public:
std::list<coarse_level_triple> sgp_generate_coarse_graphs(const matrix_t fine_g, ExperimentLoggerUtil& experiment) {

    Kokkos::Timer timer;
    ordinal_t fine_n = fine_g.numRows();
    std::list<coarse_level_triple> levels;
    coarse_level_triple finest;
    finest.coarse_mtx = fine_g;
    finest.level = 0;
    vtw_view_t vtx_weights("vertex weights", fine_n);
    Kokkos::parallel_for(fine_n, KOKKOS_LAMBDA(const ordinal_t i){
        vtx_weights(i) = 1;
    });
    finest.coarse_vtx_wgts = vtx_weights;
    levels.push_back(finest);
    printf("Fine graph copy to device time: %.8f\n", timer.seconds());
    while (levels.rbegin()->coarse_mtx.numRows() > SGPAR_COARSENING_VTX_CUTOFF) {
        printf("Calculating coarse graph %ld\n", levels.size());

        coarse_level_triple next_level = sgp_coarsen_one_level(*levels.rbegin(), experiment));

        levels.push_back(next_level);

        if(levels.size() > 200) break;
#ifdef DEBUG
        sgp_real_t coarsen_ratio = (sgp_real_t) levels.rbegin()->coarse_mtx.numRows() / (sgp_real_t) (++levels.rbegin())->coarse_mtx.numRows();
        printf("Coarsening ratio: %.8f\n", coarsen_ratio);
#endif
    }

    //don't use the coarsest level if it has too few vertices
    if (levels.rbegin()->coarse_mtx.numRows() < 10) {
        levels.pop_back();
    }

    return levels;
}

void set_heuristic(Heuristic h) {
    this->h = h;
}

};