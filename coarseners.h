#pragma once
#include <list>
#include <limits>
#include <Kokkos_Core.hpp>
#include <Kokkos_Atomic.hpp>
#include <Kokkos_UniqueToken.hpp>
#include <Kokkos_Functional.hpp>
#include "KokkosSparse_CrsMatrix.hpp"
#include "KokkosSparse_spmv.hpp"
#include "KokkosSparse_spgemm.hpp"
#include "KokkosKernels_SparseUtils.hpp"
#include "KokkosKernels_HashmapAccumulator.hpp"
#include "KokkosKernels_Uniform_Initialized_MemoryPool.hpp"
#include "ExperimentLoggerUtil.cpp"
#include "heuristics_template.h"

template<typename ordinal_t, typename edge_offset_t, typename scalar_t, class Device>
class coarse_builder {
public:

    // define internal types
    using exec_space = typename Device::execution_space;
    using mem_space = typename Device::memory_space;
    using matrix_t = KokkosSparse::CrsMatrix<scalar_t, ordinal_t, Device, void, edge_offset_t>;
    using vtx_view_t = Kokkos::View<ordinal_t*, Device>;
    using wgt_view_t = Kokkos::View<scalar_t*, Device>;
    using edge_view_t = Kokkos::View<edge_offset_t*, Device>;
    using edge_subview_t = Kokkos::View<edge_offset_t, Device>;
    using graph_type = typename matrix_t::staticcrsgraph_type;
    using policy_t = Kokkos::RangePolicy<exec_space>;
    using team_policy_t = Kokkos::TeamPolicy<exec_space>;
    using member = typename team_policy_t::member_type;
    using spgemm_kernel_handle = KokkosKernels::Experimental::KokkosKernelsHandle<edge_offset_t, ordinal_t, scalar_t, exec_space, mem_space, mem_space>;
    static constexpr ordinal_t ORD_MAX = std::numeric_limits<ordinal_t>::max();
    // contains matrix and vertex weights corresponding to current level
    // interp matrix maps previous level to this level
    struct coarse_level_triple {
        matrix_t coarse_mtx;
        vtx_view_t coarse_vtx_wgts;
        matrix_t interp_mtx;
        int level;
        bool uniform_weights;
    };

    // define behavior-controlling enums
    enum Heuristic { HECv1, HECv2, HECv3, Match, MtMetis, MIS2, GOSHv1, GOSHv2 };
    enum Builder { Sort, Hashmap, Spgemm, Spgemm_transpose_first };

    // internal parameters and data
    // default heuristic is HEC
    Heuristic h = HECv1;
    // default builder is sort
    Builder b = Sort;
    coarsen_heuristics<ordinal_t, edge_offset_t, scalar_t, Device> mapper;
    //when the results are fetched, this list is implicitly copied
    std::list<coarse_level_triple> results;
    ordinal_t coarse_vtx_cutoff = 50;
    ordinal_t min_allowed_vtx = 10;
    unsigned int max_levels = 200;

coarse_level_triple build_coarse_graph_spgemm(const coarse_level_triple level,
    const matrix_t interp_mtx,
    ExperimentLoggerUtil& experiment) {
    
    vtx_view_t f_vtx_w = level.coarse_vtx_wgts;
    matrix_t g = level.coarse_mtx;

    ordinal_t n = g.numRows();
    ordinal_t nc = interp_mtx.numCols();

    matrix_t interp_transpose = KokkosKernels::Impl::transpose_matrix(interp_mtx);

    spgemm_kernel_handle kh;
    kh.set_team_work_size(64);
    kh.set_dynamic_scheduling(true);
    KokkosSparse::SPGEMMAlgorithm spgemm_algorithm = KokkosSparse::SPGEMM_KK_MEMORY;
    kh.create_spgemm_handle(spgemm_algorithm);

    vtx_view_t adj_coarse;
    wgt_view_t wgt_coarse;
    edge_view_t row_map_coarse;

    if (b == Spgemm_transpose_first) {
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


        row_map_coarse = edge_view_t("rows_coarse", nc + 1);
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
        adj_coarse = vtx_view_t("adjacencies_coarse", kh.get_spgemm_handle()->get_c_nnz());
        wgt_coarse = wgt_view_t("weights_coarse", kh.get_spgemm_handle()->get_c_nnz());

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
    }
    else {
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


        row_map_coarse = edge_view_t("rows_coarse", nc + 1);
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
        adj_coarse = vtx_view_t("adjacencies_coarse", kh.get_spgemm_handle()->get_c_nnz());
        wgt_coarse = wgt_view_t("weights_coarse", kh.get_spgemm_handle()->get_c_nnz());

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
    }

    //now we must remove self-loop edges
    edge_view_t nonLoops("nonLoop", nc);

    //gonna reuse this to count non-self loop edges
    Kokkos::parallel_for(policy_t(0, nc), KOKKOS_LAMBDA(ordinal_t i) {
        nonLoops(i) = 0;
    });

    Kokkos::parallel_for(policy_t(0, nc), KOKKOS_LAMBDA(ordinal_t u) {
        for (edge_offset_t j = row_map_coarse(u); j < row_map_coarse(u + 1); j++) {
            if (adj_coarse(j) != u) {
                nonLoops(u)++;
            }
        }
    });

    edge_view_t row_map_nonloop("nonloop row map", nc + 1);

    Kokkos::parallel_scan(policy_t(0, nc), KOKKOS_LAMBDA(const ordinal_t i,
        edge_offset_t & update, const bool final) {
        const edge_offset_t val_i = nonLoops(i);
        update += val_i;
        if (final) {
            row_map_nonloop(i + 1) = update;
        }
    });

    edge_subview_t rmn_subview = Kokkos::subview(row_map_nonloop, nc);
    edge_offset_t rmn = 0;
    Kokkos::deep_copy(rmn, rmn_subview);

    vtx_view_t entries_nonloop("nonloop entries", rmn);
    wgt_view_t values_nonloop("nonloop values", rmn);

    Kokkos::parallel_for(policy_t(0, nc), KOKKOS_LAMBDA(const ordinal_t i) {
        nonLoops(i) = 0;
    });

    Kokkos::parallel_for(policy_t(0, nc), KOKKOS_LAMBDA(const ordinal_t u) {
        for (edge_offset_t j = row_map_coarse(u); j < row_map_coarse(u + 1); j++) {
            if (adj_coarse(j) != u) {
                edge_offset_t offset = row_map_nonloop(u) + nonLoops(u)++;
                entries_nonloop(offset) = adj_coarse(j);
                values_nonloop(offset) = wgt_coarse(j);
            }
        }
    });
    //done removing self-loop edges

    kh.destroy_spgemm_handle();

    graph_type gc_graph(entries_nonloop, row_map_nonloop);
    matrix_t gc("gc", nc, values_nonloop, gc_graph);

    vtx_view_t c_vtx_w("coarse vtx weights", interp_mtx.numCols());
    KokkosSparse::spmv("N", 1.0, interp_transpose, f_vtx_w, 0.0, c_vtx_w);

    coarse_level_triple next_level;
    next_level.coarse_mtx = gc;
    next_level.coarse_vtx_wgts = c_vtx_w;
    next_level.level = level.level + 1;
    next_level.interp_mtx = interp_mtx;
    next_level.uniform_weights = false;
    return next_level;
}

struct prefix_sum
{
    vtx_view_t input;
    edge_view_t output;

    prefix_sum(vtx_view_t input,
        edge_view_t output)
        : input(input)
        , output(output) {}

    KOKKOS_INLINE_FUNCTION
        void operator() (const ordinal_t i, edge_offset_t& update, const bool final) const {
        const edge_offset_t val_i = input(i);
        update += val_i;
        if (final) {
            output(i + 1) = update;
        }
    }
};

struct functorDedupeAfterSort
{
    //compiler may get confused what the reduction type is without this
    typedef edge_offset_t value_type;

    edge_view_t row_map;
    vtx_view_t entries, entriesOut;
    wgt_view_t wgts, wgtsOut;
    vtx_view_t dedupe_edge_count;

    functorDedupeAfterSort(edge_view_t row_map,
        vtx_view_t entries,
        vtx_view_t entriesOut,
        wgt_view_t wgts,
        wgt_view_t wgtsOut,
        vtx_view_t dedupe_edge_count)
        : row_map(row_map)
        , entries(entries)
        , entriesOut(entriesOut)
        , wgts(wgts)
        , wgtsOut(wgtsOut)
        , dedupe_edge_count(dedupe_edge_count) {}

    KOKKOS_INLINE_FUNCTION
        void operator()(const member& thread, edge_offset_t& thread_sum) const
    {
        ordinal_t u = thread.league_rank();
        edge_offset_t start = row_map(u);
        edge_offset_t end = row_map(u + 1);
        Kokkos::parallel_scan(Kokkos::TeamThreadRange(thread, start, end), [&](const edge_offset_t& i, edge_offset_t& update, const bool final) {
            if (i == start) {
                update += 1;
            }
            else if (entries(i) != entries(i - 1)) {
                update += 1;
            }
            if (final) {
                entriesOut(start + update - 1) = entries(i);
                Kokkos::atomic_add(&wgtsOut(start + update - 1), wgts(i));
                if (i + 1 == end) {
                    dedupe_edge_count(u) = update;
                }
            }
        });
        Kokkos::single(Kokkos::PerTeam(thread), [&]() {
            thread_sum += dedupe_edge_count(u);
        });
    }

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

struct functorCollapseDirectedToUndirected
{
    const edge_view_t source_row_map;
    const edge_view_t target_row_map;
    const vtx_view_t source_edge_counts;
    vtx_view_t target_edge_counts;
    const vtx_view_t source_destinations;
    vtx_view_t target_destinations;
    const wgt_view_t source_wgts;
    wgt_view_t target_wgts;

    functorCollapseDirectedToUndirected(const edge_view_t source_row_map,
        const edge_view_t target_row_map,
        const vtx_view_t source_edge_counts,
        vtx_view_t target_edge_counts,
        const vtx_view_t source_destinations,
        vtx_view_t target_destinations,
        const wgt_view_t source_wgts,
        wgt_view_t target_wgts)
        : source_row_map(source_row_map),
        target_row_map(target_row_map),
        source_edge_counts(source_edge_counts),
        target_edge_counts(target_edge_counts),
        source_destinations(source_destinations),
        target_destinations(target_destinations),
        source_wgts(source_wgts),
        target_wgts(target_wgts)
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const member& thread) const {
        ordinal_t u = thread.league_rank();
        edge_offset_t u_origin = source_row_map(u);
        edge_offset_t u_dest_offset = target_row_map(u);
        Kokkos::parallel_for(Kokkos::TeamThreadRange(thread, source_edge_counts(u)), [=](const edge_offset_t u_idx) {
            ordinal_t v = source_destinations(u_origin + u_idx);
            scalar_t wgt = source_wgts(u_origin + u_idx);
            edge_offset_t v_dest_offset = target_row_map(v);
            edge_offset_t v_dest = v_dest_offset + Kokkos::atomic_fetch_add(&target_edge_counts(v), 1);
            edge_offset_t u_dest = u_dest_offset + Kokkos::atomic_fetch_add(&target_edge_counts(u), 1);

            target_destinations(u_dest) = v;
            target_wgts(u_dest) = wgt;
            target_destinations(v_dest) = u;
            target_wgts(v_dest) = wgt;
        });
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
        
        // hash function is hash_size-1 (note: hash_size must be a power of 2)
        ordinal_t hash_func_pow2 = _hash_size - 1;


        // Set pointer to hash indices
        ordinal_t* used_hash_indices = (ordinal_t*)(ptr_temp);
        ptr_temp += _hash_size;

        // Set pointer to hash begins
        ordinal_t* hash_begins = (ordinal_t*)(ptr_temp);
        ptr_temp += _hash_size;

        // Set pointer to hash nexts
        ordinal_t* hash_nexts = (ordinal_t*)(ptr_temp);

        // Set pointer to hash keys
        ordinal_t* keys = (ordinal_t*) entries.data() + row_map(idx);

        // Set pointer to hash values
        scalar_t* values = (scalar_t*) wgts.data() + row_map(idx);
        
        KokkosKernels::Experimental::HashmapAccumulator<hash_size_type, hash_key_type, hash_value_type, KokkosKernels::Experimental::HashOpType::bitwiseAnd> 
            hash_map(_hash_size, hash_func_pow2, hash_begins, hash_nexts, keys, values);

        // These are updated by Hashmap_Accumulator insert functions.
        ordinal_t used_hash_size = 0;
        ordinal_t used_hash_count = 0;

        // Loop over stuff
        for (edge_offset_t i = row_map(idx); i < row_map(idx + 1); i++)
        {
            ordinal_t key = entries(i);
            scalar_t value = wgts(i);

            int r = hash_map.sequential_insert_into_hash_mergeAdd_TrackHashes(
                key,
                value,
                &used_hash_size,
                &used_hash_count,
                used_hash_indices);

            // Check return code
            if (r)
            {
                // insert should return nonzero if the insert failed, but for sequential_insert_into_hash_TrackHashes
                // the 'full' case is currently ignored, so r will always be 0.
            }
        }

        // Reset the Begins values to -1 before releasing the memory pool chunk.
        // If you don't do this the next thread that grabs this memory chunk will not work properly.
        for (ordinal_t i = 0; i < used_hash_count; i++)
        {
            ordinal_t dirty_hash = used_hash_indices[i];
            //entries(insert_at) = hash_map.keys[i];
            //wgts(insert_at) = hash_map.values[i];

            hash_map.hash_begins[dirty_hash] = ORD_MAX;
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

void deduplicate_graph(const ordinal_t n, const bool use_team,
    vtx_view_t edges_per_source, vtx_view_t dest_by_source, wgt_view_t wgt_by_source,
    const edge_view_t source_bucket_offset, ExperimentLoggerUtil& experiment, edge_offset_t& gc_nedges) {

    if (b == Hashmap) {
        ordinal_t remaining_count = n;
        vtx_view_t remaining("remaining vtx", n);
        Kokkos::parallel_for(policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t i){
            remaining(i) = i;
        });
        do {
            //determine size for hashmap
            ordinal_t avg_entries = 0;
            if (typeid(typename exec_space::memory_space) != typeid(typename Kokkos::DefaultHostExecutionSpace::memory_space) && static_cast<double>(remaining_count) / static_cast<double>(n) > 0.01) {
                Kokkos::parallel_reduce("calc average among remaining", policy_t(0, remaining_count), KOKKOS_LAMBDA(const ordinal_t i, ordinal_t & thread_sum){
                    ordinal_t u = remaining(i);
                    ordinal_t degree = edges_per_source(u);
                    thread_sum += degree;
                }, avg_entries);
                //degrees are often skewed so we want to err on the side of bigger hashmaps
                avg_entries = avg_entries * 2 / remaining_count;
                if (avg_entries < 50) avg_entries = 50;
            }
            else {
                Kokkos::parallel_reduce("calc max", policy_t(0, remaining_count), KOKKOS_LAMBDA(const ordinal_t i, ordinal_t & thread_max){
                    ordinal_t u = remaining(i);
                    ordinal_t degree = edges_per_source(u);
                    if (degree > thread_max) {
                        thread_max = degree;
                    }
                }, Kokkos::Max<ordinal_t, Kokkos::HostSpace>(avg_entries));
                //need precisely one larger than max, don't remember why atm
                avg_entries++;
            }

            typedef typename KokkosKernels::Impl::UniformMemoryPool<exec_space, ordinal_t> uniform_memory_pool_t;
            // Set the hash_size as the next power of 2 bigger than hash_size_hint.
            // - hash_size must be a power of two since we use & rather than % (which is slower) for
            // computing the hash value for HashmapAccumulator.
            ordinal_t max_entries = avg_entries;
            ordinal_t hash_size = 1;
            while (hash_size < max_entries) { hash_size *= 2; }

            // Create Uniform Initialized Memory Pool
            KokkosKernels::Impl::PoolType pool_type = KokkosKernels::Impl::ManyThread2OneChunk;

            if (typeid(typename exec_space::memory_space) == typeid(typename Kokkos::DefaultHostExecutionSpace::memory_space)) {
                //	pool_type = KokkosKernels::Impl::OneThread2OneChunk;
            }

            // Determine memory chunk size for UniformMemoryPool
            ordinal_t mem_chunk_size = hash_size;      // for hash indices
            mem_chunk_size += hash_size;            // for hash begins
            mem_chunk_size += max_entries;     // for hash nexts
            // Set a cap on # of chunks to 32.  In application something else should be done
            // here differently if we're OpenMP vs. GPU but for this example we can just cap
            // our number of chunks at 32.
            ordinal_t mem_chunk_count = exec_space::concurrency();

            if (typeid(typename exec_space::memory_space) != typeid(typename Kokkos::DefaultHostExecutionSpace::memory_space)) {
                //decrease number of mem_chunks to reduce memory usage if necessary
                size_t mem_needed = static_cast<size_t>(mem_chunk_count) * static_cast<size_t>(mem_chunk_size) * sizeof(ordinal_t);
                size_t max_mem_allowed = 536870912;//1073741824;
                if (mem_needed > max_mem_allowed) {
                    size_t chunk_dif = mem_needed - max_mem_allowed;
                    chunk_dif = chunk_dif / (static_cast<size_t>(mem_chunk_size) * sizeof(ordinal_t));
                    chunk_dif++;
                    mem_chunk_count -= chunk_dif;
                }
            }

            uniform_memory_pool_t memory_pool(mem_chunk_count, mem_chunk_size, ORD_MAX, pool_type);

            functorHashmapAccumulator<exec_space, uniform_memory_pool_t>
                hashmapAccumulator(source_bucket_offset, dest_by_source, wgt_by_source, edges_per_source, memory_pool, hash_size, max_entries, remaining);

            ordinal_t old_remaining_count = remaining_count;
            Kokkos::parallel_reduce("hashmap time", policy_t(0, old_remaining_count), hashmapAccumulator, remaining_count);

            if (remaining_count > 0) {
                vtx_view_t new_remaining("new remaining vtx", remaining_count);

                Kokkos::parallel_scan("move remaining vertices", policy_t(0, old_remaining_count), KOKKOS_LAMBDA(const ordinal_t i, ordinal_t & update, const bool final){
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
        Kokkos::parallel_reduce(policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t i, edge_offset_t & sum){
            sum += edges_per_source(i);
        }, gc_nedges);
    }
    else {
        /*
        edge_offset_t total_unduped = 0;
        ordinal_t max_unduped = 0;
        Kokkos::parallel_reduce("find max", n, KOKKOS_LAMBDA(const ordinal_t i, ordinal_t& l_max){
            ordinal_t s = source_bucket_offset(i+1) - source_bucket_offset(i);
            if (l_max <= s) {
                l_max = s;
            }
        }, Kokkos::Max<ordinal_t, Kokkos::HostSpace>(max_unduped));

        Kokkos::parallel_reduce("find total", n, KOKKOS_LAMBDA(const ordinal_t i, edge_offset_t& sum){
            ordinal_t s = source_bucket_offset(i+1) - source_bucket_offset(i);
            sum += s;
        }, total_unduped);

        edge_offset_t avg_unduped = total_unduped / n;

        printf("avg: %u, max: %u, n: %u, nc: %u\n", avg_unduped, max_unduped, n, nc);
        */

        // sort the (implicit) crs matrix
        Kokkos::Timer radix;
        KokkosKernels::Impl::sort_crs_matrix<exec_space, edge_view_t, vtx_view_t, wgt_view_t>(source_bucket_offset, dest_by_source, wgt_by_source);
        experiment.addMeasurement(ExperimentLoggerUtil::Measurement::RadixSort, radix.seconds());
        radix.reset();

        // combine adjacent entries having same destination
        if (use_team) {
            // thread team version
            wgt_view_t wgts_out("wgts after dedupe", wgt_by_source.extent(0));
            vtx_view_t dest_out("dest after dedupe", dest_by_source.extent(0));
            functorDedupeAfterSort deduper(source_bucket_offset, dest_by_source, dest_out, wgt_by_source, wgts_out, edges_per_source);
            Kokkos::parallel_reduce("deduplicated sorted", team_policy_t(n, 64), deduper, gc_nedges);
            Kokkos::deep_copy(wgt_by_source, wgts_out);
            Kokkos::deep_copy(dest_by_source, dest_out);
        } 
        else {
            // no thread team version
            functorDedupeAfterSort deduper(source_bucket_offset, dest_by_source, dest_by_source, wgt_by_source, wgt_by_source, edges_per_source);
            Kokkos::parallel_reduce("deduplicated sorted", policy_t(0, n), deduper, gc_nedges);
        }

        experiment.addMeasurement(ExperimentLoggerUtil::Measurement::RadixDedupe, radix.seconds());
        radix.reset();
    }

}

struct translationFunctor {

    matrix_t vcmap, g;
    vtx_view_t mapped_edges, edges_per_source;
    edge_view_t source_bucket_offset;
    vtx_view_t edges_out;
    wgt_view_t wgts_out;
    ordinal_t workLength;

    translationFunctor(matrix_t vcmap,
            matrix_t g,
            vtx_view_t mapped_edges,
            vtx_view_t edges_per_source,
            edge_view_t source_bucket_offset,
            vtx_view_t edges_out,
            wgt_view_t wgts_out) :
        vcmap(vcmap),
        g(g),
        mapped_edges(mapped_edges),
        edges_per_source(edges_per_source),
        source_bucket_offset(source_bucket_offset),
        workLength(g.numRows()),
        edges_out(edges_out),
        wgts_out(wgts_out) {}

    KOKKOS_INLINE_FUNCTION
        void operator()(const member& t) const 
    {
        ordinal_t i = t.league_rank() * t.team_size() + t.team_rank();
        if(i >= workLength) return;
        ordinal_t u = vcmap.graph.entries(i);
        edge_offset_t start = g.graph.row_map(i);
        edge_offset_t end = g.graph.row_map(i + 1);
        ordinal_t nonLoopEdgesTotal = 0;
        Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(t, start, end), [=] (const edge_offset_t idx, ordinal_t& local_sum) {
            ordinal_t v = mapped_edges(idx);
            if (u != v) {
                edge_offset_t offset = Kokkos::atomic_fetch_add(&edges_per_source(u), 1);

                offset += source_bucket_offset(u);

                edges_out(offset) = v;
                wgts_out(offset) = g.values(idx);
            }
        }, nonLoopEdgesTotal);
    }
};

coarse_level_triple build_nonskew(const matrix_t g,
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

    Kokkos::parallel_scan("calc source offsets", policy_t(0, nc), prefix_sum(edges_per_source, source_bucket_offset));

    Kokkos::parallel_for("reset edges per source", policy_t(0, nc), KOKKOS_LAMBDA(ordinal_t i) {
        edges_per_source(i) = 0; // will use as counter again
    });

    experiment.addMeasurement(ExperimentLoggerUtil::Measurement::Prefix, timer.seconds());
    timer.reset();

    edge_subview_t sbo_subview = Kokkos::subview(source_bucket_offset, nc);
    edge_offset_t size_pre_dedupe = 0;
    Kokkos::deep_copy(size_pre_dedupe, sbo_subview);

    vtx_view_t dest_by_source("dest_by_source", size_pre_dedupe);
    wgt_view_t wgt_by_source("wgt_by_source", size_pre_dedupe);

    auto execSpaceEnum = KokkosKernels::Impl::kk_get_exec_space_type<exec_space>();
    int vectorLength = KokkosKernels::Impl::kk_get_suggested_vector_size(n, g.nnz(), execSpaceEnum);
    translationFunctor translateF(vcmap, g, mapped_edges, edges_per_source, source_bucket_offset, dest_by_source, wgt_by_source);
    team_policy_t dummy(1, 1, vectorLength);
    int teamSize = dummy.team_size_max(translateF, Kokkos::ParallelForTag());
    Kokkos::parallel_for("move edges to coarse matrix", team_policy_t((n + teamSize - 1) / teamSize, teamSize, vectorLength), translateF);

    experiment.addMeasurement(ExperimentLoggerUtil::Measurement::Bucket, timer.seconds());
    timer.reset();

    deduplicate_graph(nc, false,
        edges_per_source, dest_by_source, wgt_by_source,
        source_bucket_offset, experiment, gc_nedges);

    experiment.addMeasurement(ExperimentLoggerUtil::Measurement::Dedupe, timer.seconds());
    timer.reset();

    edge_view_t source_offsets("source_offsets", nc + 1);

    Kokkos::parallel_scan("calc source offsets again", policy_t(0, nc), prefix_sum(edges_per_source, source_offsets));

    edge_subview_t edge_total_subview = Kokkos::subview(source_offsets, nc);
    Kokkos::deep_copy(gc_nedges, edge_total_subview);

    vtx_view_t dest_idx("dest_idx", gc_nedges);
    wgt_view_t wgts("wgts", gc_nedges);

    Kokkos::parallel_for("move deduped edges to new coarse matrix", team_policy_t(nc, Kokkos::AUTO), KOKKOS_LAMBDA(const member & thread) {
        ordinal_t u = thread.league_rank();
        edge_offset_t start_origin = source_bucket_offset(u);
        edge_offset_t start_dest = source_offsets(u);
        Kokkos::parallel_for(Kokkos::TeamThreadRange(thread, edges_per_source(u)), [=](const edge_offset_t idx) {
            dest_idx(start_dest + idx) = dest_by_source(start_origin + idx);
            wgts(start_dest + idx) = wgt_by_source(start_origin + idx);
            });
    });
    experiment.addMeasurement(ExperimentLoggerUtil::Measurement::WriteGraph, timer.seconds());
    timer.reset();

    graph_type gc_graph(dest_idx, source_offsets);
    matrix_t gc("gc", nc, wgts, gc_graph);

    coarse_level_triple next_level;
    next_level.coarse_mtx = gc;
    return next_level;
}

matrix_t collapse_directed_to_undirected(const ordinal_t nc,
    const vtx_view_t source_edge_counts,
    const edge_view_t source_row_map,
    const vtx_view_t source_destinations,
    const wgt_view_t source_wgts) {

    vtx_view_t coarse_degree("coarse degree", nc);
    Kokkos::deep_copy(coarse_degree, source_edge_counts);

    Kokkos::parallel_for("count directed edges owned by opposite endpoint", team_policy_t(nc, Kokkos::AUTO), KOKKOS_LAMBDA(const member & thread) {
        ordinal_t u = thread.league_rank();
        edge_offset_t start = source_row_map(u);
        edge_offset_t end = start + source_edge_counts(u);
        Kokkos::parallel_for(Kokkos::TeamThreadRange(thread, start, end), [=](const edge_offset_t idx) {
            ordinal_t v = source_destinations(idx);
            //increment other vertex
            Kokkos::atomic_fetch_add(&coarse_degree(v), 1);
        });
    });

    edge_view_t target_row_map("target row map", nc + 1);

    Kokkos::parallel_scan("calc target row map", policy_t(0, nc), prefix_sum(coarse_degree, target_row_map));

    Kokkos::parallel_for("reset coarse edges", policy_t(0, nc), KOKKOS_LAMBDA(const ordinal_t i){
        coarse_degree(i) = 0;
    });

    edge_offset_t coarse_edges_total = 0;
    edge_subview_t coarse_edge_total_subview = Kokkos::subview(target_row_map, nc);
    Kokkos::deep_copy(coarse_edges_total, coarse_edge_total_subview);

    vtx_view_t dest_idx("dest_idx", coarse_edges_total);
    wgt_view_t wgts("wgts", coarse_edges_total);

    Kokkos::parallel_for("move edges into correct size matrix", team_policy_t(nc, Kokkos::AUTO), functorCollapseDirectedToUndirected(source_row_map,
        target_row_map, source_edge_counts, coarse_degree, source_destinations, dest_idx, source_wgts, wgts));

    graph_type gc_graph(dest_idx, target_row_map);
    matrix_t gc("gc", nc, wgts, gc_graph);
    return gc;
}

coarse_level_triple build_skew(const matrix_t g,
    const matrix_t vcmap,
    vtx_view_t mapped_edges,
    vtx_view_t degree_initial,
    ExperimentLoggerUtil& experiment,
    Kokkos::Timer& timer) {

    ordinal_t n = g.numRows();
    ordinal_t nc = vcmap.numCols();
    edge_offset_t gc_nedges = 0;

    vtx_view_t edges_per_source("edges_per_source", nc);

    //recount with edges only belonging to coarse vertex of smaller degree
    Kokkos::parallel_for("recount edges", team_policy_t(n, Kokkos::AUTO), KOKKOS_LAMBDA(const member & thread) {
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
            Kokkos::atomic_add(&edges_per_source(u), nonLoopEdgesTotal);
        });
    });

    experiment.addMeasurement(ExperimentLoggerUtil::Measurement::Count, timer.seconds());
    timer.reset();

    edge_view_t source_bucket_offset("source_bucket_offsets", nc + 1);

    Kokkos::parallel_scan("calc source offsets", policy_t(0, nc), prefix_sum(edges_per_source, source_bucket_offset));
    edge_subview_t sbo_subview = Kokkos::subview(source_bucket_offset, nc);
    edge_offset_t size_pre_dedupe = 0;
    Kokkos::deep_copy(size_pre_dedupe, sbo_subview);

    experiment.addMeasurement(ExperimentLoggerUtil::Measurement::Prefix, timer.seconds());
    timer.reset();

    Kokkos::parallel_for("reset edges per source", policy_t(0, nc), KOKKOS_LAMBDA(const ordinal_t i){
        edges_per_source(i) = 0;
    });
    vtx_view_t dest_by_source("dest by source", size_pre_dedupe);
    wgt_view_t wgt_by_source("wgt by source", size_pre_dedupe);
    Kokkos::parallel_for("combine fine rows", team_policy_t(n, Kokkos::AUTO), KOKKOS_LAMBDA(const member & thread) {
        ordinal_t outer_idx = thread.league_rank();
        ordinal_t u = vcmap.graph.entries(outer_idx);
        edge_offset_t start = g.graph.row_map(outer_idx);
        edge_offset_t end = g.graph.row_map(outer_idx + 1);
        Kokkos::parallel_for(Kokkos::TeamThreadRange(thread, start, end), [=](const edge_offset_t idx) {
            ordinal_t v = mapped_edges(idx);
            bool degree_less = degree_initial(u) < degree_initial(v);
            bool degree_equal = degree_initial(u) == degree_initial(v);
            if (degree_less || (degree_equal && u < v)) {
                edge_offset_t offset = Kokkos::atomic_fetch_add(&edges_per_source(u), 1);

                offset += source_bucket_offset(u);

                dest_by_source(offset) = v;
                wgt_by_source(offset) = g.values(idx);
            }
        });
    });
    gc_nedges = 0;

    experiment.addMeasurement(ExperimentLoggerUtil::Measurement::Bucket, timer.seconds());
    timer.reset();

    deduplicate_graph(nc, true,
        edges_per_source, dest_by_source, wgt_by_source,
        source_bucket_offset, experiment, gc_nedges);

    experiment.addMeasurement(ExperimentLoggerUtil::Measurement::Dedupe, timer.seconds());
    timer.reset();

    matrix_t gc = collapse_directed_to_undirected(nc, edges_per_source, source_bucket_offset, dest_by_source, wgt_by_source);

    experiment.addMeasurement(ExperimentLoggerUtil::Measurement::WriteGraph, timer.seconds());
    timer.reset();

    coarse_level_triple next_level;
    next_level.coarse_mtx = gc;
    return next_level;
}

coarse_level_triple build_high_duplicity(const matrix_t g,
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
    Kokkos::parallel_for("recount edges", team_policy_t(n, Kokkos::AUTO), KOKKOS_LAMBDA(const member & thread) {
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

    Kokkos::parallel_scan("calc source offsets", policy_t(0, n), prefix_sum(dedupe_count, row_map_copy));

    Kokkos::parallel_for("reset dedupe count", policy_t(0, n), KOKKOS_LAMBDA(ordinal_t i) {
        dedupe_count(i) = 0; // will use as counter again
    });

    experiment.addMeasurement(ExperimentLoggerUtil::Measurement::Prefix, timer.seconds());
    timer.reset();

    edge_subview_t fine_recount_subview = Kokkos::subview(row_map_copy, n);
    edge_offset_t fine_recount = 0;
    Kokkos::deep_copy(fine_recount, fine_recount_subview);

    vtx_view_t dest_fine("fine to coarse dests", fine_recount);
    wgt_view_t wgt_fine("fine to coarse wgts", fine_recount);

    Kokkos::parallel_for("move edges to new matrix", team_policy_t(n, Kokkos::AUTO), KOKKOS_LAMBDA(const member & thread) {
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
    experiment.addMeasurement(ExperimentLoggerUtil::Measurement::Bucket, timer.seconds());
    timer.reset();
    
    //deduplicate coarse adjacencies within each fine row
    deduplicate_graph(n, true,
        dedupe_count, dest_fine, wgt_fine,
        row_map_copy, experiment, gc_nedges);

    experiment.addMeasurement(ExperimentLoggerUtil::Measurement::Dedupe, timer.seconds());
    timer.reset();

    edge_view_t source_bucket_offset("source_bucket_offsets", nc + 1);
    vtx_view_t edges_per_source("edges_per_source", nc);

    Kokkos::parallel_for("sum fine row sizes", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t i){
        ordinal_t u = vcmap.graph.entries(i);
        Kokkos::atomic_fetch_add(&edges_per_source(u), dedupe_count(i));
    });
    experiment.addMeasurement(ExperimentLoggerUtil::Measurement::Count, timer.seconds());
    timer.reset();
    Kokkos::parallel_scan("calc source offsets", policy_t(0, nc), prefix_sum(edges_per_source, source_bucket_offset));
    experiment.addMeasurement(ExperimentLoggerUtil::Measurement::Prefix, timer.seconds());
    timer.reset();
    Kokkos::parallel_for("reset edges per source", policy_t(0, nc), KOKKOS_LAMBDA(const ordinal_t i){
        edges_per_source(i) = 0;
    });
    vtx_view_t dest_by_source("dest by source", gc_nedges);
    wgt_view_t wgt_by_source("wgt by source", gc_nedges);
    Kokkos::parallel_for("combine deduped fine rows", team_policy_t(n, Kokkos::AUTO), KOKKOS_LAMBDA(const member & thread) {
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

    deduplicate_graph(nc, true,
        edges_per_source, dest_by_source, wgt_by_source,
        source_bucket_offset, experiment, gc_nedges);

    experiment.addMeasurement(ExperimentLoggerUtil::Measurement::Dedupe, timer.seconds());
    timer.reset();

    matrix_t gc = collapse_directed_to_undirected(nc, edges_per_source, source_bucket_offset, dest_by_source, wgt_by_source);

    experiment.addMeasurement(ExperimentLoggerUtil::Measurement::WriteGraph, timer.seconds());
    timer.reset();

    coarse_level_triple next_level;
    next_level.coarse_mtx = gc;
    return next_level;
}

struct countingFunctor {

    matrix_t vcmap, g;
    vtx_view_t mapped_edges, degree_initial;
    vtx_view_t c_vtx_w, f_vtx_w;
    ordinal_t workLength;

    countingFunctor(matrix_t vcmap,
            matrix_t g,
            vtx_view_t mapped_edges,
            vtx_view_t degree_initial,
            vtx_view_t c_vtx_w,
            vtx_view_t f_vtx_w) :
        vcmap(vcmap),
        g(g),
        mapped_edges(mapped_edges),
        degree_initial(degree_initial),
        c_vtx_w(c_vtx_w),
        f_vtx_w(f_vtx_w),
        workLength(g.numRows()) {}

    KOKKOS_INLINE_FUNCTION
        void operator()(const member& t) const 
    {
        ordinal_t i = t.league_rank() * t.team_size() + t.team_rank();
        if(i >= workLength) return;
        ordinal_t u = vcmap.graph.entries(i);
        edge_offset_t start = g.graph.row_map(i);
        edge_offset_t end = g.graph.row_map(i + 1);
        ordinal_t nonLoopEdgesTotal = 0;
        Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(t, start, end), [=] (const edge_offset_t idx, ordinal_t& local_sum) {
            ordinal_t v = vcmap.graph.entries(g.graph.entries(idx));
            mapped_edges(idx) = v;
            if (u != v) {
                local_sum++;
            }
        }, nonLoopEdgesTotal);
        Kokkos::single(Kokkos::PerThread(t), [=]() {
            Kokkos::atomic_add(&degree_initial(u), nonLoopEdgesTotal);
            Kokkos::atomic_add(&c_vtx_w(u), f_vtx_w(i));
        });
    
    }
};

coarse_level_triple build_coarse_graph(const coarse_level_triple level,
    const matrix_t vcmap,
    ExperimentLoggerUtil& experiment) {

    if (b == Spgemm || b == Spgemm_transpose_first) {
        return build_coarse_graph_spgemm(level, vcmap, experiment);
    }

    matrix_t g = level.coarse_mtx;
    ordinal_t n = g.numRows();
    ordinal_t nc = vcmap.numCols();

    //radix sort source vertices, then sort edges
    Kokkos::View<const edge_offset_t, Device> rm_subview = Kokkos::subview(g.graph.row_map, n);
    edge_offset_t size_rm = 0;
    Kokkos::deep_copy(size_rm, rm_subview);
    vtx_view_t mapped_edges("mapped edges", size_rm);

    Kokkos::Timer timer;

    vtx_view_t degree_initial("edges_per_source", nc);
    vtx_view_t f_vtx_w = level.coarse_vtx_wgts;
    vtx_view_t c_vtx_w = vtx_view_t("coarse vertex weights", nc);

    auto execSpaceEnum = KokkosKernels::Impl::kk_get_exec_space_type<exec_space>();
    int vectorLength = KokkosKernels::Impl::kk_get_suggested_vector_size(n, g.nnz(), execSpaceEnum);
    countingFunctor countF(vcmap, g, mapped_edges, degree_initial, c_vtx_w, f_vtx_w);
    team_policy_t dummy(1, 1, vectorLength);
    int teamSize = dummy.team_size_max(countF, Kokkos::ParallelForTag());
    //count edges per vertex
    Kokkos::parallel_for("count edges per coarse vertex (also compute coarse vertex weights)", team_policy_t((n + teamSize - 1) / teamSize, teamSize, vectorLength), countF);

    edge_offset_t total_unduped = 0;
    ordinal_t max_unduped = 0;

    Kokkos::parallel_reduce("find max", policy_t(0, nc), KOKKOS_LAMBDA(const ordinal_t i, ordinal_t& l_max){
        if (l_max <= degree_initial(i)) {
            l_max = degree_initial(i);
        }
    }, Kokkos::Max<ordinal_t, Kokkos::HostSpace>(max_unduped));

    Kokkos::parallel_reduce("find total", policy_t(0, nc), KOKKOS_LAMBDA(const ordinal_t i, edge_offset_t& sum){
        sum += degree_initial(i);
    }, total_unduped);

    edge_offset_t avg_unduped = total_unduped / nc;

    coarse_level_triple next_level;
    //optimized subroutines for sufficiently irregular graphs or high average adjacency rows
    //don't do optimizations if running on CPU (the default host space)
    if(avg_unduped > (nc/4) && typeid(typename exec_space::memory_space) != typeid(typename Kokkos::DefaultHostExecutionSpace::memory_space)){
        next_level = build_high_duplicity(g, vcmap, mapped_edges, degree_initial, experiment, timer);
    } else if (avg_unduped > 50 && (max_unduped / 10) > avg_unduped && typeid(typename exec_space::memory_space) != typeid(typename Kokkos::DefaultHostExecutionSpace::memory_space)) {
        next_level = build_skew(g, vcmap, mapped_edges, degree_initial, experiment, timer);
    } else {
        next_level = build_nonskew(g, vcmap, mapped_edges, degree_initial, experiment, timer);
    }

    next_level.coarse_vtx_wgts = c_vtx_w;
    next_level.level = level.level + 1;
    next_level.interp_mtx = vcmap;
    next_level.uniform_weights = false;
    return next_level;
}

graph_type coarsen_de_bruijn_graph(graph_type g, matrix_t interp){
    ordinal_t n = g.numRows();
    ordinal_t nc = interp.numCols();
    //vtx 0 maps vertices with no edges, which we don't care about
    //we remove vtx 0 from the graph
    edge_view_t edge_count("edge count", nc - 1);
    Kokkos::parallel_for("count edges", n, KOKKOS_LAMBDA(const ordinal_t i){
        ordinal_t u = interp.graph.entries(i);
        if(u > 0 && (g.row_map(i + 1) - g.row_map(i) > 0)){
            ordinal_t f = g.entries(g.row_map(i));
            ordinal_t v = interp.graph.entries(f);
            if(u != v){
                //only two possible edges for each u
                //one in and one out
                //don't care which is which
                //shift vtx id down by 1
                Kokkos::atomic_increment(&edge_count(u - 1));
            }
        }
    });
    edge_view_t row_map("row map", nc);
    Kokkos::parallel_scan("calc source offsets", policy_t(0, nc - 1), prefix_sum(edge_count, row_map));
    edge_subview_t rm_subview = Kokkos::subview(row_map, nc - 1);
    edge_offset_t total_edges = 0;
    Kokkos::deep_copy(total_edges, rm_subview);
    vtx_view_t entries("entries", total_edges);
    Kokkos::parallel_for("reset edge count", nc - 1, KOKKOS_LAMBDA(const ordinal_t i){
        edge_count(i) = 0;
    });
    Kokkos::parallel_for("write edges", n, KOKKOS_LAMBDA(const ordinal_t i){
        ordinal_t u = interp.graph.entries(i);
        if(u > 0 && (g.row_map(i + 1) - g.row_map(i) > 0)){
            ordinal_t f = g.entries(g.row_map(i));
            ordinal_t v = interp.graph.entries(f);
            if(u != v){
                //shift vtx ids down by 1
                edge_offset_t insert = row_map(u - 1) + Kokkos::atomic_fetch_add(&edge_count(u - 1), 1);
                entries(row_map(u - 1)) = v - 1;
            }
        }
    });
    graph_type gc(entries, row_map);
    return gc;
}

//remove all out-edges for any vertex with more than 1 out-edge
//remove all in-edges for any vertex with more than 1 in-edge
//combine remaining in and out edges into one graph
graph_type prune_edges(graph_type g1, graph_type g2){
    //g1 contains in edges, g2 contains out edges
    ordinal_t n = g1.numRows();
    ordinal_t total_paths = 0;
    edge_view_t row_map("pruned row map", n+1);
    edge_view_t edge_count("edge count", n);
    Kokkos::parallel_scan("count path edges", n, KOKKOS_LAMBDA(const ordinal_t i){
        //vertex i has exactly 1 out edge
        if(g2.row_map(i + 1) - g2.row_map(i) == 1){
            //id of the "in" vertex of the edge
            ordinal_t v = g2.entries(g2.row_map(i));
            //vertex v has exactly 1 in edge
            if(g1.row_map(v + 1) - g1.row_map(v) == 1){
                Kokkos::atomic_increment(&edge_count(i));
                Kokkos::atomic_increment(&edge_count(v));
            }
        }
    });
    Kokkos::parallel_scan("calc source offsets", policy_t(0, n), prefix_sum(edge_count, row_map));
    edge_subview_t rm_subview = Kokkos::subview(row_map, n);
    edge_offset_t total_paths = 0;
    Kokkos::deep_copy(total_paths, rm_subview);
    vtx_view_t entries("pruned out entries", total_paths);
    Kokkos::parallel_for("write path edges", n, KOKKOS_LAMBDA(const ordinal_t i){
        //vertex i has exactly 1 out edge
        if(g2.row_map(i + 1) - g2.row_map(i) == 1){
            //id of the "in" vertex of the edge
            ordinal_t v = g2.entries(g2.row_map(i));
            //vertex v has exactly 1 in edge
            if(g1.row_map(v + 1) - g1.row_map(v) == 1){
                //out edge
                entries(row_map(i)) = v;
                //in edge
                entries(row_map(v + 1) - 1) = i;
            }
        }
    });
    graph_type pruned(entries, row_map);
    return pruned;
}

matrix_t generate_coarse_mapping(const matrix_t g,
    bool uniform_weights,
    ExperimentLoggerUtil& experiment) {

    Kokkos::Timer timer;
    matrix_t interpolation_graph;
    int choice = 0;

    switch (h) {
        case HECv1:
            choice = 0;
            break;
        case HECv2:
            choice = 1;
            break;
        case HECv3:
            choice = 2;
            break;
        case Match:
            choice = 0;
            break;
        case MtMetis:
            choice = 1;
            break;
    }

    switch (h) {
        case HECv1:
        case HECv2:
        case HECv3:
            interpolation_graph = mapper.sgp_coarsen_HEC(g, uniform_weights, experiment, choice);
            break;
        case Match:
        case MtMetis:
            interpolation_graph = mapper.sgp_coarsen_match(g, uniform_weights, experiment, choice);
            break;
        case MIS2:
            interpolation_graph = mapper.sgp_coarsen_mis_2(g, experiment);
            break;
        case GOSHv2:
            interpolation_graph = mapper.sgp_coarsen_GOSH_v2(g, experiment);
            break;
        case GOSHv1:
            interpolation_graph = mapper.sgp_coarsen_GOSH(g, experiment);
            break;
    }
    experiment.addMeasurement(ExperimentLoggerUtil::Measurement::Map, timer.seconds());
    return interpolation_graph;
}

void coarsen_de_bruijn_full_cycle(const graph_type g, ExperimentLoggerUtil& experiment){
    {
        edge_view_t row_map;
        vtx_view_t entries;
        KokkosKernels::Impl::transpose_graph
        <typename graph_type::row_map_type, vtx_view_t, edge_view_t, vtx_view_t, edge_view_t, exec_space>
        (g.numRows(), g.numCols(), g.row_map, g.entries, row_map, entries);
        graph_type transposed(entries, row_map);
        g = prune_edges(g, transposed);
    }
    std::list<graph_type> levels;
    std::list<matrix_t> level_interp;
    levels.push_back(g);
    while(levels.rbegin()->numRows() > 1){
        matrix_t interp = mapper.sgp_coarsen_HEC(g, experiment);
        graph_type next = coarsen_de_bruijn_graph(*levels.rbegin(), interp);
        levels.push_back(next);
        level_interp.push_back(interp);
    }
}

//we can support weighted vertices pretty easily
//this function can't return the generated list directly because of an NVCC compiler bug
//caller must use the get_levels() method after calling this function
void generate_coarse_graphs(const matrix_t fine_g, ExperimentLoggerUtil& experiment, bool uniform_weights = false) {

    Kokkos::Timer timer;
    ordinal_t fine_n = fine_g.numRows();
    std::list<coarse_level_triple>& levels = results;
    levels.clear();
    coarse_level_triple finest;
    finest.coarse_mtx = fine_g;
    //1-indexed, not zero indexed
    finest.level = 1;
    finest.uniform_weights = uniform_weights;
    vtx_view_t vtx_weights("vertex weights", fine_n);
    Kokkos::parallel_for("generate vertex weights", policy_t(0, fine_n), KOKKOS_LAMBDA(const ordinal_t i){
        vtx_weights(i) = 1;
    });
    finest.coarse_vtx_wgts = vtx_weights;
    levels.push_back(finest);
    printf("Fine graph copy to device time: %.8f\n", timer.seconds());
    while (levels.rbegin()->coarse_mtx.numRows() > coarse_vtx_cutoff) {
        printf("Calculating coarse graph %ld\n", levels.size());

        coarse_level_triple current_level = *levels.rbegin();

        matrix_t interp_graph = generate_coarse_mapping(current_level.coarse_mtx, current_level.uniform_weights, experiment);

        if (interp_graph.numCols() < min_allowed_vtx) {
            break;
        }

        timer.reset();
        coarse_level_triple next_level = build_coarse_graph(current_level, interp_graph, experiment);
        experiment.addMeasurement(ExperimentLoggerUtil::Measurement::Build, timer.seconds());
        timer.reset();

        levels.push_back(next_level);

        if(levels.size() > max_levels) break;
#ifdef DEBUG
        double coarsen_ratio = (double) levels.rbegin()->coarse_mtx.numRows() / (double) (++levels.rbegin())->coarse_mtx.numRows();
        printf("Coarsening ratio: %.8f\n", coarsen_ratio);
#endif
    }

}

std::list<coarse_level_triple> get_levels() {
    //"results" is copied, therefore the list received by the caller is independent of the internal list of this class
    return results;
}

void set_heuristic(Heuristic h) {
    this->h = h;
}

void set_deduplication_method(Builder b) {
    this->b = b;
}

void set_coarse_vtx_cutoff(ordinal_t coarse_vtx_cutoff) {
    this->coarse_vtx_cutoff = coarse_vtx_cutoff;
}

void set_min_allowed_vtx(ordinal_t min_allowed_vtx) {
    this->min_allowed_vtx = min_allowed_vtx;
}

void set_max_levels(unsigned int max_levels) {
    this->max_levels = max_levels;
}

};