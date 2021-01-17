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
    using exec_space = typename Device::execution_space;
    using matrix_t = typename KokkosSparse::CrsMatrix<scalar_t, ordinal_t, Device, void, edge_offset_t>;
    using vtx_view_t = typename Kokkos::View<ordinal_t*, Device>;
    using wgt_view_t = typename Kokkos::View<scalar_t*, Device>;
    using edge_view_t = typename Kokkos::View<edge_offset_t*, Device>;
    using edge_subview_t = typename Kokkos::View<edge_offset_t, Device>;
    using graph_type = typename matrix_t::staticcrsgraph_type;
    using policy_t = typename Kokkos::RangePolicy<exec_space>;
    using team_policy_t = typename Kokkos::TeamPolicy<exec_space>;
    using member = typename team_policy_t::member_type;
    static constexpr ordinal_t ORD_MAX = std::numeric_limits<ordinal_t>::max();

    // contains matrix and vertex weights corresponding to current level
    // interp matrix maps previous level to this level
    struct coarse_level_triple {
        matrix_t coarse_mtx;
        vtx_view_t coarse_vtx_wgts;
        matrix_t interp_mtx;
        int level;
        bool uniform_weights, valid;
    };

    enum Heuristic { HECv1, HECv2, HECv3, Match, MtMetis, MIS2, GOSH, GOSHv2 };

    bool use_hashmap = false;
    // default heuristic is HEC
    Heuristic h = HECv1;
    coarsen_heuristics<ordinal_t, edge_offset_t, scalar_t, Device> mapper;
    //when the results are fetched, this list is implicitly copied
    std::list<coarse_level_triple> results;

struct functorDedupeAfterSort
{

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

        //sgp_vid_t insert_at = row_map(idx);

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

    if (use_hashmap) {
        ordinal_t remaining_count = n;
        vtx_view_t remaining("remaining vtx", n);
        Kokkos::parallel_for(policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t i){
            remaining(i) = i;
        });
        do {
            //determine size for hashmap
            ordinal_t avg_entries = 0;
            if (typeid(exec_space::memory_space) != typeid(Kokkos::DefaultHostExecutionSpace::memory_space) && static_cast<double>(remaining_count) / static_cast<double>(n) > 0.01) {
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

            if (typeid(exec_space::memory_space) == typeid(Kokkos::DefaultHostExecutionSpace::memory_space)) {
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

            if (typeid(exec_space::memory_space) != typeid(Kokkos::DefaultHostExecutionSpace::memory_space)) {
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
            Kokkos::parallel_reduce("deduplicated sorted", team_policy_t(n, Kokkos::AUTO), deduper, gc_nedges);
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

    Kokkos::parallel_scan("calc source offsets", policy_t(0, nc), KOKKOS_LAMBDA(const ordinal_t i,
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

    Kokkos::parallel_for("move edges to coarse matrix", team_policy_t(n, Kokkos::AUTO), KOKKOS_LAMBDA(const member & thread) {
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

    deduplicate_graph(nc, false,
        edges_per_source, dest_by_source, wgt_by_source,
        source_bucket_offset, experiment, gc_nedges);

    experiment.addMeasurement(ExperimentLoggerUtil::Measurement::Dedupe, timer.seconds());
    timer.reset();

    edge_view_t source_offsets("source_offsets", nc + 1);

    Kokkos::parallel_scan("calc source offsets again", policy_t(0, nc), KOKKOS_LAMBDA(const ordinal_t i,
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

    graph_type gc_graph(dest_idx, source_offsets);
    matrix_t gc("gc", nc, wgts, gc_graph);

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

    Kokkos::parallel_scan("calc source offsets", policy_t(0, nc), KOKKOS_LAMBDA(const ordinal_t i,
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

    //reused degree initial as degree final
    vtx_view_t degree_final = degree_initial;
    Kokkos::parallel_for("init space needed for each row", policy_t(0, nc), KOKKOS_LAMBDA(const ordinal_t i){
        degree_final(i) = edges_per_source(i);
    });

    Kokkos::parallel_for("each edge must be counted twice, once for each end", team_policy_t(nc, Kokkos::AUTO), KOKKOS_LAMBDA(const member & thread) {
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

    Kokkos::parallel_scan("allocate location for each row", policy_t(0, nc), KOKKOS_LAMBDA(const ordinal_t i,
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

    edge_subview_t edge_total_subview = Kokkos::subview(source_offsets, nc);
    Kokkos::deep_copy(gc_nedges, edge_total_subview);

    vtx_view_t dest_idx("dest_idx", gc_nedges);
    wgt_view_t wgts("wgts", gc_nedges);

    Kokkos::parallel_for("move edges into correct size matrix", team_policy_t(nc, Kokkos::AUTO), KOKKOS_LAMBDA(const member & thread) {
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
    matrix_t gc("gc", nc, wgts, gc_graph);

    experiment.addMeasurement(ExperimentLoggerUtil::Measurement::WriteGraph, timer.seconds());
    timer.reset();

    coarse_level_triple next_level;
    next_level.coarse_mtx = gc;
    return next_level;
}

coarse_level_triple sgp_build_very_skew(const matrix_t g,
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

    Kokkos::parallel_scan("calc source offsets", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t i,
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
    
    //deduplicate coarse adjacencies within each fine row
    deduplicate_graph(n, true,
        dedupe_count, dest_fine, wgt_fine,
        row_map_copy, experiment, gc_nedges);

    edge_view_t source_bucket_offset("source_bucket_offsets", nc + 1);
    vtx_view_t edges_per_source("edges_per_source", nc);

    Kokkos::parallel_for("sum fine row sizes", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t i){
        ordinal_t u = vcmap.graph.entries(i);
        Kokkos::atomic_fetch_add(&edges_per_source(u), dedupe_count(i));
    });
    Kokkos::parallel_scan("calc source offsets", policy_t(0, nc), KOKKOS_LAMBDA(const ordinal_t i,
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

    //reused degree initial as degree final
    vtx_view_t degree_final = degree_initial;
    //use Kokkos::deep_copy here instead?
    Kokkos::parallel_for("copy edges per source to degree final", policy_t(0, nc), KOKKOS_LAMBDA(const ordinal_t i){
        degree_final(i) = edges_per_source(i);
    });

    Kokkos::parallel_for("count space needed for deduped matrix", team_policy_t(nc, Kokkos::AUTO), KOKKOS_LAMBDA(const member & thread) {
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

    Kokkos::parallel_scan("calc source offsets again", policy_t(0, nc), KOKKOS_LAMBDA(const ordinal_t i,
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

    edge_subview_t edge_total_subview = Kokkos::subview(source_offsets, nc);
    Kokkos::deep_copy(gc_nedges, edge_total_subview);

    vtx_view_t dest_idx("dest_idx", gc_nedges);
    wgt_view_t wgts("wgts", gc_nedges);

    Kokkos::parallel_for("move deduped edges into correct size matrix", team_policy_t(nc, Kokkos::AUTO), KOKKOS_LAMBDA(const member & thread) {
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
    matrix_t gc("gc", nc, wgts, gc_graph);

    experiment.addMeasurement(ExperimentLoggerUtil::Measurement::WriteGraph, timer.seconds());
    timer.reset();

    coarse_level_triple next_level;
    next_level.coarse_mtx = gc;
    return next_level;
}

coarse_level_triple build_coarse_graph(const coarse_level_triple level,
    const matrix_t vcmap,
    ExperimentLoggerUtil& experiment) {

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

    //count edges per vertex
    Kokkos::parallel_for("count edges per coarse vertex (also compute coarse vertex weights)", team_policy_t(n, Kokkos::AUTO), KOKKOS_LAMBDA(const member& thread) {
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

    Kokkos::parallel_reduce("find max", policy_t(0, nc), KOKKOS_LAMBDA(const ordinal_t i, ordinal_t& l_max){
        if (l_max <= degree_initial(i)) {
            l_max = degree_initial(i);
        }
    }, Kokkos::Max<ordinal_t, Kokkos::HostSpace>(max_unduped));

    Kokkos::parallel_reduce("find total", policy_t(-, nc), KOKKOS_LAMBDA(const ordinal_t i, edge_offset_t& sum){
        sum += degree_initial(i);
    }, total_unduped);

    edge_offset_t avg_unduped = total_unduped / nc;

    coarse_level_triple next_level;
    //optimized subroutines for sufficiently irregular graphs or high average adjacency rows
    //don't do optimizations if running on CPU (the default host space)
    if(avg_unduped*2 > nc && typeid(Kokkos::DefaultExecutionSpace::memory_space) != typeid(Kokkos::DefaultHostExecutionSpace::memory_space)){
        next_level = sgp_build_very_skew(g, vcmap, mapped_edges, degree_initial, experiment, timer);
    } else if (avg_unduped > 50 && (max_unduped / 10) > avg_unduped && typeid(Kokkos::DefaultExecutionSpace::memory_space) != typeid(Kokkos::DefaultHostExecutionSpace::memory_space)) {
        next_level = sgp_build_skew(g, vcmap, mapped_edges, degree_initial, experiment, timer);
    } else {
        next_level = sgp_build_nonskew(g, vcmap, mapped_edges, degree_initial, experiment, timer);
    }

    next_level.coarse_vtx_wgts = c_vtx_w;
    next_level.level = level.level + 1;
    next_level.interp_mtx = vcmap;
    next_level.uniform_weights = false;
    next_level.valid = true;
    return next_level;
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
        case GOSH:
            interpolation_graph = mapper.sgp_coarsen_GOSH(g, experiment);
            break;
    }
    experiment.addMeasurement(ExperimentLoggerUtil::Measurement::Map, timer.seconds());
    return interpolation_graph;
}

//we can support weighted vertices pretty easily
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
    while (levels.rbegin()->coarse_mtx.numRows() > SGPAR_COARSENING_VTX_CUTOFF) {
        printf("Calculating coarse graph %ld\n", levels.size());

        coarse_level_triple current_level = *levels.rbegin();

        matrix_t interp_graph = generate_coarse_mapping(current_level.coarse_mtx, current_level.uniform_weights, experiment);

        if (interp_graph.numCols() < 10) {
            break;
        }

        timer.reset();
        coarse_level_triple next_level = build_coarse_graph(current_level, interp_graph, experiment);
        experiment.addMeasurement(ExperimentLoggerUtil::Measurement::Build, timer.seconds());
        timer.reset();

        levels.push_back(next_level);

        if(levels.size() > 200) break;
#ifdef DEBUG
        double coarsen_ratio = (double) levels.rbegin()->coarse_mtx.numRows() / (double) (++levels.rbegin())->coarse_mtx.numRows();
        printf("Coarsening ratio: %.8f\n", coarsen_ratio);
#endif
    }

    //don't use the coarsest level if it has too few vertices
    if (levels.rbegin()->coarse_mtx.numRows() < 10) {
        levels.pop_back();
    }

}

std::list<coarse_level_triple> get_levels() {
    //"results" is copied, therefore the list received by the caller is independent of the internal list of this class
    return results;
}

void set_heuristic(Heuristic h) {
    this->h = h;
}

void set_deduplication_method(bool use_hashmap) {
    this->use_hashmap = use_hashmap;
}

};
