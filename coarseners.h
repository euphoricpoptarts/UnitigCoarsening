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
#include "heuristics.h"
#include "compact_graph.h"

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
    using c_edge_subview_t = Kokkos::View<const edge_offset_t, Device>;
    using vtx_subview_t = Kokkos::View<ordinal_t, Device>;
    using graph_t = typename matrix_t::staticcrsgraph_type;
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
    compact_graph<ordinal_t, edge_offset_t, scalar_t, Device> compacter;
    using interp_t = typename coarsen_heuristics<ordinal_t, edge_offset_t, scalar_t, Device>::interp_t;
    using canon_graph = typename coarsen_heuristics<ordinal_t, edge_offset_t, scalar_t, Device>::canon_graph;
    //when the results are fetched, this list is implicitly copied
    std::list<coarse_level_triple> results;
    ordinal_t coarse_vtx_cutoff = 50;
    ordinal_t min_allowed_vtx = 10;
    unsigned int max_levels = 200;

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

canon_graph coarsen_de_bruijn_graph(canon_graph g, interp_t interp, graph_t glue){
    ordinal_t nc = glue.numRows();
    vtx_view_t g1("right entries", nc);
    vtx_view_t g2("left entries", nc);
    Kokkos::parallel_for("init entries", nc, KOKKOS_LAMBDA(const ordinal_t i){
        g1(i) = ORD_MAX;
        g2(i) = ORD_MAX;
    });
    Kokkos::parallel_for("write edges", nc, KOKKOS_LAMBDA(const ordinal_t i){
        edge_offset_t end = glue.row_map(i + 1) - 1;
        ordinal_t u = abs(glue.entries(end)) - 1;
        if(g.right_edges(u) != ORD_MAX && interp.entries(g.right_edges(u)) != i + 1){
           g1(i) = interp.entries(g.right_edges(u)) - 1;
        } else if(g.left_edges(u) != ORD_MAX && interp.entries(g.left_edges(u)) != i + 1){
           g1(i) = interp.entries(g.left_edges(u)) - 1;
        }
    });
    Kokkos::parallel_for("write edges", nc, KOKKOS_LAMBDA(const ordinal_t i){
        edge_offset_t start = glue.row_map(i);
        ordinal_t u = abs(glue.entries(start)) - 1;
        if(g.right_edges(u) != ORD_MAX && interp.entries(g.right_edges(u)) != i + 1){
           g2(i) = interp.entries(g.right_edges(u)) - 1;
        } else if(g.left_edges(u) != ORD_MAX && interp.entries(g.left_edges(u)) != i + 1){
           g2(i) = interp.entries(g.left_edges(u)) - 1;
        }
    });
    canon_graph gn;
    gn.right_edges = g1;
    gn.left_edges = g2;
    gn.size = nc;
    return gn;
}

vtx_view_t transpose_null(interp_t g){
    ordinal_t n = g.n;
    ordinal_t null_aggregate_size = 0;
    Kokkos::parallel_reduce("count null aggregate vtx", n, KOKKOS_LAMBDA(const ordinal_t i, ordinal_t& sum){
        ordinal_t v = g.entries(i);
        if(v == 0){
            sum++;
        }
    }, null_aggregate_size);
    vtx_view_t nulls("null aggregate fine vertices", null_aggregate_size);
    Kokkos::parallel_scan("write null aggregate fine vertices", n, KOKKOS_LAMBDA(const ordinal_t i, ordinal_t& update, const bool final){
        ordinal_t v = g.entries(i);
        if(v == 0){
            if(final){
                edge_offset_t insert = update;
                nulls(insert) = i;
            }
            update++;
        }
    });
    return nulls;
}

//transposes the interpolation matrix
graph_t transpose_non_null(interp_t g){
    ordinal_t n = g.n;
    //-1 cuz not counting null aggregate
    ordinal_t nc = g.nc - 1;
    edge_view_t row_map("pruned row map", nc+1);
    vtx_view_t edge_count("edge count", nc);
    Kokkos::parallel_for("count transpose edges", n, KOKKOS_LAMBDA(const ordinal_t i){
        ordinal_t v = g.entries(i);
        //a lot of vertices belong to the null aggregate, we want to process it separately
        if(v != 0){
            //-1 cuz not counting null aggregate
            Kokkos::atomic_increment(&edge_count(v - 1));
        }
    });
    Kokkos::parallel_scan("calc source offsets", policy_t(0, nc), prefix_sum(edge_count, row_map));
    edge_subview_t rm_subview = Kokkos::subview(row_map, nc);
    edge_offset_t total_e = 0;
    Kokkos::deep_copy(total_e, rm_subview);
    Kokkos::parallel_for("reset edge count", nc, KOKKOS_LAMBDA(const ordinal_t i){
        edge_count(i) = 0;
    });
    vtx_view_t entries("out entries", total_e);
    Kokkos::parallel_for("write transpose edges", n, KOKKOS_LAMBDA(const ordinal_t i){
        ordinal_t v = g.entries(i);
        if(v != 0){
            edge_offset_t insert = row_map(v - 1) + Kokkos::atomic_fetch_add(&edge_count(v - 1), 1);
            entries(insert) = i;
        }
    });
    graph_t transposed(entries, row_map);
    return transposed;
}

graph_t transpose_and_sort(interp_t interp, canon_graph g){
    graph_t interp_transpose = transpose_non_null(interp);
    ordinal_t nc = interp_transpose.numRows();
    printf("nc: %u\n", nc);
    Kokkos::Timer timer;
    Kokkos::parallel_for("sort tranpose entries", policy_t(0, nc), KOKKOS_LAMBDA(const ordinal_t i){
        //sort entries where g is a bi-directed graph
        ordinal_t vtx_id = i + 1;
        edge_offset_t start = interp_transpose.row_map(i);
        edge_offset_t end = interp_transpose.row_map(i + 1);
        ordinal_t start_vertex = ORD_MAX;
        ordinal_t next = ORD_MAX;
        //find the first vertex in the ordering
        for(edge_offset_t x = start; x < end; x++){
            ordinal_t u = interp_transpose.entries(x);
            if(g.right_edges(u) == ORD_MAX || g.left_edges(u) == ORD_MAX){
                //first vertex in path
                //only two fine vertices in a coarse vertex can satisfy this condition
                interp_transpose.entries(x) = interp_transpose.entries(start);
                interp_transpose.entries(start) = u + 1;
                start_vertex = u;
                next = g.right_edges(u);
                if(next == ORD_MAX){
                    next = g.left_edges(u);
                    interp_transpose.entries(start) = -u - 1;
                }
                break;
            } else {
                ordinal_t v = g.right_edges(u);
                if(interp.entries(v) != vtx_id){
                    //first vertex in path contained in this coarse vertex
                    //only two fine vertice in a coarse vertex can satisfy either this or the previous condition
                    interp_transpose.entries(x) = interp_transpose.entries(start);
                    interp_transpose.entries(start) = -u - 1;
                    start_vertex = u;
                    next = g.left_edges(u);
                    break;
                }
                v = g.left_edges(u);
                if(interp.entries(v) != vtx_id){
                    interp_transpose.entries(x) = interp_transpose.entries(start);
                    interp_transpose.entries(start) = u + 1;
                    start_vertex = u;
                    next = g.right_edges(u);
                    break;
                }
            }
        }
        edge_offset_t real_start = start;
        start++;
        if(start_vertex == ORD_MAX){
            //printf("error for row of length: %lu at %lu\n", end - real_start, real_start);
            //for(edge_offset_t x = real_start; x < end; x++){
            //    ordinal_t u = interp_transpose.entries(x);
            //    printf("entry at %lu: %u; right edge: %u; left edge: %u\n", x, u, g.right_edges(u), g.left_edges(u));
            //}
            start_vertex = interp_transpose.entries(real_start);
            interp_transpose.entries(real_start) = start_vertex + 1;
            next = g.right_edges(start_vertex);
        }
        while(end > start){
            if(g.right_edges(next) == start_vertex){
                //next edge exists from left
                interp_transpose.entries(start) = -next - 1;
                start_vertex = next;
                next = g.left_edges(next);
            } else {
                //next edge exits from right
                interp_transpose.entries(start) = next + 1;
                start_vertex = next;
                next = g.right_edges(next);
            }
            start++;
        }
    });
    printf("Time to sort transposed entries: %.3f\n", timer.seconds());
    timer.reset();
    return interp_transpose;
}

graph_t collect_outputs_first(interp_t interp) {
    vtx_view_t entries = transpose_null(interp);
    edge_offset_t size = entries.extent(0);
    edge_view_t row_map("row map", size + 1);
    Kokkos::parallel_for("init write sizes", policy_t(0, size + 1), KOKKOS_LAMBDA(const edge_offset_t i){
        //output function interprets source kmer as abs(x) - 1
        //so we need to adjust the singleton unitigs for that
        entries(i) = entries(i) + 1;
        row_map(i) = i;
    });
    graph_t output_g(entries, row_map);
    return output_g;
}

std::list<graph_t> coarsen_de_bruijn_full_cycle(canon_graph cur, ExperimentLoggerUtil& experiment){
    std::list<graph_t> glue_list;
    int count = 0;
    Kokkos::Timer timer;
    bool first = true;
    graph_t glue_last;
    while(cur.size > 0){
        count++;
        printf("Calculating coarse graph %d\n", count);
        printf("input vertices: %lu\n", cur.size);
        timer.reset();
        interp_t interp = mapper.coarsen_HEC(cur, experiment);
        experiment.addMeasurement(ExperimentLoggerUtil::Measurement::Map, timer.seconds());
        timer.reset();
        graph_t glue = transpose_and_sort(interp, cur);
        experiment.addMeasurement(ExperimentLoggerUtil::Measurement::InterpTranspose, timer.seconds());
        timer.reset();
        if(first){
            glue_list.push_back(collect_outputs_first(interp));
            glue_last = glue; 
        } else {
            vtx_view_t nulls = transpose_null(interp);
            glue_list.push_back(compacter.collect_outputs(glue_last, nulls));
            glue_last = compacter.collect_unitigs(glue_last, glue);
        }
        first = false;
        experiment.addMeasurement(ExperimentLoggerUtil::Measurement::CompactGlues, timer.seconds());
        timer.reset();
        cur = coarsen_de_bruijn_graph(cur, interp, glue);
        experiment.addMeasurement(ExperimentLoggerUtil::Measurement::Build, timer.seconds());
        timer.reset();
    }
    return glue_list;
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
