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
    compact_graph<ordinal_t, edge_offset_t, scalar_t, Device> compacter;
    using interp_t = typename coarsen_heuristics<ordinal_t, edge_offset_t, scalar_t, Device>::interp_t;
    //when the results are fetched, this list is implicitly copied
    std::list<coarse_level_triple> results;
    ordinal_t coarse_vtx_cutoff = 50;
    ordinal_t min_allowed_vtx = 10;
    unsigned int max_levels = 200;

struct crosses {
    vtx_view_t in;
    vtx_view_t out;
};

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

vtx_view_t coarsen_de_bruijn_graph(vtx_view_t g, interp_t interp){
    ordinal_t n = g.extent(0);
    //-1 so as not to count the null aggregate
    ordinal_t nc = interp.nc - 1;
    vtx_view_t entries("entries", nc);
    Kokkos::parallel_for("init entries", nc, KOKKOS_LAMBDA(const ordinal_t i){
        entries(i) = ORD_MAX;
    });
    Kokkos::parallel_for("write edges", n, KOKKOS_LAMBDA(const ordinal_t i){
        ordinal_t u = interp.entries(i);
        if(u > 0 && u != ORD_MAX){
            //u is not the null aggregate
            ordinal_t f = g(i);
            if(f != ORD_MAX){
                //f is a real edge
                //v can't be the null aggregate if an edge points to it
                ordinal_t v = interp.entries(f);
                if(u != v){
                    entries(u - 1) = v - 1;
                }
            }
        }
    });
    return entries;
}

//remove all out-edges for any vertex with more than 1 out-edge
//remove all in-edges for any vertex with more than 1 in-edge
//output graph contains only out-edges
//index is out vertex, value is in vertex
vtx_view_t prune_edges(graph_type g){
    ordinal_t n = g.numRows();
    vtx_view_t edge_count("in edge count", n);
    Kokkos::parallel_for("count path edges", n, KOKKOS_LAMBDA(const ordinal_t i){
        for(edge_offset_t j = g.row_map(i); j < g.row_map(i + 1); j++){
            ordinal_t v = g.entries(j);
            //count in edge for v
            Kokkos::atomic_increment(&edge_count(v));
        }
    });
    vtx_view_t entries("pruned out entries", n);
    Kokkos::parallel_for("init entries", n, KOKKOS_LAMBDA(const ordinal_t i){
        entries(i) = ORD_MAX;
    });
    Kokkos::parallel_for("write path edges", n, KOKKOS_LAMBDA(const ordinal_t i){
        //i must have 1 out edge
        if(g.row_map(i + 1) - g.row_map(i) == 1){
            ordinal_t v = g.entries(g.row_map(i));
            //vertex v has exactly 1 in edge
            if(edge_count(v) == 1){
                entries(i) = v;
            }
        }
    });
    return entries;
}

vtx_view_t transpose_null(interp_t g, ordinal_t null_id){
    ordinal_t n = g.n;
    ordinal_t null_aggregate_size = 0;
    Kokkos::parallel_reduce("count null aggregate vtx", n, KOKKOS_LAMBDA(const ordinal_t i, ordinal_t& sum){
        ordinal_t v = g.entries(i);
        if(v == null_id){
            sum++;
        }
    }, null_aggregate_size);
    vtx_view_t nulls("null aggregate fine vertices", null_aggregate_size);
    Kokkos::parallel_scan("write null aggregate fine vertices", n, KOKKOS_LAMBDA(const ordinal_t i, ordinal_t& update, const bool final){
        ordinal_t v = g.entries(i);
        if(v == null_id){
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
graph_type transpose_non_null(interp_t g){
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
    graph_type transposed(entries, row_map);
    return transposed;
}

graph_type transpose_and_sort(interp_t interp, vtx_view_t g){
    graph_type interp_transpose = transpose_non_null(interp);
    ordinal_t nc = interp_transpose.numRows();
    Kokkos::Timer timer;
    Kokkos::parallel_for("sort tranpose entries", policy_t(0, nc), KOKKOS_LAMBDA(const ordinal_t i){
        //bubble-sort entries where g is a directed acyclic graph
        ordinal_t vtx_id = i + 1;
        edge_offset_t start = interp_transpose.row_map(i);
        edge_offset_t end = interp_transpose.row_map(i + 1);
        ordinal_t end_vertex = ORD_MAX;
        //find the last vertex in the ordering
        for(edge_offset_t x = start; x < end; x++){
            ordinal_t u = interp_transpose.entries(x);
            if(g(u) == ORD_MAX){
                //last vertex in path
                //only one fine vertex in a coarse vertex can satisfy this condition
                interp_transpose.entries(x) = interp_transpose.entries(end - 1);
                interp_transpose.entries(end - 1) = u;
                end_vertex = u;
                break;
            } else {
                ordinal_t v = g(u);
                if(interp.entries(v) != vtx_id){
                    //last vertex in path contained in this coarse vertex
                    //only one fine vertex in a coarse vertex can satisfy either this or the previous condition
                    interp_transpose.entries(x) = interp_transpose.entries(end - 1);
                    interp_transpose.entries(end - 1) = u;
                    end_vertex = u;
                    break;
                }
            }
        }
        end--;
        while(end > start){
            //find the vertex before end_vertex
            for(edge_offset_t x = start; x < end; x++){
                ordinal_t u = interp_transpose.entries(x);
                //u MUST have an edge
                //and v must be in the same aggregate
                ordinal_t v = g(u);
                if(v == end_vertex){
                    interp_transpose.entries(x) = interp_transpose.entries(end - 1);
                    interp_transpose.entries(end - 1) = u;
                    end_vertex = u;
                    break;
                }
            }
            end--;
        }
    });
    //printf("Time to sort transposed entries: %.3f\n", timer.seconds());
    timer.reset();
    return interp_transpose;
}

graph_type collect_outputs_first(interp_t interp, ordinal_t null_id) {
    vtx_view_t entries = transpose_null(interp, null_id);
    edge_offset_t size = entries.extent(0);
    edge_view_t row_map("row map", size + 1);
    Kokkos::parallel_for("init write sizes", policy_t(0, size + 1), KOKKOS_LAMBDA(const edge_offset_t i){
        row_map(i) = i;
    });
    graph_type output(entries, row_map);
    return output;
}

ordinal_t relabel_crosses(crosses c, vtx_view_t cross_glues, interp_t interp, ordinal_t offset){
    vtx_subview_t relabel_count("relabel count");
    Kokkos::parallel_scan("enumerate crosses", interp.entries.extent(0), KOKKOS_LAMBDA(const ordinal_t i, ordinal_t& update, const bool final){
        if(interp.entries(i) == ORD_MAX){
            if(final){
                interp.entries(i) = update;
            }
            update++;
        }
        if(final && i + 1 == interp.entries.extent(0)){
            relabel_count() = update;
        }
    });
    ordinal_t next_offset = 0;
    Kokkos::deep_copy(next_offset, relabel_count);
    next_offset += offset;
    Kokkos::parallel_for("relabel crosses", cross_glues.extent(0), KOKKOS_LAMBDA(const ordinal_t x){
        ordinal_t i = cross_glues(x);
        if(c.in(i) != ORD_MAX){
            c.in(i) = offset + interp.entries(c.in(i));
        } else if(c.out(i) != ORD_MAX){
            c.out(i) = offset + interp.entries(c.out(i));
        }
    });
    return next_offset;
}

vtx_view_t coarsen_crosses(crosses c, vtx_view_t& rem_idx, interp_t interp){
    ordinal_t null_count = 0;
    Kokkos::parallel_reduce("count nulls", rem_idx.extent(0), KOKKOS_LAMBDA(const ordinal_t x, ordinal_t& update){
        ordinal_t i = rem_idx(x);
        ordinal_t f = c.in(i);
        if(f == ORD_MAX){
            f = c.out(i);
        }
        if(f != ORD_MAX && interp.entries(f) == 0){
            update++;
        }
    }, null_count);
    vtx_view_t cross_glues("cross glues", null_count);
    Kokkos::parallel_scan("write nulls", rem_idx.extent(0), KOKKOS_LAMBDA(const ordinal_t x, ordinal_t &update, const bool final){
        ordinal_t i = rem_idx(x);
        ordinal_t f = c.in(i);
        if(f == ORD_MAX){
            f = c.out(i);
        }
        if(f != ORD_MAX && interp.entries(f) == 0){
            if(final){
                cross_glues(update) = i;
            }
            update++;
        }
    });
    ordinal_t non_null_count = 0;
    Kokkos::parallel_reduce("count nulls", rem_idx.extent(0), KOKKOS_LAMBDA(const ordinal_t x, ordinal_t& update){
        ordinal_t i = rem_idx(x);
        ordinal_t f = c.in(i);
        if(f == ORD_MAX){
            f = c.out(i);
        }
        if(f != ORD_MAX && interp.entries(f) != 0){
            update++;
        }
    }, non_null_count);
    vtx_view_t new_rem_idx("new rem idx", non_null_count);
    Kokkos::parallel_scan("write non-nulls", rem_idx.extent(0), KOKKOS_LAMBDA(const ordinal_t x, ordinal_t &update, const bool final){
        ordinal_t i = rem_idx(x);
        ordinal_t f = c.in(i);
        if(f == ORD_MAX){
            f = c.out(i);
        }
        if(f != ORD_MAX && interp.entries(f) != 0){
            if(final){
                new_rem_idx(update) = i;
            }
            update++;
        }
    });
    rem_idx = new_rem_idx;
    Kokkos::parallel_for("coarsen ins", rem_idx.extent(0), KOKKOS_LAMBDA(const ordinal_t x){
        ordinal_t i = rem_idx(x);
        ordinal_t f = c.in(i);
        if(f != ORD_MAX){
            ordinal_t coarse_f = interp.entries(f) - 1;
            c.in(i) = coarse_f;
        }
    });
    Kokkos::parallel_for("coarsen ins", rem_idx.extent(0), KOKKOS_LAMBDA(const ordinal_t x){
        ordinal_t i = rem_idx(x);
        ordinal_t f = c.out(i);
        if(f != ORD_MAX){
            ordinal_t coarse_f = interp.entries(f) - 1;
            c.out(i) = coarse_f;
        }
    });
    Kokkos::parallel_for("modify interp", null_count, KOKKOS_LAMBDA(const ordinal_t x){
        ordinal_t i = cross_glues(x);
        ordinal_t f = c.in(i);
        if(f == ORD_MAX){
            f = c.out(i);
        }
        interp.entries(f) = ORD_MAX;
    });
    return cross_glues;
}

vtx_view_t init_sequence(ordinal_t n){
    vtx_view_t sequence("sequence", n);
    Kokkos::parallel_for("init sequence", n, KOKKOS_LAMBDA(const ordinal_t i){
        sequence(i) = i;
    });
    return sequence;
}

std::list<graph_type> coarsen_de_bruijn_full_cycle(vtx_view_t cur, crosses c, ExperimentLoggerUtil& experiment){
    std::list<graph_type> glue_list, cross_list;
    int count = 0;
    Kokkos::Timer timer;
    bool first = true;
    graph_type glue_last;
    vtx_view_t rem_idx = init_sequence(c.in.extent(0));
    ordinal_t cross_offset = 0;
    while(cur.extent(0) > 0){
        count++;
        printf("Calculating coarse graph %d\n", count);
        printf("input vertices: %lu\n", cur.extent(0));
        timer.reset();
        interp_t interp = mapper.coarsen_HEC(cur, experiment);
        experiment.addMeasurement(ExperimentLoggerUtil::Measurement::Map, timer.seconds());
        timer.reset();
        graph_type glue = transpose_and_sort(interp, cur);
        vtx_view_t crossing_aggs = coarsen_crosses(c, rem_idx, interp);
        experiment.addMeasurement(ExperimentLoggerUtil::Measurement::InterpTranspose, timer.seconds());
        timer.reset();
        if(first){
            glue_list.push_back(collect_outputs_first(interp, 0));
            cross_list.push_back(collect_outputs_first(interp, ORD_MAX));
            glue_last = glue; 
        } else {
            vtx_view_t nulls = transpose_null(interp, 0);
            vtx_view_t crosses = transpose_null(interp, ORD_MAX);
            glue_list.push_back(compacter.collect_outputs(glue_last, nulls));
            cross_list.push_back(compacter.collect_outputs(glue_last, crosses));
            glue_last = compacter.collect_unitigs(glue_last, glue);
        }
        cross_offset = relabel_crosses(c, crossing_aggs, interp, cross_offset);
        first = false;
        experiment.addMeasurement(ExperimentLoggerUtil::Measurement::CompactGlues, timer.seconds());
        timer.reset();
        cur = coarsen_de_bruijn_graph(cur, interp);
        experiment.addMeasurement(ExperimentLoggerUtil::Measurement::Build, timer.seconds());
        timer.reset();
    }
    ordinal_t total_rows = 0;
    ordinal_t cross_rows = 0;
    for(graph_type g : glue_list){
        total_rows += g.numRows();
    }
    for(graph_type g : cross_list){
        cross_rows += g.numRows();
    }
#ifdef HUGE
    printf("Total vtx after glueing: %lu\n", total_rows);
    printf("Total cut vtx after glueing: %lu\n", cross_rows);
    printf("Total cut vtx after glueing (verify): %lu\n", cross_offset);
#elif defined(LARGE)
    printf("Total vtx after glueing: %u\n", total_rows);
    printf("Total cut vtx after glueing: %u\n", cross_rows);
    printf("Total cut vtx after glueing (verify): %u\n", cross_offset);
#else
    printf("Total vtx after glueing: %u\n", total_rows);
    printf("Total cut vtx after glueing: %u\n", cross_rows);
    printf("Total cut vtx after glueing (verify): %u\n", cross_offset);
#endif
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
