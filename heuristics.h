#pragma once
#include <limits>
#include <Kokkos_Core.hpp>
#include <Kokkos_Atomic.hpp>
#include <Kokkos_Random.hpp>
#include <Kokkos_Sort.hpp>
#include <Kokkos_Functional.hpp>
#include "KokkosSparse_CrsMatrix.hpp"
#include "KokkosGraph_MIS2.hpp"
#include "KokkosKernels_SparseUtils.hpp"
#include "ExperimentLoggerUtil.cpp"

template<typename ordinal_t, typename edge_offset_t, typename scalar_t, class Device>
class coarsen_heuristics {
public:
    using exec_space = typename Device::execution_space;
    using matrix_t = typename KokkosSparse::CrsMatrix<scalar_t, ordinal_t, Device, void, edge_offset_t>;
    using vtx_view_t = typename Kokkos::View<ordinal_t*, Device>;
    using wgt_view_t = typename Kokkos::View<scalar_t*, Device>;
    using edge_view_t = typename Kokkos::View<edge_offset_t*, Device>;
    using edge_subview_t = typename Kokkos::View<edge_offset_t, Device>;
    using vtx_subview_t = typename Kokkos::View<ordinal_t, Device>;
    using rand_view_t = typename Kokkos::View<uint64_t*, Device>;
    using graph_type = typename matrix_t::staticcrsgraph_type;
    using policy_t = typename Kokkos::RangePolicy<exec_space>;
    using team_policy_t = typename Kokkos::TeamPolicy<exec_space>;
    using member = typename team_policy_t::member_type;
    using part_view_t = typename Kokkos::View<int*, Device>;
    using pool_t = Kokkos::Random_XorShift64_Pool<Device>;
    using gen_t = typename pool_t::generator_type;
    using hasher_t = Kokkos::pod_hash<ordinal_t>;
    static constexpr ordinal_t ORD_MAX = std::numeric_limits<ordinal_t>::max();

    struct interp_t
    {
        vtx_view_t entries;
        ordinal_t nc, n;

        interp_t(vtx_view_t _entries, ordinal_t _nc, ordinal_t _n) :
            entries(_entries), nc(_nc), n(_n) {}
    };


    template <class in, class out>
    Kokkos::View<out*, Device> sort_order(Kokkos::View<in*, Device> array, in max, in min) {
        typedef Kokkos::BinOp1D< Kokkos::View<in*, Device> > BinOp;
        BinOp bin_op(array.extent(0), min, max);
        //VERY important that final parameter is true
        Kokkos::BinSort< Kokkos::View<in*, Device>, BinOp, exec_space, out >
            sorter(array, bin_op, true);
        sorter.create_permute_vector();
        return sorter.get_permute_vector();
    }

    vtx_view_t generate_permutation(ordinal_t n, pool_t rand_pool) {
        rand_view_t randoms("randoms", n);

        Kokkos::Timer t;
        Kokkos::parallel_for("create random entries", policy_t(0, n), KOKKOS_LAMBDA(ordinal_t i){
            gen_t generator = rand_pool.get_state();
            randoms(i) = generator.urand64();
            rand_pool.free_state(generator);
        });
        //printf("random time: %.4f\n", t.seconds());
        t.reset();

        int t_buckets = 2*n;
        vtx_view_t buckets("buckets", t_buckets);
        Kokkos::parallel_for("init buckets", policy_t(0, t_buckets), KOKKOS_LAMBDA(ordinal_t i){
            buckets(i) = ORD_MAX;
        });

        uint64_t max = std::numeric_limits<uint64_t>::max();
        uint64_t bucket_size = max / t_buckets;
        Kokkos::parallel_for("insert buckets", policy_t(0, n), KOKKOS_LAMBDA(ordinal_t i){
            ordinal_t bucket = randoms(i) / bucket_size;
            //jesus take the wheel
            for(;; bucket++){
                if(bucket >= t_buckets) bucket -= t_buckets;
                if(buckets(bucket) == ORD_MAX){
                    //attempt to insert into bucket
                    if(Kokkos::atomic_compare_exchange_strong(&buckets(bucket), ORD_MAX, i)){
                        break;
                    }
                }
            }
        });
        
        vtx_view_t permute("permutation", n);
        Kokkos::parallel_scan("extract permutation", policy_t(0, t_buckets), KOKKOS_LAMBDA(const ordinal_t i, ordinal_t& update, const bool final){
            if(buckets(i) != ORD_MAX){
                if(final){
                    permute(update) = buckets(i);
                }
                update++;
            }
        });

        /*
        uint64_t max = std::numeric_limits<uint64_t>::max();
        typedef Kokkos::BinOp1D< rand_view_t > BinOp;
        BinOp bin_op(n, 0, max);
        //VERY important that final parameter is true
        Kokkos::BinSort< rand_view_t, BinOp, exec_space, ordinal_t >
            sorter(randoms, bin_op, true);
        sorter.create_permute_vector();
        */
        //printf("sort time: %.4f\n", t.seconds());
        t.reset();
        return permute;//sorter.get_permute_vector();
    }

    //hn is a list of vertices such that vertex i wants to aggregate with vertex hn(i)
    ordinal_t parallel_map_construct(vtx_view_t vcmap, const ordinal_t n, const vtx_view_t hn, ExperimentLoggerUtil& experiment) {

        Kokkos::View<ordinal_t, Device> nvertices_coarse("nvertices");
        //coarse vertex 0 is used to map vertices with no edges
        ordinal_t nvc = 1;
        Kokkos::deep_copy(nvertices_coarse, nvc);

        //construct mapping using heaviest edges
        Kokkos::parallel_for("compute mappings", policy_t(0, n), KOKKOS_LAMBDA(ordinal_t u) {
            ordinal_t v = hn(u);
            if (v != ORD_MAX) {
                bool success = false;
                if(hn(v) == u){
                    //let the thread with no outgoing edge handle it
                    if(vcmap(u) == ORD_MAX - 3){
                        vcmap(u) = ORD_MAX;
                    } else {
                        success = true;
                    }
                }
                while(!success){
                    if (Kokkos::atomic_compare_exchange_strong(&vcmap(u), ORD_MAX, ORD_MAX - 1)) {
                        if (Kokkos::atomic_compare_exchange_strong(&vcmap(v), ORD_MAX, ORD_MAX - 1)) {
                            ordinal_t c = u + 1;
                            //printf("%u\n", c);
                            vcmap(u) = c;
                            vcmap(v) = c;
                            success = true;
                        }
                        else {
                            //u can join v's aggregate if it already has one
                            if (vcmap(v) <= n) {
                                vcmap(u) = vcmap(v);
                                success = true;
                            }
                            else {
                                vcmap(u) = ORD_MAX;
                            }
                        }
                    } else {
                        success = true;
                    }
                }
            }
        });
        Kokkos::parallel_scan("assign aggregates", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t u, ordinal_t& update, const bool final){
            if(vcmap(u) - 1 == u){
                if(final){
                    ordinal_t cv = update + 1;
                    vcmap(u) = cv;
                }
                update++;
            } else if(vcmap(u) > 0 && final){
                vcmap(u) = vcmap(u) - 1 + n;
            }
            if(final && (u + 1) == n){
                nvertices_coarse() = nvertices_coarse() + update;
            }
        });
        Kokkos::parallel_for("propagate aggregates", policy_t(0, n), KOKKOS_LAMBDA(ordinal_t u) {
            if(vcmap(u) >= n) {
                ordinal_t c_id = vcmap(u) - n;
                vcmap(u) = vcmap(c_id);
            }
        });
        ordinal_t nc = 0;
        Kokkos::deep_copy(nc, nvertices_coarse);
        return nc;
    }

    matrix_t coarsen_mis_2(const matrix_t& g,
        ExperimentLoggerUtil& experiment) {

        ordinal_t n = g.numRows();

        typename matrix_t::staticcrsgraph_type::entries_type::non_const_value_type nc = 0;
        vtx_view_t vcmap = KokkosGraph::Experimental::graph_mis2_coarsen<Device, typename matrix_t::staticcrsgraph_type::row_map_type, typename matrix_t::staticcrsgraph_type::entries_type, vtx_view_t>(g.graph.row_map, g.graph.entries, nc);

        edge_view_t row_map("interpolate row map", n + 1);

        Kokkos::parallel_for(policy_t(0, n + 1), KOKKOS_LAMBDA(ordinal_t u){
            row_map(u) = u;
        });

        vtx_view_t entries("interpolate entries", n);
        wgt_view_t values("interpolate values", n);
        //compute the interpolation weights
        Kokkos::parallel_for(policy_t(0, n), KOKKOS_LAMBDA(ordinal_t u){
            entries(u) = vcmap(u);
            values(u) = 1.0;
        });

        graph_type graph(entries, row_map);
        matrix_t interp("interpolate", nc, values, graph);

        return interp;
    }

    interp_t coarsen_HEC(const vtx_view_t g,
        ExperimentLoggerUtil& experiment) {

        ordinal_t n = g.extent(0);

        vtx_view_t hn("heavies", n);

        vtx_view_t vcmap("vcmap", n);

        Kokkos::parallel_for("initialize vcmap", policy_t(0, n), KOKKOS_LAMBDA(ordinal_t i) {
            vcmap(i) = ORD_MAX;
            hn(i) = ORD_MAX;
        });

        Kokkos::Timer timer;

        experiment.addMeasurement(ExperimentLoggerUtil::Measurement::Permute, timer.seconds());
        timer.reset();

        Kokkos::parallel_for("edge choose", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t i) {
            //i has an out edge
            if(g(i) != ORD_MAX){
                //only one edge is possible
                //write its heaviest neighbor as me
                ordinal_t v = g(i);
                hn(v) = i;
            }
        });
        Kokkos::parallel_for("edge choose", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t i) {
            //i has no in edge
            if(hn(i) == ORD_MAX){
                //i has an out edge
                if(g(i) != ORD_MAX){
                    ordinal_t v = g(i);
                    hn(i) = v;
                    vcmap(i) = ORD_MAX - 3;
                } else {
                    //no edges, assign to output vertex
                    vcmap(i) = 0;
                }
            }
        });
        experiment.addMeasurement(ExperimentLoggerUtil::Measurement::Heavy, timer.seconds());
        timer.reset();
        ordinal_t nc = 0;
        nc = parallel_map_construct(vcmap, n, hn, experiment);
        printf("map construct time: %.3fs\n", timer.seconds());
        experiment.addMeasurement(ExperimentLoggerUtil::Measurement::MapConstruct, timer.seconds());
        timer.reset();
    
        interp_t interp(vcmap, nc, n);    

        return interp;
    }

};
