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

        ordinal_t perm_length = n;
        Kokkos::View<ordinal_t, Device> nvertices_coarse("nvertices");
        //coarse vertex 0 is used to map vertices with no edges
        ordinal_t nvc = 1;
        Kokkos::deep_copy(nvertices_coarse, nvc);

        int swap = 1;
        //find vertices that must be aggregated
        vtx_view_t rem_vtx;
#ifdef HUGE
        printf("edgeful vertices: %lu\n", perm_length);
#else
        printf("edgeful vertices: %u\n", perm_length);
#endif
        int count = 0;
        //construct mapping using heaviest edges
        while (perm_length > 0) {
            Kokkos::parallel_for("compute mappings", policy_t(0, perm_length), KOKKOS_LAMBDA(ordinal_t i) {
                ordinal_t u;
                //on first pass we don't have a set of remaining vertices
                if(count == 0){
                    u = i;
                } else {
                    u = rem_vtx(i);
                }
                //v can be ORD_MAX on the first pass, but shouldn't be on following passes
                ordinal_t v = hn(u);
                int condition = u < v;
                //need to enforce an ordering condition to allow hard-stall conditions to be broken
                //but these hard-stall conditions are rare; so we wait to break them until a few iterations have occurred
                if (v != ORD_MAX && (count < 5 || condition ^ swap)) {
                    if (Kokkos::atomic_compare_exchange_strong(&vcmap(u), ORD_MAX, ORD_MAX - 1)) {
                        if (u == v || Kokkos::atomic_compare_exchange_strong(&vcmap(v), ORD_MAX, ORD_MAX - 2)) {
                            //do nothing here
                        }
                        else {
                            //u can join v's aggregate if it already has one
                            if (vcmap(v) < n) {
                                vcmap(u) = vcmap(v);
                            }
                            else {
                                vcmap(u) = ORD_MAX;
                            }
                        }
                    }
                }
            });
            Kokkos::fence();
            Kokkos::View<ordinal_t, Device> old_nvc("nvertices old");
            Kokkos::deep_copy(old_nvc, nvertices_coarse);
            Kokkos::parallel_scan("assign aggregates", policy_t(0, perm_length), KOKKOS_LAMBDA(const ordinal_t i, ordinal_t& update, const bool final){
                ordinal_t u;
                if(count == 0){
                    u = i;
                } else {
                    u = rem_vtx(i);
                }
                if(vcmap(u) == (ORD_MAX - 1)){
                    if(final){
                        ordinal_t cv = update + old_nvc();
                        vcmap(u) = cv;
                        vcmap(hn(u)) = cv;
                    }
                    update++;
                }
                if(final && (i + 1) == perm_length){
                    nvertices_coarse() = nvertices_coarse() + update;
                }
            });
            Kokkos::fence();
            //add the ones that failed to be reprocessed next round
            Kokkos::Timer timer;
            ordinal_t next_length = 0;
            Kokkos::parallel_reduce("count vtx not already mapped", policy_t(0, perm_length), KOKKOS_LAMBDA(const ordinal_t i, ordinal_t& update) {
                ordinal_t u;
                if(count == 0){
                    u = i;
                } else {
                    u = rem_vtx(i);
                }
                if (vcmap(u) >= n) {
                    update++;
                }
            }, next_length);
            vtx_view_t next_perm("next perm", next_length);
            Kokkos::parallel_scan("write vtx not already mapped", policy_t(0, perm_length), KOKKOS_LAMBDA(const ordinal_t i, ordinal_t& update, const bool final) {
                ordinal_t u;
                if(count == 0){
                    u = i;
                } else {
                    u = rem_vtx(i);
                }
                if (vcmap(u) >= n) {
                    if(final){
                        next_perm(update) = u;
                    }
                    update++;
                }
            });
            Kokkos::fence();
            swap = swap ^ 1;
            perm_length = next_length;
            rem_vtx = next_perm;
            experiment.addMeasurement(ExperimentLoggerUtil::Measurement::CoarsenPair, timer.seconds());
            timer.reset();
            count++;
        }
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
        experiment.addMeasurement(ExperimentLoggerUtil::Measurement::MapConstruct, timer.seconds());
        timer.reset();
    
        interp_t interp(vcmap, nc, n);    

        return interp;
    }

    ordinal_t countInf(vtx_view_t target) {
        ordinal_t totalInf = 0;

        Kokkos::parallel_reduce(policy_t(0, target.extent(0)), KOKKOS_LAMBDA(ordinal_t i, ordinal_t & thread_sum) {
            if (target(i) == ORD_MAX) {
                thread_sum++;
            }
        }, totalInf);

        return totalInf;
    }

    struct MatchByHashSorted {
        vtx_view_t vcmap, unmapped;
        Kokkos::View<uint32_t*, Device> hashes;
        ordinal_t unmapped_total;
        Kokkos::View<ordinal_t, Device> nvertices_coarse;
        MatchByHashSorted(vtx_view_t vcmap,
            vtx_view_t unmapped,
            Kokkos::View<uint32_t*, Device> hashes,
            ordinal_t unmapped_total,
            Kokkos::View<ordinal_t, Device> nvertices_coarse) :
            vcmap(vcmap),
            unmapped(unmapped),
            hashes(hashes),
            unmapped_total(unmapped_total),
            nvertices_coarse(nvertices_coarse) {}

        KOKKOS_INLINE_FUNCTION
            void operator()(const ordinal_t i, ordinal_t& update, const bool final) const {

            ordinal_t u = unmapped(i);
            ordinal_t tentative = 0;
            if (i == 0) {
                tentative = i;
            }
            else if (hashes(i - 1) != hashes(i)) {
                tentative = i;
            }

            if (tentative > update) {
                update = tentative;
            }

            if (final) {
                //update should contain the index of the first hash that equals hash(i), could be i
                //we want to determine if i is an odd offset from update
                ordinal_t isOddOffset = (i - update) & 1;
                //if even (0 counts as even) we match unmapped(i) with unmapped(i+1) if hash(i) == hash(i+1)
                //if odd do nothing
                if (isOddOffset == 0) {
                    if (i + 1 < unmapped_total) {
                        if (hashes(i) == hashes(i + 1)) {
                            ordinal_t v = unmapped(i + 1);
                            vcmap(u) = Kokkos::atomic_fetch_add(&nvertices_coarse(), 1);
                            vcmap(v) = vcmap(u);
                        }
                    }
                }
            }
        }

        KOKKOS_INLINE_FUNCTION
            void join(volatile ordinal_t& update, volatile const ordinal_t& input) const {
            if (input > update) update = input;
        }

    };

    matrix_t coarsen_match(const matrix_t& g,
        bool uniform_weights,
        ExperimentLoggerUtil& experiment,
        int match_choice) {

        ordinal_t n = g.numRows();

        vtx_view_t hn("heavies", n);

        vtx_view_t vcmap("vcmap", n);

        Kokkos::parallel_for("initialize vcmap", policy_t(0, n), KOKKOS_LAMBDA(ordinal_t i) {
            vcmap(i) = ORD_MAX;
        });

        rand_view_t randoms("randoms", n);

        pool_t rand_pool(std::time(nullptr));
        Kokkos::Timer timer;

        vtx_view_t vperm = generate_permutation(n, rand_pool);

        vtx_view_t reverse_map("reversed", n);
        Kokkos::parallel_for("construct reverse map", policy_t(0, n), KOKKOS_LAMBDA(ordinal_t i) {
            reverse_map(vperm(i)) = i;
        });
        experiment.addMeasurement(ExperimentLoggerUtil::Measurement::Permute, timer.seconds());
        timer.reset();

        if (uniform_weights) {
            //all weights equal at this level so choose heaviest edge randomly
            Kokkos::parallel_for("Random HN", policy_t(0, n), KOKKOS_LAMBDA(ordinal_t i) {
                gen_t generator = rand_pool.get_state();
                ordinal_t adj_size = g.graph.row_map(i + 1) - g.graph.row_map(i);
                ordinal_t offset = g.graph.row_map(i) + (generator.urand64() % adj_size);
                hn(i) = g.graph.entries(offset);
                rand_pool.free_state(generator);
            });
        }
        else {
            Kokkos::parallel_for("Heaviest HN", policy_t(0, n), KOKKOS_LAMBDA(ordinal_t i) {
                ordinal_t hn_i = g.graph.entries(g.graph.row_map(i));
                scalar_t max_ewt = g.values(g.graph.row_map(i));

                edge_offset_t end_offset = g.graph.row_map(i + 1);// +g.edges_per_source[i];

                for (edge_offset_t j = g.graph.row_map(i) + 1; j < end_offset; j++) {
                    if (max_ewt < g.values(j)) {
                        max_ewt = g.values(j);
                        hn_i = g.graph.entries(j);
                    }

                }
                hn(i) = hn_i;
            });
        }
        timer.reset();
        vtx_view_t match("match", n);
        Kokkos::parallel_for(policy_t(0, n), KOKKOS_LAMBDA(ordinal_t i){
            match(i) = ORD_MAX;
        });

        ordinal_t perm_length = n;

        Kokkos::View<ordinal_t, Device> nvertices_coarse("nvertices");

        //construct mapping using heaviest edges
        int swap = 1;
        timer.reset();
        while (perm_length > 0) {
            vtx_view_t next_perm("next perm", perm_length);
            Kokkos::View<ordinal_t, Device> next_length("next_length");

            //match vertices with heaviest unmatched edge
            Kokkos::parallel_for(policy_t(0, perm_length), KOKKOS_LAMBDA(ordinal_t i){
                ordinal_t u = vperm(i);
                ordinal_t v = hn(u);
                int condition = reverse_map(u) < reverse_map(v);
                //need to enforce an ordering condition to allow hard-stall conditions to be broken
                if (condition ^ swap) {
                    if (Kokkos::atomic_compare_exchange_strong(&match(u), ORD_MAX, v)) {
                        if (Kokkos::atomic_compare_exchange_strong(&match(v), ORD_MAX, u)) {
                            ordinal_t cv = Kokkos::atomic_fetch_add(&nvertices_coarse(), 1);
                            vcmap(u) = cv;
                            vcmap(v) = cv;
                        }
                        else {
                            match(u) = ORD_MAX;
                        }
                    }
                }
            });
            Kokkos::fence();

            //add the ones that failed to be reprocessed next round
            //maybe count these then create next_perm to save memory?
            Kokkos::parallel_for(policy_t(0, perm_length), KOKKOS_LAMBDA(ordinal_t i){
                ordinal_t u = vperm(i);
                if (vcmap(u) == ORD_MAX) {
                    ordinal_t h = ORD_MAX;

                    if (uniform_weights) {
                        ordinal_t max_ewt = 0;
                        //we have to iterate over the edges anyways because we need to check if any are unmatched!
                        //so instead of randomly choosing a heaviest edge, we instead use the reverse permutation order as the weight
                        for (edge_offset_t j = g.graph.row_map(u); j < g.graph.row_map(u + 1); j++) {
                            ordinal_t v = g.graph.entries(j);
                            //v must be unmatched to be considered
                            if (vcmap(v) == ORD_MAX) {
                                //using <= so that zero weight edges may still be chosen
                                if (max_ewt <= reverse_map(v)) {
                                    max_ewt = reverse_map(v);
                                    h = v;
                                }
                            }
                        }
                    }
                    else {
                        scalar_t max_ewt = 0;
                        for (edge_offset_t j = g.graph.row_map(u); j < g.graph.row_map(u + 1); j++) {
                            ordinal_t v = g.graph.entries(j);
                            //v must be unmatched to be considered
                            if (vcmap(v) == ORD_MAX) {
                                //using <= so that zero weight edges may still be chosen
                                if (max_ewt <= g.values(j)) {
                                    max_ewt = g.values(j);
                                    h = v;
                                }
                            }
                        }
                    }

                    if (h != ORD_MAX) {
                        ordinal_t add_next = Kokkos::atomic_fetch_add(&next_length(), 1);
                        next_perm(add_next) = u;
                        hn(u) = h;
                    }
                }
            });
            Kokkos::fence();
            swap = swap ^ 1;
            Kokkos::deep_copy(perm_length, next_length);
            vperm = next_perm;
        }

        if (match_choice == 1) {
            ordinal_t unmapped = countInf(vcmap);
            double unmappedRatio = static_cast<double>(unmapped) / static_cast<double>(n);

            //leaf matches
            if (unmappedRatio > 0.25) {
                Kokkos::parallel_for(policy_t(0, n), KOKKOS_LAMBDA(ordinal_t u){
                    if (vcmap(u) != ORD_MAX) {
                        ordinal_t lastLeaf = ORD_MAX;
                        for (edge_offset_t j = g.graph.row_map(u); j < g.graph.row_map(u + 1); j++) {
                            ordinal_t v = g.graph.entries(j);
                            //v must be unmatched to be considered
                            if (vcmap(v) == ORD_MAX) {
                                //must be degree 1 to be a leaf
                                if (g.graph.row_map(v + 1) - g.graph.row_map(v) == 1) {
                                    if (lastLeaf == ORD_MAX) {
                                        lastLeaf = v;
                                    }
                                    else {
                                        vcmap(lastLeaf) = Kokkos::atomic_fetch_add(&nvertices_coarse(), 1);
                                        vcmap(v) = vcmap(lastLeaf);
                                        lastLeaf = ORD_MAX;
                                    }
                                }
                            }
                        }
                    }
                });
            }

            unmapped = countInf(vcmap);
            unmappedRatio = static_cast<double>(unmapped) / static_cast<double>(n);

            //twin matches
            if (unmappedRatio > 0.25) {
                vtx_view_t unmappedVtx("unmapped vertices", unmapped);
                Kokkos::View<uint32_t*, Device> hashes("hashes", unmapped);

                Kokkos::View<ordinal_t, Device> unmappedIdx("unmapped index");
                hasher_t hasher;
                //compute digests of adjacency lists
                Kokkos::parallel_for("create digests", team_policy_t(n, Kokkos::AUTO), KOKKOS_LAMBDA(const member & thread) {
                    ordinal_t u = thread.league_rank();
                    if (vcmap(u) == ORD_MAX) {
                        uint32_t hash = 0;
                        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(thread, g.graph.row_map(u), g.graph.row_map(u + 1)), [=](const edge_offset_t j, uint32_t& thread_sum) {
                            thread_sum += hasher(g.graph.entries(j));
                            }, hash);
                        Kokkos::single(Kokkos::PerTeam(thread), [=]() {
                            ordinal_t idx = Kokkos::atomic_fetch_add(&unmappedIdx(), 1);
                            unmappedVtx(idx) = u;
                            hashes(idx) = hash;
                            });
                    }
                });
                uint32_t max = std::numeric_limits<uint32_t>::max();
                typedef Kokkos::BinOp1D< Kokkos::View<uint32_t*, Device> > BinOp;
                BinOp bin_op(unmapped, 0, max);
                //VERY important that final parameter is true
                Kokkos::BinSort< Kokkos::View<uint32_t*, Device>, BinOp, Kokkos::DefaultExecutionSpace, ordinal_t >
                    sorter(hashes, bin_op, true);
                sorter.create_permute_vector();
                sorter.template sort< Kokkos::View<uint32_t*, Device> >(hashes);
                sorter.template sort< vtx_view_t >(unmappedVtx);

                MatchByHashSorted matchTwinFunctor(vcmap, unmappedVtx, hashes, unmapped, nvertices_coarse);
                Kokkos::parallel_scan("match twins", policy_t(0, unmapped), matchTwinFunctor);
            }

            unmapped = countInf(vcmap);
            unmappedRatio = static_cast<double>(unmapped) / static_cast<double>(n);

            //relative matches
            if (unmappedRatio > 0.25) {

                //get possibly mappable vertices of unmapped
                vtx_view_t mappableVtx("mappable vertices", unmapped);
                Kokkos::parallel_scan("get unmapped", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t i, ordinal_t & update, const bool final){
                    if (vcmap(i) == ORD_MAX) {
                        if (final) {
                            mappableVtx(update) = i;
                        }

                        update++;
                    }
                });


                ordinal_t mappable_count = unmapped;
                do {

                    Kokkos::parallel_for("reset hn", policy_t(0, mappable_count), KOKKOS_LAMBDA(ordinal_t i){
                        ordinal_t u = mappableVtx(i);
                        hn(u) = ORD_MAX;
                    });

                    //choose relatives for unmapped vertices
                    Kokkos::parallel_for("assign relatives", policy_t(0, n), KOKKOS_LAMBDA(ordinal_t i){
                        if (vcmap(i) != ORD_MAX) {
                            ordinal_t last_free = ORD_MAX;
                            for (edge_offset_t j = g.graph.row_map(i); j < g.graph.row_map(i + 1); j++) {
                                ordinal_t v = g.graph.entries(j);
                                if (vcmap(v) == ORD_MAX) {
                                    if (last_free != ORD_MAX) {
                                        //there can be multiple threads updating this but it doesn't matter as long as they have some value
                                        hn(last_free) = v;
                                        hn(v) = last_free;
                                        last_free = ORD_MAX;
                                    }
                                    else {
                                        last_free = v;
                                    }
                                }
                            }
                        }
                    });

                    //create a list of all mappable vertices according to set entries of hn
                    ordinal_t old_mappable = mappable_count;
                    mappable_count = 0;
                    Kokkos::parallel_reduce("count mappable", policy_t(0, old_mappable), KOKKOS_LAMBDA(const ordinal_t i, ordinal_t & thread_sum){
                        ordinal_t u = mappableVtx(i);
                        if (hn(u) != ORD_MAX) {
                            thread_sum++;
                        }
                    }, mappable_count);

                    vtx_view_t nextMappable("next mappable vertices", mappable_count);

                    Kokkos::parallel_scan("get next mappable", policy_t(0, old_mappable), KOKKOS_LAMBDA(const ordinal_t i, ordinal_t & update, const bool final){
                        ordinal_t u = mappableVtx(i);
                        if (hn(u) != ORD_MAX) {
                            if (final) {
                                nextMappable(update) = u;
                            }

                            update++;
                        }
                    });
                    mappableVtx = nextMappable;

                    //match vertices with chosen relative
                    if (mappable_count > 0) {
                        Kokkos::parallel_for(policy_t(0, mappable_count), KOKKOS_LAMBDA(ordinal_t i){
                            ordinal_t u = mappableVtx(i);
                            ordinal_t v = hn(u);
                            int condition = reverse_map(u) < reverse_map(v);
                            //need to enforce an ordering condition to allow hard-stall conditions to be broken
                            if (condition ^ swap) {
                                if (Kokkos::atomic_compare_exchange_strong(&match(u), ORD_MAX, v)) {
                                    if (Kokkos::atomic_compare_exchange_strong(&match(v), ORD_MAX, u)) {
                                        ordinal_t cv = Kokkos::atomic_fetch_add(&nvertices_coarse(), 1);
                                        vcmap(u) = cv;
                                        vcmap(v) = cv;
                                    }
                                    else {
                                        match(u) = ORD_MAX;
                                    }
                                }
                            }
                        });
                    }
                    Kokkos::fence();
                    swap = swap ^ 1;
                } while (mappable_count > 0);
            }
        }

        //create singleton aggregates of remaining unmatched vertices
        Kokkos::parallel_for(policy_t(0, n), KOKKOS_LAMBDA(ordinal_t i){
            if (vcmap(i) == ORD_MAX) {
                vcmap(i) = Kokkos::atomic_fetch_add(&nvertices_coarse(), 1);
            }
        });

        experiment.addMeasurement(ExperimentLoggerUtil::Measurement::MapConstruct, timer.seconds());
        timer.reset();

        ordinal_t nc = 0;
        Kokkos::deep_copy(nc, nvertices_coarse);

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
};
