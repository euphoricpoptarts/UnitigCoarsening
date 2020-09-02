#pragma once

#include "ExperimentLoggerUtil.cpp"
#include "definitions_kokkos.h"

namespace sgpar {
namespace sgpar_kokkos {
    Kokkos::View<int*> mis_2(const matrix_type& g) {

        sgp_vid_t n = g.numRows();

        Kokkos::View<int*> state("is membership", n);

        sgp_vid_t unassigned_total = n;
        Kokkos::View<uint64_t*> randoms("randomized", n);
        pool_t rand_pool(std::time(nullptr));
        Kokkos::parallel_for("create random entries", n, KOKKOS_LAMBDA(sgp_vid_t i){
            gen_t generator = rand_pool.get_state();
            randoms(i) = generator.urand64();
            rand_pool.free_state(generator);
        });

        /*vtx_view_t unassigned("unassigned vtx", n);
        Kokkos::parallel_for(n, KOKKOS_LAMBDA(const sgp_vid_t i){
            unassigned(i) = i;
        });*/

        while (unassigned_total > 0) {

            Kokkos::View<int*> tuple_state("tuple state", n);
            Kokkos::View<uint64_t*> tuple_rand("tuple rand", n);
            vtx_view_t tuple_idx("tuple index", n);

            Kokkos::View<int*> tuple_state_update("tuple state", n);
            Kokkos::View<uint64_t*> tuple_rand_update("tuple rand", n);
            vtx_view_t tuple_idx_update("tuple index", n);
            Kokkos::parallel_for(n, KOKKOS_LAMBDA(const sgp_vid_t i){
                tuple_state(i) = state(i);
                tuple_rand(i) = randoms(i);
                tuple_idx(i) = i;
            });

            for (int k = 0; k < 2; k++) {

                Kokkos::parallel_for(n, KOKKOS_LAMBDA(const sgp_vid_t i){
                    int max_state = tuple_state(i);
                    uint64_t max_rand = tuple_rand(i);
                    sgp_vid_t max_idx = tuple_idx(i);

                    for (sgp_eid_t j = g.graph.row_map(i); j < g.graph.row_map(i + 1); j++) {
                        sgp_vid_t v = g.graph.entries(j);
                        bool is_max = false;
                        if (tuple_state(v) > max_state) {
                            is_max = true;
                        }
                        else if (tuple_state(v) == max_state) {
                            if (tuple_rand(v) > max_rand) {
                                is_max = true;
                            }
                            else if (tuple_rand(v) == max_rand) {
                                if (tuple_idx(v) > max_idx) {
                                    is_max = true;
                                }
                            }
                        }
                        if (is_max) {
                            max_state = tuple_state(v);
                            max_rand = tuple_rand(v);
                            max_idx = tuple_idx(v);
                        }
                    }
                    tuple_state_update(i) = max_state;
                    tuple_rand_update(i) = max_rand;
                    tuple_idx_update(i) = max_idx;
                });

                Kokkos::parallel_for(n, KOKKOS_LAMBDA(const sgp_vid_t i){
                    tuple_state(i) = tuple_state_update(i);
                    tuple_rand(i) = tuple_rand_update(i);
                    tuple_idx(i) = tuple_idx_update(i);
                });
            }

            unassigned_total = 0;
            Kokkos::parallel_reduce(n, KOKKOS_LAMBDA(const sgp_vid_t i, sgp_vid_t & thread_sum){
                if (state(i) == 0) {
                    if (tuple_state(i) == state(i) && tuple_rand(i) == randoms(i) && tuple_idx(i) == i) {
                        state(i) = 1;
                    }
                    else if (tuple_state(i) == 1) {
                        state(i) = -1;
                    }
                }
                if (state(i) == 0) {
                    thread_sum++;
                }
            }, unassigned_total);
        }
        return state;
    }

    SGPAR_API int sgp_coarsen_mis_2(matrix_type& interp,
        sgp_vid_t* nvertices_coarse_ptr,
        const matrix_type& g,
        const int coarsening_level,
        sgp_pcg32_random_t* rng,
        ExperimentLoggerUtil& experiment) {

        sgp_vid_t n = g.numRows();
        /*typedef KokkosKernels::Experimental::KokkosKernelsHandle
            <sgp_eid_t, sgp_vid_t, sgp_wgt_t,
            typename Device::execution_space, typename Device::memory_space, typename Device::memory_space > KernelHandle;

        KernelHandle kh;
        kh.set_team_work_size(16);
        kh.set_dynamic_scheduling(true);

        kh.create_distance2_graph_coloring_handle();
        KokkosGraph::Experimental::graph_color_distance2(&kh, n, g.graph.row_map, g.graph.entries);
        Kokkos::View<sgp_vid_t*> colors = kh.get_distance2_graph_coloring_handle()->get_vertex_colors();
        kh.destroy_distance2_graph_coloring_handle();*/

        Kokkos::View<int*> colors = mis_2(g);

        Kokkos::View<sgp_vid_t> nvc("nvertices_coarse");
        Kokkos::View<sgp_vid_t*> vcmap("vcmap", n);

        int first_color = 1;

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
        //Kokkos::parallel_for(n, KOKKOS_LAMBDA(sgp_vid_t i){
        //    if (vcmap(i) == SGP_INFTY) {
        //        vcmap(i) = Kokkos::atomic_fetch_add(&nvc(), 1);
        //    }
        //});

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

    template <class in, class out>
    Kokkos::View<out*> sort_order(Kokkos::View<in*> array, in max, in min) {
        typedef Kokkos::BinOp1D< Kokkos::View<in*> > BinOp;
        BinOp bin_op(array.extent(0), min, max);
        //VERY important that final parameter is true
        Kokkos::BinSort< Kokkos::View<in*>, BinOp, Kokkos::DefaultExecutionSpace, out >
            sorter(array, bin_op, true);
        sorter.create_permute_vector();
        return sorter.get_permute_vector();
    }

    vtx_view_t generate_permutation(sgp_vid_t n, pool_t rand_pool) {
        Kokkos::View<uint64_t*> randoms("randoms", n);

        Kokkos::parallel_for("create random entries", n, KOKKOS_LAMBDA(sgp_vid_t i){
            gen_t generator = rand_pool.get_state();
            randoms(i) = generator.urand64();
            rand_pool.free_state(generator);
        });

        uint64_t max = std::numeric_limits<uint64_t>::max();
        typedef Kokkos::BinOp1D< Kokkos::View<uint64_t*> > BinOp;
        BinOp bin_op(n, 0, max);
        //VERY important that final parameter is true
        Kokkos::BinSort< Kokkos::View<uint64_t*>, BinOp, Kokkos::DefaultExecutionSpace, sgp_vid_t >
            sorter(randoms, bin_op, true);
        sorter.create_permute_vector();
        return sorter.get_permute_vector();
    }

    sgp_vid_t parallel_map_construct(vtx_view_t vcmap, const sgp_vid_t n, const vtx_view_t vperm, const vtx_view_t hn, const vtx_view_t ordering) {

        vtx_view_t match("match", n);
        Kokkos::parallel_for(n, KOKKOS_LAMBDA(sgp_vid_t i){
            match(i) = SGP_INFTY;
        });
        sgp_vid_t perm_length = n;
        Kokkos::View<sgp_vid_t> nvertices_coarse("nvertices");

        //construct mapping using heaviest edges
        int swap = 1;
        vtx_view_t curr_perm = vperm;
        while (perm_length > 0) {
            vtx_view_t next_perm("next perm", perm_length);
            Kokkos::View<sgp_vid_t> next_length("next_length");

            Kokkos::parallel_for(perm_length, KOKKOS_LAMBDA(sgp_vid_t i){
                sgp_vid_t u = curr_perm(i);
                sgp_vid_t v = hn(u);
                int condition = ordering(u) < ordering(v);
                //need to enforce an ordering condition to allow hard-stall conditions to be broken
                if (condition ^ swap) {
                    if (Kokkos::atomic_compare_exchange_strong(&match(u), SGP_INFTY, v)) {
                        if (u == v || Kokkos::atomic_compare_exchange_strong(&match(v), SGP_INFTY, u)) {
                            sgp_vid_t cv = Kokkos::atomic_fetch_add(&nvertices_coarse(), 1);
                            vcmap(u) = cv;
                            vcmap(v) = cv;
                        }
                        else {
                            if (vcmap(v) < n) {
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
                sgp_vid_t u = curr_perm(i);
                if (vcmap(u) >= n) {
                    sgp_vid_t add_next = Kokkos::atomic_fetch_add(&next_length(), 1);
                    next_perm(add_next) = u;
                    //been noticing some memory erros on my machine, probably from memory overclock
                    //this fixes the problem, and is lightweight
                    match(u) = SGP_INFTY;
                }
            });
            Kokkos::fence();
            swap = swap ^ 1;
            Kokkos::deep_copy(perm_length, next_length);
            curr_perm = next_perm;
        }
        sgp_vid_t nc = 0;
        Kokkos::deep_copy(nc, nvertices_coarse);
        return nc;
    }

    SGPAR_API int sgp_coarsen_HEC(matrix_type& interp,
        sgp_vid_t* nvertices_coarse_ptr,
        const matrix_type& g,
        const int coarsening_level,
        sgp_pcg32_random_t* rng,
        ExperimentLoggerUtil& experiment) {

        sgp_vid_t n = g.numRows();

        vtx_view_t hn("heavies", n);

        vtx_view_t vcmap("vcmap", n);

        Kokkos::parallel_for("initialize vcmap", n, KOKKOS_LAMBDA(sgp_vid_t i) {
            vcmap(i) = SGP_INFTY;
        });

        pool_t rand_pool(std::time(nullptr));
        Kokkos::Timer timer;

        vtx_view_t vperm = generate_permutation(n, rand_pool);

        vtx_view_t reverse_map("reversed", n);
        Kokkos::parallel_for("construct reverse map", n, KOKKOS_LAMBDA(sgp_vid_t i) {
            reverse_map(vperm(i)) = i;
        });
        experiment.addMeasurement(ExperimentLoggerUtil::Measurement::Permute, timer.seconds());
        timer.reset();

        if (coarsening_level == 1) {
            //all weights equal at this level so choose heaviest edge randomly
            Kokkos::parallel_for("Random HN", n, KOKKOS_LAMBDA(sgp_vid_t i) {
                gen_t generator = rand_pool.get_state();
                sgp_vid_t adj_size = g.graph.row_map(i + 1) - g.graph.row_map(i);
                sgp_vid_t offset = g.graph.row_map(i) + (generator.urand64() % adj_size);
                hn(i) = g.graph.entries(offset);
                rand_pool.free_state(generator);
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
        experiment.addMeasurement(ExperimentLoggerUtil::Measurement::Heavy, timer.seconds());
        timer.reset();
        sgp_vid_t nc = parallel_map_construct(vcmap, n, vperm, hn, reverse_map);
        experiment.addMeasurement(ExperimentLoggerUtil::Measurement::MapConstruct, timer.seconds());
        timer.reset();

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

    SGPAR_API int sgp_recoarsen_HEC(matrix_type& interp,
        sgp_vid_t* nvertices_coarse_ptr,
        const matrix_type& g,
        const eigenview_t part) {

        sgp_vid_t n = g.numRows();

        vtx_view_t hn("heavies", n);

        vtx_view_t vcmap("vcmap", n);

        Kokkos::parallel_for("initialize vcmap", n, KOKKOS_LAMBDA(sgp_vid_t i) {
            vcmap(i) = SGP_INFTY;
        });

        pool_t rand_pool(std::time(nullptr));
        Kokkos::Timer timer;

        vtx_view_t vperm = generate_permutation(n, rand_pool);

        vtx_view_t reverse_map("reversed", n);
        Kokkos::parallel_for("construct reverse map", n, KOKKOS_LAMBDA(sgp_vid_t i) {
            reverse_map(vperm(i)) = i;
        });
        //experiment.addMeasurement(ExperimentLoggerUtil::Measurement::Permute, timer.seconds());
        timer.reset();

        Kokkos::parallel_for("Heaviest HN", n, KOKKOS_LAMBDA(sgp_vid_t i) {
            sgp_vid_t hn_i = SGP_INFTY;
            sgp_vid_t order_max = 0;
            sgp_wgt_t max_ewt = 0;

            bool same_part_found = false;

            sgp_eid_t end_offset = g.graph.row_map(i + 1);
            for (sgp_eid_t j = g.graph.row_map(i); j < end_offset; j++) {
                sgp_wgt_t wgt = g.values(j);
                sgp_vid_t v = g.graph.entries(j);
                sgp_vid_t order = reverse_map(v);
                bool choose = false;
                if (same_part_found) {
                    if (part(i) == part(v)) {
                        if (max_ewt < wgt) {
                            choose = true;
                        }
                        else if (max_ewt == wgt && order_max <= order) {
                            choose = true;
                        }
                    }
                }
                else {
                    if (part(i) == part(v)) {
                        choose = true;
                        same_part_found = true;
                    }
                    // else {
                    //     if (max_ewt < wgt) {
                    //         choose = true;
                    //     }
                    //     else if (max_ewt == wgt && order_max <= order) {
                    //         choose = true;
                    //     }
                    // }
                }

                if (choose) {
                    max_ewt = wgt;
                    order_max = order;
                    hn_i = v;
                }
            }
            if (hn_i == SGP_INFTY) {
                sgp_vid_t rand = reverse_map(i);
                for (sgp_vid_t j = rand + 1; j < n; j++) {
                    sgp_vid_t v = vperm(j);
                    if (part(i) == part(v)) {
                        hn_i = v;
                        break;
                    }
                }
                if (hn_i == SGP_INFTY) {
                    for (sgp_vid_t j = 0; j < rand; j++) {
                        sgp_vid_t v = vperm(j);
                        if (part(i) == part(v)) {
                            hn_i = v;
                            break;
                        }
                    }
                }
            }
            if (hn_i == SGP_INFTY) {
                hn_i = i;
            }
            hn(i) = hn_i;
        });

        timer.reset();
        sgp_vid_t nc = parallel_map_construct(vcmap, n, vperm, hn, reverse_map);
        //experiment.addMeasurement(ExperimentLoggerUtil::Measurement::MapConstruct, timer.seconds());
        timer.reset();

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

    sgp_vid_t countInf(vtx_view_t target) {
        sgp_vid_t totalInf = 0;

        Kokkos::parallel_reduce(target.extent(0), KOKKOS_LAMBDA(sgp_vid_t i, sgp_vid_t & thread_sum) {
            if (target(i) == SGP_INFTY) {
                thread_sum++;
            }
        }, totalInf);

        return totalInf;
    }

    struct MatchByHashSorted {
        vtx_view_t vcmap, unmapped;
        Kokkos::View<uint32_t*> hashes;
        sgp_vid_t unmapped_total;
        Kokkos::View<sgp_vid_t> nvertices_coarse;
        MatchByHashSorted(vtx_view_t vcmap,
            vtx_view_t unmapped,
            Kokkos::View<uint32_t*> hashes,
            sgp_vid_t unmapped_total,
            Kokkos::View<sgp_vid_t> nvertices_coarse) :
            vcmap(vcmap),
            unmapped(unmapped),
            hashes(hashes),
            unmapped_total(unmapped_total),
            nvertices_coarse(nvertices_coarse) {}

        KOKKOS_INLINE_FUNCTION
            void operator()(const sgp_vid_t i, sgp_vid_t& update, const bool final) const {

            sgp_vid_t u = unmapped(i);
            sgp_vid_t tentative = 0;
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
                sgp_vid_t isOddOffset = (i - update) & 1;
                //if even (0 counts as even) we match unmapped(i) with unmapped(i+1) if hash(i) == hash(i+1)
                //if odd do nothing
                if (isOddOffset == 0) {
                    if (i + 1 < unmapped_total) {
                        if (hashes(i) == hashes(i + 1)) {
                            sgp_vid_t v = unmapped(i + 1);
                            vcmap(u) = Kokkos::atomic_fetch_add(&nvertices_coarse(), 1);
                            vcmap(v) = vcmap(u);
                        }
                    }
                }
            }
        }

        KOKKOS_INLINE_FUNCTION
            void join(volatile sgp_vid_t& update, volatile const sgp_vid_t& input) const {
            if (input > update) update = input;
        }

    };


    SGPAR_API int sgp_recoarsen_match(matrix_type& interp,
        sgp_vid_t* nvertices_coarse_ptr,
        const matrix_type& g,
        const int coarsening_level,
        const eigenview_t part,
        ExperimentLoggerUtil& experiment) {

        sgp_vid_t n = g.numRows();

        vtx_view_t hn("heavies", n);

        vtx_view_t vcmap("vcmap", n);

        Kokkos::parallel_for("initialize vcmap", n, KOKKOS_LAMBDA(sgp_vid_t i) {
            vcmap(i) = SGP_INFTY;
        });

        Kokkos::View<uint64_t*> randoms("randoms", n);

        pool_t rand_pool(std::time(nullptr));
        Kokkos::Timer timer;

        vtx_view_t vperm = generate_permutation(n, rand_pool);

        vtx_view_t reverse_map("reversed", n);
        Kokkos::parallel_for("construct reverse map", n, KOKKOS_LAMBDA(sgp_vid_t i) {
            reverse_map(vperm(i)) = i;
        });
        experiment.addMeasurement(ExperimentLoggerUtil::Measurement::Permute, timer.seconds());
        timer.reset();
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

            //find a valid heavy edge match
            //add the ones that are matchable to next round
            //maybe count these then create next_perm to save memory?
            Kokkos::parallel_for(perm_length, KOKKOS_LAMBDA(sgp_vid_t i){
                sgp_vid_t u = vperm(i);
                if (vcmap(u) == SGP_INFTY) {
                    sgp_vid_t h = SGP_INFTY;

                    sgp_wgt_t max_ewt = 0;
                    uint64_t max_rand = 0;
                    gen_t generator = rand_pool.get_state();
                    for (sgp_eid_t j = g.graph.row_map(u); j < g.graph.row_map(u + 1); j++) {
                        sgp_vid_t v = g.graph.entries(j);
                        //v must be unmatched and in same partition to be considered
                        if (vcmap(v) == SGP_INFTY && part(v) == part(u)) {
                            bool swap = false;
                            uint64_t rand = generator.urand64();
                            sgp_wgt_t wgt = g.values(j);

                            if (max_ewt < wgt) {
                                swap = true;
                            }
                            else if (max_ewt == wgt) {
                                //using <= so that zero rand may still be chosen
                                if (max_rand <= rand) {
                                    swap = true;
                                }
                            }
                            if (swap) {
                                h = v;
                                max_ewt = wgt;
                                max_rand = rand;
                            }
                        }
                    }
                    rand_pool.free_state(generator);

                    if (h != SGP_INFTY) {
                        sgp_vid_t add_next = Kokkos::atomic_fetch_add(&next_length(), 1);
                        next_perm(add_next) = u;
                        hn(u) = h;
                    }
                }
            });
            Kokkos::fence();
            Kokkos::deep_copy(perm_length, next_length);
            vperm = next_perm;

            if (perm_length > 0) {
                //match vertices with heaviest unmatched edge
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
                                match(u) = SGP_INFTY;
                            }
                        }
                    }
                });
                Kokkos::fence();
            }

            swap = swap ^ 1;
        }

#ifdef MTMETIS
        sgp_vid_t unmapped = countInf(vcmap);
        double unmappedRatio = static_cast<double>(unmapped) / static_cast<double>(n);

        //leaf matches
        if (unmappedRatio > 0.25) {
            Kokkos::parallel_for(n, KOKKOS_LAMBDA(sgp_vid_t u){
                if (vcmap(u) != SGP_INFTY) {
                    for (sgp_real_t k = 0.0; k < 1.1; k++) {
                        sgp_vid_t lastLeaf = SGP_INFTY;
                        for (sgp_eid_t j = g.graph.row_map(u); j < g.graph.row_map(u + 1); j++) {
                            sgp_vid_t v = g.graph.entries(j);
                            //v must be unmatched and in partition k to be considered
                            if (vcmap(v) == SGP_INFTY && part(v) == k) {
                                //must be degree 1 to be a leaf
                                if (g.graph.row_map(v + 1) - g.graph.row_map(v) == 1) {
                                    if (lastLeaf == SGP_INFTY) {
                                        lastLeaf = v;
                                    }
                                    else {
                                        vcmap(lastLeaf) = Kokkos::atomic_fetch_add(&nvertices_coarse(), 1);
                                        vcmap(v) = vcmap(lastLeaf);
                                        lastLeaf = SGP_INFTY;
                                    }
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
            Kokkos::View<uint32_t*> hashes("hashes", unmapped);

            Kokkos::View<sgp_vid_t> unmappedIdx("unmapped index");
            hasher_t hasher;
            //compute digests of adjacency lists
            Kokkos::parallel_for("create digests", policy(n, Kokkos::AUTO), KOKKOS_LAMBDA(const member & thread) {
                sgp_vid_t u = thread.league_rank();
                if (vcmap(u) == SGP_INFTY) {
                    uint32_t hash = 0;
                    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(thread, g.graph.row_map(u), g.graph.row_map(u + 1)), [=](const sgp_eid_t j, uint32_t& thread_sum) {
                        thread_sum += hasher(g.graph.entries(j));
                        }, hash);
                    Kokkos::single(Kokkos::PerTeam(thread), [=]() {
                        hash += hasher(part(u));
                        sgp_vid_t idx = Kokkos::atomic_fetch_add(&unmappedIdx(), 1);
                        unmappedVtx(idx) = u;
                        hashes(idx) = hash;
                        });
                }
            });
            uint32_t max = std::numeric_limits<uint32_t>::max();
            typedef Kokkos::BinOp1D< Kokkos::View<uint32_t*> > BinOp;
            BinOp bin_op(unmapped, 0, max);
            //VERY important that final parameter is true
            Kokkos::BinSort< Kokkos::View<uint32_t*>, BinOp, Kokkos::DefaultExecutionSpace, sgp_vid_t >
                sorter(hashes, bin_op, true);
            sorter.create_permute_vector();
            sorter.template sort< Kokkos::View<uint32_t*> >(hashes);
            sorter.template sort< vtx_view_t >(unmappedVtx);

            MatchByHashSorted matchTwinFunctor(vcmap, unmappedVtx, hashes, unmapped, nvertices_coarse);
            Kokkos::parallel_scan("match twins", unmapped, matchTwinFunctor);
        }

        unmapped = countInf(vcmap);
        unmappedRatio = static_cast<double>(unmapped) / static_cast<double>(n);

        //relative matches
        if (unmappedRatio > 0.25) {

            //get possibly mappable vertices of unmapped
            vtx_view_t mappableVtx("mappable vertices", unmapped);
            Kokkos::parallel_scan("get unmapped", n, KOKKOS_LAMBDA(const sgp_vid_t i, sgp_vid_t & update, const bool final){
                if (vcmap(i) == SGP_INFTY) {
                    if (final) {
                        mappableVtx(update) = i;
                    }

                    update++;
                }
            });


            sgp_vid_t mappable_count = unmapped;
            do {

                Kokkos::parallel_for("reset hn", mappable_count, KOKKOS_LAMBDA(sgp_vid_t i){
                    sgp_vid_t u = mappableVtx(i);
                    hn(u) = SGP_INFTY;
                });

                //choose relatives for unmapped vertices
                Kokkos::parallel_for("assign relatives", n, KOKKOS_LAMBDA(sgp_vid_t i){
                    if (vcmap(i) != SGP_INFTY) {
                        for (sgp_real_t k = 0.0; k < 1.1; k++) {
                            sgp_vid_t last_free = SGP_INFTY;
                            for (sgp_eid_t j = g.graph.row_map(i); j < g.graph.row_map(i + 1); j++) {
                                sgp_vid_t v = g.graph.entries(j);
                                if (vcmap(v) == SGP_INFTY && part(v) == k) {
                                    if (last_free != SGP_INFTY) {
                                        //there can be multiple threads updating this but it doesn't matter as long as they have some value
                                        hn(last_free) = v;
                                        hn(v) = last_free;
                                        last_free = SGP_INFTY;
                                    }
                                    else {
                                        last_free = v;
                                    }
                                }
                            }
                        }
                    }
                });

                //create a list of all mappable vertices according to set entries of hn
                sgp_vid_t old_mappable = mappable_count;
                mappable_count = 0;
                Kokkos::parallel_reduce("count mappable", old_mappable, KOKKOS_LAMBDA(const sgp_vid_t i, sgp_vid_t & thread_sum){
                    sgp_vid_t u = mappableVtx(i);
                    if (hn(u) != SGP_INFTY) {
                        thread_sum++;
                    }
                }, mappable_count);

                vtx_view_t nextMappable("next mappable vertices", mappable_count);

                Kokkos::parallel_scan("get next mappable", old_mappable, KOKKOS_LAMBDA(const sgp_vid_t i, sgp_vid_t & update, const bool final){
                    sgp_vid_t u = mappableVtx(i);
                    if (hn(u) != SGP_INFTY) {
                        if (final) {
                            nextMappable(update) = u;
                        }

                        update++;
                    }
                });
                mappableVtx = nextMappable;

                //match vertices with chosen relative
                if (mappable_count > 0) {
                    Kokkos::parallel_for(mappable_count, KOKKOS_LAMBDA(sgp_vid_t i){
                        sgp_vid_t u = mappableVtx(i);
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
                                    match(u) = SGP_INFTY;
                                }
                            }
                        }
                    });
                }
                Kokkos::fence();
                swap = swap ^ 1;
            } while (mappable_count > 0);
        }
#endif

        //create singleton aggregates of remaining unmatched vertices
        Kokkos::parallel_for(n, KOKKOS_LAMBDA(sgp_vid_t i){
            if (vcmap(i) == SGP_INFTY) {
                vcmap(i) = Kokkos::atomic_fetch_add(&nvertices_coarse(), 1);
            }
        });

        experiment.addMeasurement(ExperimentLoggerUtil::Measurement::MapConstruct, timer.seconds());
        timer.reset();

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

    SGPAR_API int sgp_coarsen_match(matrix_type& interp,
        sgp_vid_t* nvertices_coarse_ptr,
        const matrix_type& g,
        const int coarsening_level,
        sgp_pcg32_random_t* rng,
        ExperimentLoggerUtil& experiment) {

        sgp_vid_t n = g.numRows();

        vtx_view_t hn("heavies", n);

        vtx_view_t vcmap("vcmap", n);

        Kokkos::parallel_for("initialize vcmap", n, KOKKOS_LAMBDA(sgp_vid_t i) {
            vcmap(i) = SGP_INFTY;
        });

        Kokkos::View<uint64_t*> randoms("randoms", n);

        pool_t rand_pool(std::time(nullptr));
        Kokkos::Timer timer;

        vtx_view_t vperm = generate_permutation(n, rand_pool);

        vtx_view_t reverse_map("reversed", n);
        Kokkos::parallel_for("construct reverse map", n, KOKKOS_LAMBDA(sgp_vid_t i) {
            reverse_map(vperm(i)) = i;
        });
        experiment.addMeasurement(ExperimentLoggerUtil::Measurement::Permute, timer.seconds());
        timer.reset();

        if (coarsening_level == 1) {
            //all weights equal at this level so choose heaviest edge randomly
            Kokkos::parallel_for("Random HN", n, KOKKOS_LAMBDA(sgp_vid_t i) {
                gen_t generator = rand_pool.get_state();
                sgp_vid_t adj_size = g.graph.row_map(i + 1) - g.graph.row_map(i);
                sgp_vid_t offset = g.graph.row_map(i) + (generator.urand64() % adj_size);
                hn(i) = g.graph.entries(offset);
                rand_pool.free_state(generator);
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
        timer.reset();
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

            //match vertices with heaviest unmatched edge
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
                            match(u) = SGP_INFTY;
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
                    sgp_vid_t h = SGP_INFTY;

                    if (coarsening_level == 1) {
                        sgp_vid_t max_ewt = 0;
                        //we have to iterate over the edges anyways because we need to check if any are unmatched!
                        //so instead of randomly choosing a heaviest edge, we instead use the reverse permutation order as the weight
                        for (sgp_eid_t j = g.graph.row_map(u); j < g.graph.row_map(u + 1); j++) {
                            sgp_vid_t v = g.graph.entries(j);
                            //v must be unmatched to be considered
                            if (vcmap(v) == SGP_INFTY) {
                                //using <= so that zero weight edges may still be chosen
                                if (max_ewt <= reverse_map(v)) {
                                    max_ewt = reverse_map(v);
                                    h = v;
                                }
                            }
                        }
                    }
                    else {
                        sgp_wgt_t max_ewt = 0;
                        for (sgp_eid_t j = g.graph.row_map(u); j < g.graph.row_map(u + 1); j++) {
                            sgp_vid_t v = g.graph.entries(j);
                            //v must be unmatched to be considered
                            if (vcmap(v) == SGP_INFTY) {
                                //using <= so that zero weight edges may still be chosen
                                if (max_ewt <= g.values(j)) {
                                    max_ewt = g.values(j);
                                    h = v;
                                }
                            }
                        }
                    }

                    if (h != SGP_INFTY) {
                        sgp_vid_t add_next = Kokkos::atomic_fetch_add(&next_length(), 1);
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

#ifdef MTMETIS
        sgp_vid_t unmapped = countInf(vcmap);
        double unmappedRatio = static_cast<double>(unmapped) / static_cast<double>(n);

        //leaf matches
        if (unmappedRatio > 0.25) {
            Kokkos::parallel_for(n, KOKKOS_LAMBDA(sgp_vid_t u){
                if (vcmap(u) != SGP_INFTY) {
                    sgp_vid_t lastLeaf = SGP_INFTY;
                    for (sgp_eid_t j = g.graph.row_map(u); j < g.graph.row_map(u + 1); j++) {
                        sgp_vid_t v = g.graph.entries(j);
                        //v must be unmatched to be considered
                        if (vcmap(v) == SGP_INFTY) {
                            //must be degree 1 to be a leaf
                            if (g.graph.row_map(v + 1) - g.graph.row_map(v) == 1) {
                                if (lastLeaf == SGP_INFTY) {
                                    lastLeaf = v;
                                }
                                else {
                                    vcmap(lastLeaf) = Kokkos::atomic_fetch_add(&nvertices_coarse(), 1);
                                    vcmap(v) = vcmap(lastLeaf);
                                    lastLeaf = SGP_INFTY;
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
            Kokkos::View<uint32_t*> hashes("hashes", unmapped);

            Kokkos::View<sgp_vid_t> unmappedIdx("unmapped index");
            hasher_t hasher;
            //compute digests of adjacency lists
            Kokkos::parallel_for("create digests", policy(n, Kokkos::AUTO), KOKKOS_LAMBDA(const member & thread) {
                sgp_vid_t u = thread.league_rank();
                if (vcmap(u) == SGP_INFTY) {
                    uint32_t hash = 0;
                    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(thread, g.graph.row_map(u), g.graph.row_map(u + 1)), [=](const sgp_eid_t j, uint32_t& thread_sum) {
                        thread_sum += hasher(g.graph.entries(j));
                        }, hash);
                    Kokkos::single(Kokkos::PerTeam(thread), [=]() {
                        sgp_vid_t idx = Kokkos::atomic_fetch_add(&unmappedIdx(), 1);
                        unmappedVtx(idx) = u;
                        hashes(idx) = hash;
                        });
                }
            });
            uint32_t max = std::numeric_limits<uint32_t>::max();
            typedef Kokkos::BinOp1D< Kokkos::View<uint32_t*> > BinOp;
            BinOp bin_op(unmapped, 0, max);
            //VERY important that final parameter is true
            Kokkos::BinSort< Kokkos::View<uint32_t*>, BinOp, Kokkos::DefaultExecutionSpace, sgp_vid_t >
                sorter(hashes, bin_op, true);
            sorter.create_permute_vector();
            sorter.template sort< Kokkos::View<uint32_t*> >(hashes);
            sorter.template sort< vtx_view_t >(unmappedVtx);

            MatchByHashSorted matchTwinFunctor(vcmap, unmappedVtx, hashes, unmapped, nvertices_coarse);
            Kokkos::parallel_scan("match twins", unmapped, matchTwinFunctor);
        }

        unmapped = countInf(vcmap);
        unmappedRatio = static_cast<double>(unmapped) / static_cast<double>(n);

        //relative matches
        if (unmappedRatio > 0.25) {

            //get possibly mappable vertices of unmapped
            vtx_view_t mappableVtx("mappable vertices", unmapped);
            Kokkos::parallel_scan("get unmapped", n, KOKKOS_LAMBDA(const sgp_vid_t i, sgp_vid_t & update, const bool final){
                if (vcmap(i) == SGP_INFTY) {
                    if (final) {
                        mappableVtx(update) = i;
                    }

                    update++;
                }
            });


            sgp_vid_t mappable_count = unmapped;
            do {

                Kokkos::parallel_for("reset hn", mappable_count, KOKKOS_LAMBDA(sgp_vid_t i){
                    sgp_vid_t u = mappableVtx(i);
                    hn(u) = SGP_INFTY;
                });

                //choose relatives for unmapped vertices
                Kokkos::parallel_for("assign relatives", n, KOKKOS_LAMBDA(sgp_vid_t i){
                    if (vcmap(i) != SGP_INFTY) {
                        sgp_vid_t last_free = SGP_INFTY;
                        for (sgp_eid_t j = g.graph.row_map(i); j < g.graph.row_map(i + 1); j++) {
                            sgp_vid_t v = g.graph.entries(j);
                            if (vcmap(v) == SGP_INFTY) {
                                if (last_free != SGP_INFTY) {
                                    //there can be multiple threads updating this but it doesn't matter as long as they have some value
                                    hn(last_free) = v;
                                    hn(v) = last_free;
                                    last_free = SGP_INFTY;
                                }
                                else {
                                    last_free = v;
                                }
                            }
                        }
                    }
                });

                //create a list of all mappable vertices according to set entries of hn
                sgp_vid_t old_mappable = mappable_count;
                mappable_count = 0;
                Kokkos::parallel_reduce("count mappable", old_mappable, KOKKOS_LAMBDA(const sgp_vid_t i, sgp_vid_t & thread_sum){
                    sgp_vid_t u = mappableVtx(i);
                    if (hn(u) != SGP_INFTY) {
                        thread_sum++;
                    }
                }, mappable_count);

                vtx_view_t nextMappable("next mappable vertices", mappable_count);

                Kokkos::parallel_scan("get next mappable", old_mappable, KOKKOS_LAMBDA(const sgp_vid_t i, sgp_vid_t & update, const bool final){
                    sgp_vid_t u = mappableVtx(i);
                    if (hn(u) != SGP_INFTY) {
                        if (final) {
                            nextMappable(update) = u;
                        }

                        update++;
                    }
                });
                mappableVtx = nextMappable;

                //match vertices with chosen relative
                if (mappable_count > 0) {
                    Kokkos::parallel_for(mappable_count, KOKKOS_LAMBDA(sgp_vid_t i){
                        sgp_vid_t u = mappableVtx(i);
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
                                    match(u) = SGP_INFTY;
                                }
                            }
                        }
                    });
                }
                Kokkos::fence();
                swap = swap ^ 1;
            } while (mappable_count > 0);
        }
#endif

        //create singleton aggregates of remaining unmatched vertices
        Kokkos::parallel_for(n, KOKKOS_LAMBDA(sgp_vid_t i){
            if (vcmap(i) == SGP_INFTY) {
                vcmap(i) = Kokkos::atomic_fetch_add(&nvertices_coarse(), 1);
            }
        });

        experiment.addMeasurement(ExperimentLoggerUtil::Measurement::MapConstruct, timer.seconds());
        timer.reset();

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
}
}