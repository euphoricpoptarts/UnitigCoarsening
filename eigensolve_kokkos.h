#pragma once

#include "definitions_kokkos.h"
#include "KokkosBlas1_dot.hpp"
#include "coarsen_kokkos.h"
#include <limits>
#include <cstdlib>
#include <cmath>

namespace sgpar {
namespace sgpar_kokkos {

SGPAR_API int sgp_vec_normalize_kokkos(eigenview_t& u, sgp_vid_t n) {
    sgp_real_t squared_sum = 0;

    squared_sum = KokkosBlas::dot(u, u);
    sgp_real_t sum_inv = 1 / sqrt(squared_sum);

    Kokkos::parallel_for(n, KOKKOS_LAMBDA(int64_t i) {
        u(i) = u(i) * sum_inv;
    });
    return EXIT_SUCCESS;
}


SGPAR_API int sgp_vec_orthogonalize_kokkos(eigenview_t& u1, eigenview_t& u2, sgp_vid_t n) {

    sgp_real_t mult1 = KokkosBlas::dot(u1,u2);

    Kokkos::parallel_for(n, KOKKOS_LAMBDA(sgp_vid_t i) {
        u1(i) -= mult1 * u2(i);
    });
    return EXIT_SUCCESS;
}

SGPAR_API int sgp_vec_D_orthogonalize_kokkos(eigenview_t& u1, eigenview_t& u2,
    Kokkos::View<sgp_wgt_t*>& D, sgp_vid_t n) {

    //u1[i] = u1[i] - (dot(u1, D*u2)/dot(u2, D*u2)) * u2[i]

    sgp_real_t mult1 = KokkosBlas::dot(u1, u2);

    sgp_real_t mult_numer = 0.0;
    sgp_real_t mult_denom = 0.0;

    Kokkos::parallel_reduce(n, KOKKOS_LAMBDA(const sgp_vid_t & i, sgp_real_t & thread_mult_numer) {
        thread_mult_numer += u1(i) * D(i) * u2(i);
    }, mult_numer);
    Kokkos::parallel_reduce(n, KOKKOS_LAMBDA(const sgp_vid_t & i, sgp_real_t & thread_mult_denom) {
        thread_mult_denom += u2(i) * D(i) * u2(i);
    }, mult_denom);

    Kokkos::parallel_for(n, KOKKOS_LAMBDA(sgp_vid_t i) {
        u1(i) -= mult_numer * u2(i) / mult_denom;
    });
    return EXIT_SUCCESS;
}

SGPAR_API void sgp_power_iter_eigenvalue_log(eigenview_t& u, const matrix_type& g) {
    sgp_real_t eigenval = 0;
    sgp_real_t eigenval_max = 0;
    sgp_real_t eigenval_min = 2;

    sgp_vid_t n = g.numRows();

    eigenview_t v("v", n);
    KokkosSparse::spmv("N", 1.0, g, u, 0.0, v);

    Kokkos::parallel_reduce(n, KOKKOS_LAMBDA(sgp_vid_t i, sgp_real_t& local_sum) {
        sgp_vid_t weighted_degree = g.graph.row_map(i + 1) - g.graph.row_map(i);
        sgp_real_t u_i = weighted_degree * u(i);
        sgp_real_t matvec_i = v(i);
        u_i -= matvec_i;
        v(i) = u_i;
        local_sum += (u_i * u_i) * 1e9;
    }, eigenval);

    Kokkos::parallel_reduce(n, KOKKOS_LAMBDA(sgp_vid_t i, sgp_real_t& local_max) {
        sgp_real_t eigenval_est = v(i) / u(i);
        if (local_max < eigenval_est) {
            local_max = eigenval_est;
        }
    }, Kokkos::Max<sgp_real_t, Kokkos::HostSpace>(eigenval_max));

    Kokkos::parallel_reduce(n, KOKKOS_LAMBDA(sgp_vid_t i, sgp_real_t& local_min) {
        sgp_real_t eigenval_est = v(i) / u(i);
        if (local_min > eigenval_est) {
            local_min = eigenval_est;
        }
    }, Kokkos::Min<sgp_real_t, Kokkos::HostSpace>(eigenval_min));

    printf("eigenvalue = %1.9lf (%1.9lf %1.9lf), "
        "edge cut lb %5.0lf "
        "gap ratio %.0lf\n",
        eigenval * 1e-9,
        eigenval_min, eigenval_max,
        eigenval * 1e-9 * n / 4,
        ceil(1.0 / (1.0 - eigenval * 1e-9)));
}

SGPAR_API int sgp_power_iter(eigenview_t& u, const matrix_type& g, int normLap, int final
#ifdef EXPERIMENT
    , ExperimentLoggerUtil& experiment
#endif
    ) {

    sgp_vid_t n = g.numRows();

    eigenview_t vec1("vec1", n);
    Kokkos::parallel_for(n, KOKKOS_LAMBDA(sgp_vid_t i) {
        vec1(i) = 1.0;
    });

    wgt_view_t weighted_degree("weighted degree",n);
    Kokkos::parallel_for("", n, KOKKOS_LAMBDA(sgp_vid_t i) {
        sgp_wgt_t degree_wt_i = 0;
        sgp_eid_t end_offset = g.graph.row_map(i + 1);
        for (sgp_eid_t j = g.graph.row_map(i); j < end_offset; j++) {
            degree_wt_i += g.values(j);
        }
        weighted_degree(i) = degree_wt_i;
    });

    sgp_wgt_t gb = 2.0;
    if (!normLap) {
        Kokkos::parallel_reduce("max weighted degree", n, KOKKOS_LAMBDA(sgp_vid_t i, sgp_wgt_t& local_max){
            sgp_wgt_t swap = 2.0 * weighted_degree(i);
            if (local_max < swap) {
                local_max = swap;
            }
        }, Kokkos::Max<sgp_wgt_t, Kokkos::HostSpace>(gb));
        if (gb < 2.0) {
            gb = 2.0;
        }
    }

    sgp_vec_normalize_kokkos(vec1, n);
    if (!normLap) {
        sgp_vec_orthogonalize_kokkos(u, vec1, n);
    }
    else {
        sgp_vec_D_orthogonalize_kokkos(u, vec1, weighted_degree, n);
    }
    sgp_vec_normalize_kokkos(u, n);

    eigenview_t v("v", n);
    //is this necessary?
    Kokkos::deep_copy(v, u);

    sgp_real_t tol = SGPAR_POWERITER_TOL;
    uint64_t niter = 0;
    uint64_t iter_max = (uint64_t)SGPAR_POWERITER_ITER / (uint64_t)n;
    sgp_real_t dotprod = 0, lastDotprod = 1;
    while (fabs(dotprod - lastDotprod) > tol && (niter < iter_max)) {

        Kokkos::deep_copy(u, v);

        KokkosSparse::spmv("N", 1.0, g, u, 0.0, v);

        // v = Lu
        Kokkos::parallel_for(n, KOKKOS_LAMBDA(sgp_vid_t i) {
            // sgp_real_t v_i = g.weighted_degree[i]*u[i];
            sgp_real_t weighted_degree_inv, v_i;
            if (!normLap) {
                v_i = (gb - weighted_degree(i)) * u(i);
            }
            else {
                weighted_degree_inv = 1.0 / weighted_degree(i);
                v_i = 0.5 * u(i);
            }
            // v_i -= matvec_i;
            if (!normLap) {
                v_i += v(i);
            }
            else {
                v_i += 0.5 * v(i) * weighted_degree_inv;
            }
            v(i) = v_i;
        });

        if (!normLap) {
            sgp_vec_orthogonalize_kokkos(v, vec1, n);
        }
        sgp_vec_normalize_kokkos(v, n);
        lastDotprod = dotprod;
        dotprod = KokkosBlas::dot(u, v);
        niter++;
    }
    int max_iter_reached = 0;
    if (niter >= iter_max) {
        printf("exceeded max iter count, ");
        max_iter_reached = 1;
    }
    printf("number of iterations: %lu\n", niter);

#ifdef EXPERIMENT
    experiment.addCoarseLevel(niter, max_iter_reached, n);
#endif

    if (!normLap && final) {
        sgp_power_iter_eigenvalue_log(u, g);
    }

    return EXIT_SUCCESS;
}

//edge cuts are bounded by |E| of finest graph (assuming unweighted edges for finest graph)
//also we assume ALL coarse edge weights are integral
sgp_eid_t fm_refine(eigenview_t& partition, const matrix_type& g, const vtx_view_t& vtx_w) {
    sgp_vid_t n = g.numRows();

    sgp_eid_t maxE = 0;

    Kokkos::parallel_reduce("max e find", policy(n, Kokkos::AUTO), KOKKOS_LAMBDA(const member& thread, sgp_eid_t & t_max){
        sgp_vid_t i = thread.league_rank();
        sgp_eid_t weighted_degree = 0;
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(thread, g.graph.row_map(i), g.graph.row_map(i + 1)), [=](const sgp_eid_t j, sgp_eid_t& local_sum) {
            local_sum += g.values(j);
        }, weighted_degree);
        Kokkos::single(Kokkos::PerTeam(thread), [&]() {
            if (t_max < weighted_degree) {
                t_max = weighted_degree;
            }
        });
    }, Kokkos::Max<sgp_eid_t, Kokkos::HostSpace>(maxE));

#ifdef _DEBUG
    printf("maxE: %u\n", maxE);
#endif

    sgp_eid_t totalBuckets = 2 * maxE + 1;
    vtx_view_t bucketsA("buckets part 1", totalBuckets), bucketsB("buckets part 2", totalBuckets);
    vtx_view_t ll_next("linked list nexts", n), ll_prev("linked list prevs", n);
    Kokkos::View<int64_t*> gains("gains", n), balances("balances", n);
    vtx_view_t swap_order("swap order", n);
    vtx_view_t free_vtx("free vertices", n);
    edge_view_t cutsizes("cutsizes", n);

    for (sgp_eid_t i = 0; i < 2 * maxE + 1; i++) {
        bucketsA(i) = SGP_INFTY;
        bucketsB(i) = SGP_INFTY;
    }

    for (sgp_vid_t i = 0; i < n; i++) {
        ll_next(i) = SGP_INFTY;
        ll_prev(i) = SGP_INFTY;
        free_vtx(i) = 1;
    }

    sgp_eid_t cutsize = 0;
    //initialize gains datastructure
    for (sgp_vid_t i = 0; i < n; i++) {
        sgp_real_t gain = 0;
        for (sgp_eid_t j = g.graph.row_map(i); j < g.graph.row_map(i + 1); j++) {
            sgp_vid_t v = g.graph.entries(j);
            if (partition(i) != partition(v)) {
                gain += g.values(j);
                cutsize += g.values(j);
            }
            else {
                gain -= g.values(j);
            }
        }
        gains(i) = gain;
        sgp_eid_t bucket = gain + maxE;
        vtx_view_t insert_into_bucket = bucketsA;
        if (partition(i) == 1.0) {
            insert_into_bucket = bucketsB;
        }
        sgp_vid_t bucket_first = insert_into_bucket(bucket);
        if (bucket_first != SGP_INFTY) {
            ll_prev(bucket_first) = i;
        }
        insert_into_bucket(bucket) = i;
        ll_next(i) = bucket_first;
    }

    int64_t balance = 0;
    for (sgp_vid_t i = 0; i < n; i++) {
        if (partition(i) == 0) {
            balance += vtx_w(i);
        }
        else {
            balance -= vtx_w(i);
        }
    }

    printf("Unrefined balance: %li, cutsize: %lu\n", balance, cutsize);

    int64_t start_balance = abs(balance);
    int64_t start_cut = cutsize;
    int64_t bucket_offsetA = 2 * maxE;
    int64_t bucket_offsetB = bucket_offsetA;
    sgp_vid_t total_swaps = 0;
    sgp_eid_t min_cut = start_cut;
    sgp_vid_t argmin = SGP_INFTY;
    int64_t min_imb = start_balance;
    bool start_counter = true;
    int counter = 0;
    while (bucket_offsetA >= 0 || bucket_offsetB >= 0) {
        sgp_vid_t swap_a = SGP_INFTY;
        if (bucket_offsetA >= 0) swap_a = bucketsA(bucket_offsetA);
        sgp_vid_t swap_b = SGP_INFTY; 
        if (bucket_offsetB >= 0) swap_b = bucketsB(bucket_offsetB);
        sgp_vid_t swap = SGP_INFTY;
        
        bool choose_a = false, choose_b = false;

        //select a vertex to swap and remove from datastructure
        if (swap_a != SGP_INFTY && swap_b == SGP_INFTY) {
            if (balance > 0 || bucket_offsetB < 0) {
                choose_a = true;
            }
            bucket_offsetB--;
        }
        else if (swap_a == SGP_INFTY && swap_b != SGP_INFTY) {
            if (balance < 0 || bucket_offsetA < 0) {
                choose_b = true;
            }
            bucket_offsetA--;
        }
        else if (swap_a != SGP_INFTY && swap_b != SGP_INFTY) {
            if (balance > 0) {
                choose_a = true;
            }
            else if (balance < 0) {
                choose_b = true;
            }
            else {
                if (bucket_offsetA > bucket_offsetB) {
                    choose_a = true;
                }
                else {
                    choose_b = true;
                }
            }
        }
        else {
            bucket_offsetA--;
            bucket_offsetB--;
        }

        if (choose_a) {
            swap = swap_a;
            sgp_vid_t next = ll_next(swap);
            bucketsA(bucket_offsetA) = next;
            if (next != SGP_INFTY) {
                ll_prev(next) = SGP_INFTY;
            }
        }
        else if (choose_b) {
            swap = swap_b;
            sgp_vid_t next = ll_next(swap);
            bucketsB(bucket_offsetB) = next;
            if (next != SGP_INFTY) {
                ll_prev(next) = SGP_INFTY;
            }
        }

        //swap and modify datastructure
        if (swap != SGP_INFTY) {
            //the gain only counts the outward edges, but in an undirected graphs these edges are duplicated from other vertices
            cutsize -= 2*gains(swap);
            if (partition(swap) == 0.0) {
                partition(swap) = 1.0;
                balance -= 2*vtx_w(swap);
            }
            else {
                partition(swap) = 0.0;
                balance += 2*vtx_w(swap);
            }
            free_vtx(swap) = 0;
            swap_order(total_swaps) = swap;
            cutsizes(total_swaps) = cutsize;
            balances(total_swaps) = balance;
            if(start_counter) counter++;

            //we ideally want both the cut and imbalance to improve
            //however we prioritize the balance, and bound how much the cut can decay by
            if (abs(balance) < min_imb && cutsize < 1.05 * min_cut) {
                min_imb = abs(balance);
                min_cut = cutsize;
                argmin = total_swaps;
                start_counter = true;
                counter = 0;
            }
            else if (abs(balance) <= min_imb && min_cut > cutsize) {
                min_imb = abs(balance);
                min_cut = cutsize;
                argmin = total_swaps;
                start_counter = true;
                counter = 0;
            }

            total_swaps++;
            for (sgp_eid_t j = g.graph.row_map(swap); j < g.graph.row_map(swap + 1); j++) {
                sgp_vid_t v = g.graph.entries(j);
                if (free_vtx(v) == 1) {
                    //multiply by 2 because this edge already counted towards v's gain, now it must count in the opposite direction
                    int64_t gain_change = 2*g.values(j);
                    if (partition(swap) == partition(v)) {
                        gain_change = -gain_change;
                    }
                    //connect previous and last nodes of ll together
                    sgp_vid_t old_ll_next = ll_next(v);
                    sgp_vid_t old_ll_prev = ll_prev(v);
                    if (old_ll_next != SGP_INFTY) {
                        ll_prev(old_ll_next) = old_ll_prev;
                    }
                    if (old_ll_prev != SGP_INFTY) {
                        ll_next(old_ll_prev) = old_ll_next;
                    }

                    int64_t old_gain = gains(v);
                    int64_t next_gain = old_gain + gain_change;
                    gains(v) = next_gain;
                    //old_ll_next becomes new head of ll if v was old head
                    if (old_ll_prev == SGP_INFTY) {
                        if (partition(v) == 0.0) {
                            bucketsA(old_gain + maxE) = old_ll_next;
                        }
                        else {
                            bucketsB(old_gain + maxE) = old_ll_next;
                        }
                    }

                    //old head of ll that v will be inserted into
                    sgp_vid_t new_ll_head = bucketsA(next_gain + maxE);
                    if (partition(v) == 0.0) {
                        bucketsA(next_gain + maxE) = v;
                        //need to move the seek head back if some entries are being moved up
                        if (next_gain + maxE > bucket_offsetA) {
                            bucket_offsetA = next_gain + maxE;
                        }
                    }
                    else {
                        new_ll_head = bucketsB(next_gain + maxE);
                        bucketsB(next_gain + maxE) = v;
                        if (next_gain + maxE > bucket_offsetB) {
                            bucket_offsetB = next_gain + maxE;
                        }
                    }

                    if (new_ll_head != SGP_INFTY) {
                        ll_prev(new_ll_head) = v;
                    }
                    ll_next(v) = new_ll_head;
                    ll_prev(v) = SGP_INFTY;
                }
            }
            if (counter > 500) {
                bucket_offsetA = -1;
                bucket_offsetB = -1;
            }
        }
    }

    sgp_vid_t undo_from = 0;
    if (argmin != SGP_INFTY) {
        undo_from = argmin + 1;
    }

    for (sgp_vid_t i = undo_from; i < total_swaps; i++) {
        sgp_vid_t undo = swap_order(i);
        if (partition(undo) == 0.0) {
            partition(undo) = 1.0;
        }
        else {
            partition(undo) = 0.0;
        }
    }

    printf("Refined balance: %li, cutsize: %lu\n", min_imb, min_cut);
    return min_cut;
}

eigenview_t init_gggp(const matrix_type& cg,
                      const vtx_view_t& c_vtx_w){
    printf("Doing GGGP\n");

    sgp_vid_t gc_n = cg.numRows();

    sgp_vid_t vtx_w_total = 0;

    printf("coarse vertex count: %u\n", gc_n);
    for (sgp_vid_t i = 0; i < c_vtx_w.extent(0); i++) {
        vtx_w_total += c_vtx_w(i);
    }

    edge_mirror_t row_map = Kokkos::create_mirror(cg.graph.row_map);
    vtx_mirror_t entries = Kokkos::create_mirror(cg.graph.entries);
    wgt_mirror_t values = Kokkos::create_mirror(cg.values);

    Kokkos::deep_copy(row_map, cg.graph.row_map);
    Kokkos::deep_copy(entries, cg.graph.entries);
    Kokkos::deep_copy(values, cg.values);

    eigenview_t cg_m("coarse_guess", gc_n);
    eigenview_t best_cg_part("best cg part", gc_n);
    sgp_real_t cutmin = SGP_INFTY;
    for (sgp_vid_t i = 0; i < gc_n; i++) {
        //reset coarse partition
        for (sgp_vid_t j = 0; j < gc_n; j++) {
            cg_m(j) = 0;
        }
        cg_m(i) = 1;
        sgp_vid_t count = c_vtx_w(i);
        //incrementally grow partition 1
        while (count < vtx_w_total / 2) {
            sgp_vid_t argmin = i;
            sgp_real_t min = SGP_INFTY;
            //find minimum increase to cutsize for moving a vertex from partition 0 to partition 1
            for (sgp_vid_t u = 0; u < gc_n; u++) {
                if (cg_m(u) == 0) {
                    sgp_real_t cutLoss = 0;
                    for (sgp_eid_t j = row_map(u); j < row_map(u + 1); j++) {
                        sgp_vid_t v = entries(j);
                        if (cg_m(v) == 0) {
                            cutLoss += values(j);
                        }
                        else {
                            cutLoss -= values(j);
                        }
                    }
                    if (cutLoss < min) {
                        min = cutLoss;
                        argmin = u;
                    }
                }
            }
            cg_m(argmin) = 1;
            count += c_vtx_w(argmin);
        }
        sgp_real_t edge_cut = 0;
        //find total cutsize
        for (sgp_vid_t u = 0; u < gc_n; u++) {
            for (sgp_eid_t j = row_map(u); j < row_map(u + 1); j++) {
                sgp_vid_t v = entries(j);
                if (cg_m(v) != cg_m(u)) {
                    edge_cut += values(j);
                }
            }
        }
        //if cutsize less than best, replace best with current
        if (edge_cut < cutmin) {
            cutmin = edge_cut;
            for (sgp_vid_t j = 0; j < gc_n; j++) {
                best_cg_part(j) = cg_m(j);
            }
        }
    }
    return best_cg_part;
}

eigenview_t sgp_recoarsen_one_level(const matrix_type& g,
    const vtx_view_t& f_vtx_w,
    const eigenview_t partition,
    sgp_eid_t min_cut, sgp_eid_t& out_cut, int refine_layer, bool auto_replace = false) {

    sgp_eid_t last_cut = min_cut;
    eigenview_t fine_part = partition;
    printf("recoarsening level %d\n", refine_layer);
    while(true){
        last_cut = min_cut;
        matrix_type gc;
        matrix_type interpolation_graph;
        sgp_vid_t nvertices_coarse;
        vtx_view_t c_vtx_w;
        sgp_recoarsen_HEC(interpolation_graph, &nvertices_coarse, g, fine_part);

        ExperimentLoggerUtil throwaway;
        sgp_build_coarse_graph_msd(gc, c_vtx_w, interpolation_graph, g, f_vtx_w, 2, throwaway);


        eigenview_t coarse_part("coarser partition", nvertices_coarse);
        if(gc.numRows() < 5){
           return fine_part;
        }
        if(gc.numRows() < 30){
            coarse_part = init_gggp(gc, c_vtx_w);
                printf("stop recoarsening level %d\n", refine_layer);
            sgp_eid_t last_cut2 = min_cut;
            do{
                last_cut2 = min_cut;
                min_cut = fm_refine(coarse_part, gc, c_vtx_w);
            } while(last_cut2 != min_cut);
            if(auto_replace || min_cut < last_cut){
                eigenview_t fine_recoarsened("new fine partition", g.numRows());
                KokkosSparse::spmv("N", 1.0, interpolation_graph, coarse_part, 0.0, fine_recoarsened);
                fine_part = fine_recoarsened;
            out_cut = min_cut;
            }
            return fine_part;
        }
        Kokkos::parallel_for("create coarse partition", g.numRows(), KOKKOS_LAMBDA(sgp_vid_t i) {
            sgp_vid_t coarse_vtx = interpolation_graph.graph.entries(i);
            double part_wgt = static_cast<double>(f_vtx_w(i)) / static_cast<double>(c_vtx_w(coarse_vtx));
            Kokkos::atomic_add(&coarse_part(coarse_vtx), part_wgt * fine_part(i));
        });
        //not strictly necessary but you never know with floating point rounding errors
        Kokkos::parallel_for("discretize partition", nvertices_coarse, KOKKOS_LAMBDA(sgp_vid_t i) {
            coarse_part(i) = round(coarse_part(i));
        });

        sgp_eid_t last_cut2 = min_cut;
        do{
            last_cut2 = min_cut;
            min_cut = fm_refine(coarse_part, gc, c_vtx_w);
        } while(last_cut2 != min_cut);
        coarse_part = sgp_recoarsen_one_level(gc, c_vtx_w, coarse_part, min_cut, min_cut, refine_layer + 1, true);
        do{
            last_cut2 = min_cut;
            min_cut = fm_refine(coarse_part, gc, c_vtx_w);
        } while(last_cut2 != min_cut);
        if(auto_replace || min_cut < last_cut){
            eigenview_t fine_recoarsened("new fine partition", g.numRows());
            KokkosSparse::spmv("N", 1.0, interpolation_graph, coarse_part, 0.0, fine_recoarsened);
            fine_part = fine_recoarsened;
        out_cut = min_cut;
        }
        if(min_cut > 0.999*last_cut) {
            printf("stop recoarsening level %d\n", refine_layer);
            return fine_part;
        }
    }
}

SGPAR_API int sgp_eigensolve(sgp_real_t* eigenvec, std::list<matrix_type>& graphs, std::list<matrix_type>& interpolates, std::list<vtx_view_t>& vtx_weights, sgp_pcg32_random_t* rng, int refine_alg
#ifdef EXPERIMENT
    , ExperimentLoggerUtil& experiment
#endif
) {

    sgp_vid_t gc_n = graphs.rbegin()->numRows();
    eigenview_t coarse_guess("coarse_guess", gc_n);
    eigenview_t::HostMirror cg_m = Kokkos::create_mirror(coarse_guess);
    //randomly initialize guess eigenvector for coarsest graph
#ifndef FM
    for (sgp_vid_t i = 0; i < gc_n; i++) {
        cg_m(i) = ((double)sgp_pcg32_random_r(rng)) / UINT32_MAX;
    }
    Kokkos::deep_copy(coarse_guess, cg_m);
    sgp_vec_normalize_kokkos(coarse_guess, gc_n);
#else
    matrix_type cg = *graphs.rbegin();
    vtx_view_t c_vtx_w = *vtx_weights.rbegin();

    coarse_guess = init_gggp(cg, c_vtx_w);
#endif


    auto graph_iter = graphs.rbegin(), interp_iter = interpolates.rbegin();
    auto vtx_w_iter = vtx_weights.rbegin();
    auto end = --graphs.rend();

    int refine_layer = graphs.size();
    //there is always one more refinement than interpolation
    sgp_eid_t cutsize = SGP_INFTY;
    while (graph_iter != end) {
        //refine
#ifdef FM
        printf("Refining layer %i\n", refine_layer);
        refine_layer--;
        sgp_eid_t old_cutsize = cutsize;
        do {
            old_cutsize = cutsize;
            cutsize = fm_refine(coarse_guess, *graph_iter, *vtx_w_iter);
        } while (cutsize != old_cutsize); //could be larger if the balance improved
        coarse_guess = sgp_recoarsen_one_level(*graph_iter, *vtx_w_iter, coarse_guess, cutsize, cutsize, refine_layer);
	do {
            old_cutsize = cutsize;
            cutsize = fm_refine(coarse_guess, *graph_iter, *vtx_w_iter);
        } while (cutsize != old_cutsize); //could be larger if the balance improved
#else
        CHECK_SGPAR(sgp_power_iter(coarse_guess, *graph_iter, refine_alg, 0
#ifdef EXPERIMENT
            , experiment
#endif
            ));
#endif
        graph_iter++;
        vtx_w_iter++;

        //interpolate
        eigenview_t fine_vec("fine vec", graph_iter->numRows());
        KokkosSparse::spmv("N", 1.0, *interp_iter, coarse_guess, 0.0, fine_vec);
        coarse_guess = fine_vec;
        interp_iter++;
    }

#ifdef FM
    printf("Refining layer %i\n", refine_layer);
    refine_layer--;
    sgp_eid_t old_cutsize = cutsize;
    do {
        old_cutsize = cutsize;
        cutsize = fm_refine(coarse_guess, *graph_iter, *vtx_w_iter);
    } while (cutsize != old_cutsize); //could be larger if the balance improved
    coarse_guess = sgp_recoarsen_one_level(*graph_iter, *vtx_w_iter, coarse_guess, cutsize, cutsize, refine_layer);
    do {
        old_cutsize = cutsize;
        cutsize = fm_refine(coarse_guess, *graph_iter, *vtx_w_iter);
    } while (cutsize != old_cutsize); //could be larger if the balance improved
#else
    //last refine
    CHECK_SGPAR(sgp_power_iter(coarse_guess, *graph_iter, refine_alg, 1
#ifdef EXPERIMENT
        , experiment
#endif
        ));
#endif

    eigenview_t::HostMirror eigenmirror = Kokkos::create_mirror(coarse_guess);
    Kokkos::deep_copy(eigenmirror, coarse_guess);

    Kokkos::parallel_for(host_policy(0, graph_iter->numRows()), KOKKOS_LAMBDA(sgp_vid_t i) {
        eigenvec[i] = eigenmirror(i);
    });

    return EXIT_SUCCESS;
}

}
}
