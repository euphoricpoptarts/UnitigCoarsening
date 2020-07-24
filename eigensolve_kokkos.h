#pragma once

#include "definitions_kokkos.h"
#include "KokkosBlas1_dot.hpp"

namespace sgpar {
namespace sgpar_kokkos {

using eigenview_t = Kokkos::View<sgp_real_t*>;

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

SGPAR_API int sgp_eigensolve(sgp_real_t* eigenvec, std::list<matrix_type>& graphs, std::list<matrix_type>& interpolates, sgp_pcg32_random_t* rng, int refine_alg
#ifdef EXPERIMENT
    , ExperimentLoggerUtil& experiment
#endif
) {

    sgp_vid_t gc_n = graphs.rbegin()->numRows();
    eigenview_t coarse_guess("coarse_guess", gc_n);
    eigenview_t::HostMirror cg_m = Kokkos::create_mirror(coarse_guess);
    //randomly initialize guess eigenvector for coarsest graph
    if (gc_n > 200) {
        for (sgp_vid_t i = 0; i < gc_n; i++) {
            cg_m(i) = ((double)sgp_pcg32_random_r(rng)) / UINT32_MAX;
        }
    }
    else {
        printf("Doing GGGP\n");
        matrix_type cg = *graphs.rbegin();
        edge_mirror_t row_map = Kokkos::create_mirror(cg.graph.row_map);
        vtx_mirror_t entries = Kokkos::create_mirror(cg.graph.entries);
        wgt_mirror_t values = Kokkos::create_mirror(cg.values);

        Kokkos::deep_copy(row_map, cg.graph.row_map);
        Kokkos::deep_copy(entries, cg.graph.entries);
        Kokkos::deep_copy(values, cg.values);

        eigenview_t::HostMirror best_cg_part = Kokkos::create_mirror(coarse_guess);
        sgp_real_t cutmin = SGP_INFTY;
        for (sgp_vid_t i = 0; i < gc_n; i++) {
            //reset coarse partition
            for (sgp_vid_t j = 0; j < gc_n; j++) {
                cg_m(j) = 0;
            }
            cg_m(i) = 1;
            sgp_vid_t count = 1;
            //incrementally grow partition 1
            while (count < gc_n / 2) {
                sgp_vid_t argmin = i;
                sgp_real_t min = SGP_INFTY;
                //find minimum increase to cutsize
                for (sgp_vid_t u = 0; u < gc_n; u++) {
                    if (u != i) {
                        sgp_real_t cutLoss = 0;
                        for (sgp_eid_t j = row_map(u); j < row_map(u + 1); j++) {
                            sgp_vid_t v = entries(j);
                            if (cg_m(v) == 0) {
                                cutLoss += values(j);
                            }
                        }
                        if (cutLoss < min) {
                            min = cutLoss;
                            argmin = u;
                        }
                    }
                }
                cg_m(argmin) = 1;
                count++;
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
        cg_m = best_cg_part;
    }
    Kokkos::deep_copy(coarse_guess, cg_m);
    sgp_vec_normalize_kokkos(coarse_guess, gc_n);

    auto graph_iter = graphs.rbegin(), interp_iter = interpolates.rbegin();
    auto end = --graphs.rend();

    //there is always one more refinement than interpolation
    while (graph_iter != end) {
        //refine
        CHECK_SGPAR(sgp_power_iter(coarse_guess, *graph_iter, refine_alg, 0
#ifdef EXPERIMENT
            , experiment
#endif
            ));
        graph_iter++;

        //interpolate
        eigenview_t fine_vec("fine vec", graph_iter->numRows());
        KokkosSparse::spmv("N", 1.0, *interp_iter, coarse_guess, 0.0, fine_vec);
        coarse_guess = fine_vec;
        interp_iter++;
    }

    //last refine
    CHECK_SGPAR(sgp_power_iter(coarse_guess, *graph_iter, refine_alg, 1
#ifdef EXPERIMENT
        , experiment
#endif
        ));

    eigenview_t::HostMirror eigenmirror = Kokkos::create_mirror(coarse_guess);
    Kokkos::deep_copy(eigenmirror, coarse_guess);

    Kokkos::parallel_for(host_policy(0, graph_iter->numRows()), KOKKOS_LAMBDA(sgp_vid_t i) {
        eigenvec[i] = eigenmirror(i);
    });

    return EXIT_SUCCESS;
}

}
}