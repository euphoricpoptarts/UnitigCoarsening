#pragma once

#include "definitions_kokkos.h"

SGPAR_API int sgp_vec_normalize_kokkos(sgp_real_t* u, sgp_vid_t n) {

    assert(u != NULL);
    sgp_real_t squared_sum = 0;

    Kokkos::parallel_reduce(n, KOKKOS_LAMBDA(const sgp_vid_t & i, sgp_real_t & thread_squared_sum) {
        thread_squared_sum += u[i] * u[i];
    }, squared_sum);
    sgp_real_t sum_inv = 1 / sqrt(squared_sum);

    Kokkos::parallel_for(n, KOKKOS_LAMBDA(int64_t i) {
        u[i] = u[i] * sum_inv;
    });
    return EXIT_SUCCESS;
}

SGPAR_API int sgp_vec_dotproduct_kokkos(sgp_real_t* dot_prod_ptr,
    sgp_real_t* u1, sgp_real_t* u2, sgp_vid_t n) {

    sgp_real_t dot_prod = 0;

    Kokkos::parallel_reduce(n, KOKKOS_LAMBDA(const sgp_vid_t & i, sgp_real_t & thread_dot_prod) {
        thread_dot_prod += u1[i] * u2[i];
    }, dot_prod);
    *dot_prod_ptr = dot_prod;
    return EXIT_SUCCESS;
}


SGPAR_API int sgp_vec_orthogonalize_kokkos(sgp_real_t* u1, sgp_real_t* u2, sgp_vid_t n) {

    sgp_real_t mult1;
    sgp_vec_dotproduct_kokkos(&mult1, u1, u2, n);

    Kokkos::parallel_for(n, KOKKOS_LAMBDA(sgp_vid_t i) {
        u1[i] -= mult1 * u2[i];
    });
    return EXIT_SUCCESS;
}

SGPAR_API int sgp_vec_D_orthogonalize_kokkos(sgp_real_t* u1, sgp_real_t* u2,
    sgp_wgt_t* D, sgp_vid_t n) {

    //u1[i] = u1[i] - (dot(u1, D*u2)/dot(u2, D*u2)) * u2[i]

    sgp_real_t mult1;
    sgp_vec_dotproduct_kokkos(&mult1, u1, u2, n);

    sgp_real_t mult_numer = 0.0;
    sgp_real_t mult_denom = 0.0;

    Kokkos::parallel_reduce(n, KOKKOS_LAMBDA(const sgp_vid_t & i, sgp_real_t & thread_mult_numer) {
        thread_mult_numer += u1[i] * D[i] * u2[i];
    }, mult_numer);
    Kokkos::parallel_reduce(n, KOKKOS_LAMBDA(const sgp_vid_t & i, sgp_real_t & thread_mult_denom) {
        thread_mult_denom += u2[i] * D[i] * u2[i];
    }, mult_denom);

    Kokkos::parallel_for(n, KOKKOS_LAMBDA(sgp_vid_t i) {
        u1[i] -= mult_numer * u2[i] / mult_denom;
    });
    return EXIT_SUCCESS;
}

SGPAR_API void sgp_power_iter_eigenvalue_log(sgp_real_t* u, sgp_graph_t g) {
    sgp_real_t eigenval = 0;
    sgp_real_t eigenval_max = 0;
    sgp_real_t eigenval_min = 2;
    for (sgp_vid_t i = 0; i < (g.nvertices); i++) {
        sgp_vid_t weighted_degree = g.source_offsets[i + 1] - g.source_offsets[i];
        sgp_real_t u_i = weighted_degree * u[i];
        sgp_real_t matvec_i = 0;
        for (sgp_eid_t j = g.source_offsets[i];
            j < g.source_offsets[i + 1]; j++) {
            matvec_i += u[g.destination_indices[j]];
        }
        u_i -= matvec_i;
        sgp_real_t eigenval_est = u_i / u[i];
        if (eigenval_est < eigenval_min) {
            eigenval_min = eigenval_est;
        }
        if (eigenval_est > eigenval_max) {
            eigenval_max = eigenval_est;
        }
        eigenval += (u_i * u_i) * 1e9;
    }

    printf("eigenvalue = %1.9lf (%1.9lf %1.9lf), "
        "edge cut lb %5.0lf "
        "gap ratio %.0lf\n",
        eigenval * 1e-9,
        eigenval_min, eigenval_max,
        eigenval * 1e-9 * (g.nvertices) / 4,
        ceil(1.0 / (1.0 - eigenval * 1e-9)));
}

SGPAR_API int sgp_power_iter(sgp_real_t* u, const matrix_type& g, int normLap, int final
#ifdef EXPERIMENT
    , ExperimentLoggerUtil& experiment
#endif
    ) {

    sgp_vid_t n = g.numRows();

    Kokkos::initialize();
    {

        sgp_real_t* vec1 = (sgp_real_t*)malloc(n * sizeof(sgp_real_t));
        SGPAR_ASSERT(vec1 != NULL);
        for (sgp_vid_t i = 0; i < n; i++) {
            vec1[i] = 1.0;
        }

        sgp_wgt_t* weighted_degree = (sgp_wgt_t*)malloc(n * sizeof(sgp_wgt_t));
        assert(weighted_degree != NULL);
        Kokkos::parallel_for(n, KOKKOS_LAMBDA(sgp_vid_t i) {
            sgp_wgt_t degree_wt_i = 0;
            sgp_eid_t end_offset = gc.graph.row_map(i + 1);
            for (sgp_eid_t j = gc.graph.row_map(i); j < end_offset; j++) {
                degree_wt_i += gc.values(j);
            }
            weighted_degree[i] = degree_wt_i;
        });

        sgp_wgt_t gb = 2.0;
        if (!normLap) {
            gb = 2 * weighted_degree[0];
            for (sgp_vid_t i = 1; i < n; i++) {
                if (gb < 2 * g.weighted_degree[i]) {
                    gb = 2 * g.weighted_degree[i];
                }
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

        sgp_real_t* v = (sgp_real_t*)malloc(n * sizeof(sgp_real_t));
        SGPAR_ASSERT(v != NULL);
        for (sgp_vid_t i = 0; i < n; i++) {
            v[i] = u[i];
        }

        sgp_real_t tol = SGPAR_POWERITER_TOL;
        uint64_t niter = 0;
        uint64_t iter_max = (uint64_t)SGPAR_POWERITER_ITER / (uint64_t)n;
        sgp_real_t dotprod = 0, lastDotprod = 1;
        while (fabs(dotprod - lastDotprod) > tol && (niter < iter_max)) {

            //copying u everytime isn't efficient but I'm just tryna make this work for now
            Kokkos::View<sgp_real_t*> u_view("u", n);
            Kokkos::View<sgp_real_t*> v_view("v", n);

            // u = v
            Kokkos::parallel_for(n, KOKKOS_LAMBDA(sgp_vid_t i) {
                u[i] = v[i];
                u_view(i) = v[i];
                v_view(i) = 0.0;
            });

            KokkosSparse::spmv("N", 1.0, g, u_view, 0.0, v_view);

            // v = Lu
            Kokkos::parallel_for(n, KOKKOS_LAMBDA(sgp_vid_t i) {
                // sgp_real_t v_i = g.weighted_degree[i]*u[i];
                sgp_real_t weighted_degree_inv, v_i;
                if (!normLap) {
                    v_i = (gb - weighted_degree[i]) * u[i];
                }
                else {
                    weighted_degree_inv = 1.0 / weighted_degree[i];
                    v_i = 0.5 * u[i];
                }
                // v_i -= matvec_i;
                if (!normLap) {
                    v_i += v_view(i);
                }
                else {
                    v_i += 0.5 * v_view(i) * weighted_degree_inv;
                }
                v[i] = v_i;
            });

            if (!normLap) {
                sgp_vec_orthogonalize_kokkos(v, vec1, n);
            }
            sgp_vec_normalize_kokkos(v, n);
            lastDotprod = dotprod;
            sgp_vec_dotproduct_kokkos(&dotprod, u, v, n);
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

        free(vec1);
        free(v);
        if (normLap && final) {
            free(weighted_degree);
        }

    }
    Kokkos::finalize();

    return EXIT_SUCCESS;
}