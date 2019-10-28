#include <stdio.h>
#include <stdlib.h>

#define SGPAR_IMPLEMENTATION
#include "sgpar.h"

#ifdef __cplusplus
using namespace sgpar;
#endif

int main(int argc, char **argv) {

    if (argc < 3) {
        fprintf(stderr, "Usage: %s filename.csr metricsfilename.txt\n", argv[0]);
        return EXIT_FAILURE;
    }
    char *filename = argv[1];
    char *metrics = argv[2];
    
    int coarsening_alg = 0;
    if (argc >= 4) {
        coarsening_alg = atoi(argv[3]);
    }

    int refine_alg = 0;
    if (argc >= 5) {
        refine_alg = atoi(argv[4]);
    }

    int local_search_alg = 0;
    if (argc >= 6) {
        local_search_alg = atoi(argv[5]);
    }

    int num_iter = 100;
    if (argc >= 7) {
        num_iter = atoi(argv[6]);
    }

    sgp_graph_t g;
    CHECK_SGPAR( sgp_load_graph(&g, filename) );
    printf("n: %ld, m: %ld\n", g.nvertices, g.nedges);
    printf("coarsening_alg: %d, refine_alg: %d, local_alg %d, num_iter %d\n", 
                    coarsening_alg, refine_alg, local_search_alg, num_iter);

    sgp_vid_t *part;
    part = (sgp_vid_t *) malloc(g.nvertices * sizeof(sgp_vid_t));
    SGPAR_ASSERT(part != NULL);

    long edgecut_min = 1<<30;
    
    for (int i=0; i<num_iter; i++) {
        long edgecut = 0;
        CHECK_SGPAR( sgp_partition_graph(part, 2, &edgecut, coarsening_alg, 
                                        refine_alg, local_search_alg, 0, g, metrics) );
        if (edgecut < edgecut_min) {
            edgecut_min = edgecut;
        }
    }
    printf("graph %s, min edgecut found is %ld\n", 
                    filename, edgecut_min);

    /*
    FILE *outfp = fopen("parts.txt", "w");
    for (sgp_vid_t i=0;  i<g.nvertices; i++) {
        fprintf(outfp, "%d\n", part[i]); 
    }
    fclose(outfp);
    */

    CHECK_SGPAR( sgp_free_graph(&g) );
    free(part);

    return EXIT_SUCCESS;
}

