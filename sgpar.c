#include <stdio.h>
#include <stdlib.h>

#define SGPAR_IMPLEMENTATION
#include "sgpar.h"

#ifdef __cplusplus
using namespace sgpar;
#endif

int main(int argc, char **argv) {

    if (argc != 3) {
        fprintf(stderr, "Usage: %s filename.csr metricsfilename.txt\n", argv[0]);
        return EXIT_FAILURE;
    }
    char *filename = argv[1];
    char *metrics = argv[2];

    sgp_graph_t g;
    CHECK_SGPAR( sgp_load_graph(&g, filename) );
    printf("n: %ld, m: %ld\n", g.nvertices, g.nedges);

    sgp_vid_t *part;
    part = (sgp_vid_t *) malloc(g.nvertices * sizeof(sgp_vid_t));
    SGPAR_ASSERT(part != NULL);

    CHECK_SGPAR( sgp_partition_graph(part, 2, 0, 0, 0, 0, g, metrics) );
    CHECK_SGPAR( sgp_free_graph(&g) );

    free(part);

    return EXIT_SUCCESS;
}

