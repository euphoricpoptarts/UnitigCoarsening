#include "coarseners.h"
#include "heuristics.h"
#include "ExperimentLoggerUtil.cpp"
#include "definitions_kokkos.h"
#include <assert.h>

#define CHECK_RETSTAT(func)                                                    \
{                                                                              \
    int status = (func);                                                       \
    if (status != 0) {                                                         \
        printf("Error: return value %d at line %d. Exiting ...\n",             \
               status, __LINE__);                                              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

using namespace sgpar;
using namespace sgpar::sgpar_kokkos;

int load_graph(graph_type& g, char *csr_filename) {

    FILE *infp = fopen(csr_filename, "rb");
    if (infp == NULL) {
        printf("Error: Could not open input file. Exiting ...\n");
        return EXIT_FAILURE;
    }
    long n, m;
    long unused_vals[4];
    assert(fread(&n, sizeof(long), 1, infp) != 0);
    assert(fread(&m, sizeof(long), 1, infp) != 0);
    assert(fread(unused_vals, sizeof(long), 4, infp) != 0);
    edge_mirror_t row_map_mirror("row map mirror", n + 1);
    vtx_mirror_t entries_mirror("entries mirror", m);
    size_t nitems_read = fread(row_map_mirror.data(), sizeof(edge_offset_t), n + 1, infp);
    assert(nitems_read == ((size_t) n + 1));
    nitems_read = fread(entries_mirror.data(), sizeof(ordinal_t), m, infp);
    assert(nitems_read == ((size_t) m));
    CHECK_RETSTAT( fclose(infp) );
    edge_view_t row_map("row map", n + 1);
    vtx_view_t entries("entries", m);
    Kokkos::deep_copy(row_map, row_map_mirror);
    Kokkos::deep_copy(entries, entries_mirror);
    g = graph_type(entries, row_map);
    return 0;
}


int main(int argc, char **argv) {

    if (argc < 2) {
        printf("You input %d args\n", argc);
        fprintf(stderr, "Usage: %s csr_filename \n", argv[0]);
        return EXIT_FAILURE;
    }
    char *filename = argv[1];
    Kokkos::initialize();
    {
        graph_type g;
        load_graph(g, filename);
        printf("vertices: %u; nnz: %lu\n", g.numRows(), g.entries.extent(0));
        using coarsener_t = coarse_builder<ordinal_t, edge_offset_t, value_t, Device>;
        coarsener_t coarsener;
        ExperimentLoggerUtil experiment;
        coarsener.coarsen_de_bruijn_full_cycle(g, experiment);
    }
    Kokkos::finalize();
    return 0;
}
