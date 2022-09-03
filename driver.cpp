#include "coarseners.h"
#include "heuristics.h"
#include "assembler.h"
#include "compact.h"
#include "ExperimentLoggerUtil.cpp"
#include "definitions_kokkos.h"
#include <assert.h>
#include <iostream>
#include <fstream>
#include <functional>
#include <utility>
#include <filesystem>

#define CHECK_RETSTAT(func)                                                    \
{                                                                              \
    int status = (func);                                                       \
    if (status != 0) {                                                         \
        printf("Error: return value %d at line %d. Exiting ...\n",             \
               status, __LINE__);                                              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

using namespace unitig_compact;
namespace fs = std::filesystem;

int load_graph(graph_t& g, char *csr_filename) {

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
    g = graph_t(entries, row_map);
    return 0;
}

size_t getFileLength(const char *fname){
    // Obtain the size of the file.
    return fs::file_size(fname);
}

//input file looks like
//4
//AAAA
//GGGG
//CCCC
//TTTA
int load_kmers(char_view_t& out, char *fname, edge_offset_t k) {

    Kokkos::Timer t;
    std::ifstream infp(fname);
    if (!infp.is_open()) {
        printf("Error: Could not open input file. Exiting ...\n");
        return EXIT_FAILURE;
    }

    size_t sz = getFileLength(fname);

    size_t chunk_size = sz / 64;
    size_t offset = 0;
    // Create a buffer for file
    char* s = new char[chunk_size];
    edge_offset_t n = 0, total_read = 0;
    char_mirror_t char_mirror;
    char* read_to = 0;
    printf("Time to init buffer: %.3f\n", t.seconds());
    t.reset();
    while(offset < sz){
        // Read a chunk of the file into the buffer.
        infp.seekg(offset);
        if(offset + chunk_size > sz){
            chunk_size = sz - offset;
        }
        infp.read(s, chunk_size);

        const char* f = s;
        if(offset == 0){
#ifdef HUGE
            sscanf(f, "%lu", &n);
#elif defined(LARGE)
            sscanf(f, "%lu", &n);
#else
            sscanf(f, "%u", &n);
#endif
            out = char_view_t(Kokkos::ViewAllocateWithoutInitializing("chars"), n*k);
            char_mirror = Kokkos::create_mirror_view(out);
            read_to = char_mirror.data();
        }
        size_t last_read = 0;
        while(f - s < chunk_size){
            //file contains kmer counts, don't care about them
            //seek the endline
            while(f - s < chunk_size && *f != '\n') f++;
            //increment past the endline
            f++;
            if(f + k - s > chunk_size){
                break;
            } else {
                last_read = f + k - s;
            }
            strncpy(read_to, f, k);
            total_read++;
            //increment output buffer for next kmer
            read_to += k;
            //advance past the kmer
            f += k;
        }
        offset += last_read;
        if(total_read == n){
            break;
        }
    }
    printf("Time to read and process input: %.3f\n", t.seconds());
    t.reset();
    delete[] s;
    infp.close();
    printf("Time to init chars device memory: %.3f\n", t.seconds());
    t.reset();
    Kokkos::deep_copy(out, char_mirror);
    printf("Time to copy chars to device memory: %.3f\n", t.seconds());
    t.reset();
    return 0;
}

char_mirror_t move_to_main(char_view_t x){
    char_mirror_t y("mirror", x.extent(0));
    Kokkos::deep_copy(y, x);
    return y;
}

char_view_t move_to_device(char_mirror_t x){
    char_view_t y("device", x.extent(0));
    Kokkos::deep_copy(y, x);
    return y;
}

comp_vt compress_kmers(edge_offset_t k, char_view_t kmers){
    edge_offset_t k_pad = ((k + 15) / 16) * 16;
    edge_offset_t comp_size = k_pad / 16;
    edge_offset_t n = kmers.extent(0) / k;
    comp_vt kmer_compress(Kokkos::ViewAllocateWithoutInitializing("kmers compressed nonpart"), n * comp_size);
    vtx_mirror_t char_map_mirror("char map mirror", 256);
    char_map_mirror('A') = 0;
    char_map_mirror('C') = 1;
    char_map_mirror('G') = 2;
    char_map_mirror('T') = 3;
    vtx_view_t char_map("char map", 256);
    Kokkos::deep_copy(char_map, char_map_mirror);
    Kokkos::parallel_for("compress kmers", n, KOKKOS_LAMBDA(const edge_offset_t j){
        uint32_t byte = 0;
        for(edge_offset_t x = 0; x < k_pad; x++){
            if((x & 15) == 0) byte = 0;
            if(x < k){
                uint32_t write = char_map(kmers(j*k + x));
                write <<= 2*(x & 15);
                byte = byte | write;
            }
            if((x & 15) == 15) kmer_compress(j*comp_size + (x / 16)) = byte;
        }
    });
    return kmer_compress;
}

int main(int argc, char **argv) {

    if (argc != 4) {
        printf("You input %d args\n", argc);
        fprintf(stderr, "Usage: %s <k-mer file> k <output file>\n", argv[0]);
        return EXIT_FAILURE;
    }
    char *kmer_fname = argv[1];
    edge_offset_t k = atoi(argv[2]);
    std::string out_fname(argv[3]);
    Kokkos::initialize();
    {
        char_view_t kmers, kpmers;
        Kokkos::Timer t, t2, t3;
        load_kmers(kmers, kmer_fname, k);
        printf("Read input data in %.3fs\n", t.seconds());
        t.reset();
        t2.reset();
        ordinal_t n_kmers = kmers.extent(0) / k;
        printf("kmer size: %lu, kmers: %lu\n", kmers.extent(0), n_kmers);
        char_view_t rcomps = generate_rcomps(kmers, k, kmers.extent(0)/k);
        comp_vt kmer_comp = compress_kmers(k, kmers);
        comp_vt rcomps_comp = compress_kmers(k, rcomps);
        Kokkos::resize(kmers, 0);
        Kokkos::resize(rcomps, 0);
        printf("Time to create rcomps and compact data: %.3f\n", t.seconds());
        t.reset();
        t3.reset();
        edge_offset_t k_pad = ((k + 15) / 16) * 16;
        edge_offset_t comp_size = k_pad / 16;
        vtx_view_t vtx_map = generate_hashmap(kmer_comp, rcomps_comp, comp_size, n_kmers);
        printf("kmer hashmap size: %lu\n", vtx_map.extent(0));
        printf("Time to generate hashmap: %.3f\n", t3.seconds());
        t3.reset();
        std::list<graph_t> glue_list;
        ExperimentLoggerUtil experiment;
        {
            canon_graph g = assemble_pruned_graph(kmer_comp, rcomps_comp, vtx_map, comp_size);
            coarsener_t coarsener;
            Kokkos::resize(vtx_map, 0);
            printf("Time to assemble pruned graph: %.3fs\n", t.seconds());
            t.reset();
            glue_list = coarsener.coarsen_de_bruijn_full_cycle(g, experiment);
        }
        printf("glue list length: %lu\n", glue_list.size());
        printf("Time to generate glue list: %.3fs\n", t.seconds());
        printf("Aggregation time: %.3fs\n", experiment.getMeasurement(ExperimentLoggerUtil::Measurement::Map));
        printf("Heavy edge time: %.3fs\n", experiment.getMeasurement(ExperimentLoggerUtil::Measurement::Heavy));
        printf("Map construction time: %.3fs\n", experiment.getMeasurement(ExperimentLoggerUtil::Measurement::MapConstruct));
        printf("Map CAS ops time: %.3fs\n", experiment.getMeasurement(ExperimentLoggerUtil::Measurement::CoarsenCAS));
        printf("Map label time: %.3fs\n", experiment.getMeasurement(ExperimentLoggerUtil::Measurement::CoarsenLabel));
        printf("Map repeat gather time: %.3fs\n", experiment.getMeasurement(ExperimentLoggerUtil::Measurement::CoarsenRepeat));
        printf("Coarse graph build time: %.3fs\n", experiment.getMeasurement(ExperimentLoggerUtil::Measurement::Build));
        printf("Interpolation graph transpose time: %.3fs\n", experiment.getMeasurement(ExperimentLoggerUtil::Measurement::InterpTranspose));
        printf("Glue compact time: %.3fs\n", experiment.getMeasurement(ExperimentLoggerUtil::Measurement::CompactGlues));
        t.reset();
        t.reset();
        compress_unitigs_maximally2(kmer_comp, rcomps_comp, glue_list, k, out_fname);
        printf("Time to compact unitigs: %.3fs\n", t.seconds());
        t.reset();
        printf("Total time: %.3fs\n", t2.seconds());
        t2.reset();
    }
    Kokkos::finalize();
    return 0;
}
