#include "coarseners.h"
#include "heuristics.h"
#include "assembler.h"
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

using namespace sgpar;
using namespace sgpar::sgpar_kokkos;
namespace fs = std::filesystem;

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

size_t getFileLength(const char *fname){
    // Obtain the size of the file.
    return fs::file_size(fname);
}

int load_kmers(char_view_t& out, char *fname, edge_offset_t k) {

    Kokkos::Timer t;
    std::ifstream infp(fname);
    if (!infp.is_open()) {
        printf("Error: Could not open input file. Exiting ...\n");
        return EXIT_FAILURE;
    }

    auto sz = getFileLength(fname);

    // Create a buffer for file
    std::string s(sz, '\0');
    // Read the whole file into the buffer.
    infp.read(s.data(), sz);
    infp.close();

    edge_offset_t n;
    const char* f = s.c_str();
    printf("Time to load file: %.3f\n", t.seconds());
    t.reset();
#ifdef HUGE
    sscanf(f, "%lu", &n);
#elif defined(LARGE)
    sscanf(f, "%lu", &n);
#else
    sscanf(f, "%u", &n);
#endif
    char_mirror_t char_mirror("char mirror", n*k);
    char* read_to = char_mirror.data();
    for(long i = 0; i < n; i++){
        //file contains kmer counts, don't care about them
        //seek the endline
        while(*f != '\n') f++;
        //increment past the endline
        f++;
        strncpy(read_to, f, k);
        //increment output buffer for next kmer
        read_to += k;
        //advance past the kmer
        f += k;
    }
    out = char_view_t("chars", n*k);
    Kokkos::deep_copy(out, char_mirror);
    return 0;
}

void write_to_f(char_view_t unitigs, edge_view_t unitig_offsets, std::string fname){
    char_mirror_t chars("chars", unitigs.extent(0));
    Kokkos::deep_copy(chars, unitigs);
    edge_mirror_t offsets("offsets", unitig_offsets.extent(0));
    Kokkos::deep_copy(offsets, unitig_offsets);
    std::ofstream of(fname, std::ofstream::out | std::ofstream::app);
    ordinal_t n = offsets.extent(0) - 1;
    for(ordinal_t i = 0; i < n; i++){
        edge_offset_t string_size = offsets(i + 1) - offsets(i);
        char* buf_start = chars.data() + offsets(i);
        std::string s(buf_start, string_size);
        of << s << std::endl;
    }
    of.close();
}

void write_unitigs(char_view_t kmers, edge_view_t kmer_offsets, graph_type glue_action, std::string fname){
    edge_offset_t write_size = 0;
    c_edge_subview_t start_writes_sub = Kokkos::subview(glue_action.row_map, 0);
    c_edge_subview_t end_writes_sub = Kokkos::subview(glue_action.row_map, 1);
    edge_offset_t start_writes = 0, end_writes = 0;
    Kokkos::deep_copy(start_writes, start_writes_sub);
    Kokkos::deep_copy(end_writes, end_writes_sub);
    edge_view_t write_sizes("write sizes", end_writes - start_writes + 1);
    Kokkos::parallel_scan("count writes", r_policy(start_writes, end_writes), KOKKOS_LAMBDA(const edge_offset_t i, edge_offset_t& update, const bool final){
        ordinal_t u = glue_action.entries(i);
        edge_offset_t size = kmer_offsets(u + 1) - kmer_offsets(u);
        if(final){
            write_sizes(i - start_writes) = update;
            if(i + 1 == end_writes){
                write_sizes(end_writes - start_writes) = update + size;
            }
        }
        update += size;
    });
    edge_subview_t write_size_sub = Kokkos::subview(write_sizes, end_writes - start_writes);
    Kokkos::deep_copy(write_size, write_size_sub);
    char_view_t writes("writes", write_size);
    printf("write out unitigs size: %u\n", write_size);
    printf("write out unitigs count: %u\n", end_writes - start_writes);
    Kokkos::parallel_for("move writes", r_policy(start_writes, end_writes), KOKKOS_LAMBDA(const edge_offset_t i){
        ordinal_t u = glue_action.entries(i);
        edge_offset_t write_offset = write_sizes(i - start_writes);
        for(edge_offset_t j = kmer_offsets(u); j < kmer_offsets(u + 1); j++){
            writes(write_offset) = kmers(j);
            write_offset++;
        }
    });
    write_to_f(writes, write_sizes, fname);
}

void compress_unitigs(char_view_t& kmers, edge_view_t& kmer_offsets, graph_type glue_action, edge_offset_t k){
    edge_offset_t write_size = 0;
    //minus 2 because 0 is not processed, and row_map is one bigger than number of rows
    ordinal_t n = glue_action.row_map.extent(0) - 2;
    edge_view_t next_offsets("next offsets", n + 1);
    Kokkos::parallel_scan("compute offsets", r_policy(1, n+1), KOKKOS_LAMBDA(const ordinal_t u, edge_offset_t& update, const bool final){
        edge_offset_t size = 0;
        bool first = true;
        for(edge_offset_t i = glue_action.row_map(u); i < glue_action.row_map(u + 1); i++){
            ordinal_t f = glue_action.entries(i);
            size += kmer_offsets(f + 1) - kmer_offsets(f);
            if(!first){
                //subtract for overlap of k-mers/unitigs
                size -= (k - 1);
            }
            first = false;
        }
        if(final){
            next_offsets(u - 1) = update;
            if(u == n){
                next_offsets(n) = update + size;
            }
        }
        update += size;
    });
    edge_subview_t write_size_sub = Kokkos::subview(next_offsets, n);
    Kokkos::deep_copy(write_size, write_size_sub);
    char_view_t writes("writes", write_size);
    printf("compressed unitigs size: %u\n", write_size);
    Kokkos::parallel_for("move writes", r_policy(1, n+1), KOKKOS_LAMBDA(const edge_offset_t u){
        bool first = true;
        edge_offset_t write_offset = next_offsets(u - 1);
        for(edge_offset_t i = glue_action.row_map(u); i < glue_action.row_map(u + 1); i++){
            ordinal_t f = glue_action.entries(i);
            edge_offset_t start = kmer_offsets(f);
            edge_offset_t end = kmer_offsets(f + 1);
            if(!first){
                //subtract for overlap of k-mers/unitigs
                start += (k - 1);
            }
            first = false;
            for(edge_offset_t j = start; j < end; j++){
                writes(write_offset) = kmers(j);
                write_offset++;
            }
        }
    });
    kmers = writes;
    kmer_offsets = next_offsets;
}

edge_view_t sizes_init(ordinal_t n, edge_offset_t k){
    edge_view_t sizes("unitig sizes", n + 1);
    Kokkos::parallel_for("init sizes", n + 1, KOKKOS_LAMBDA(const ordinal_t i){
        sizes(i) = k*i;
    });
    return sizes;
}

void compress_unitigs_maximally(char_view_t kmers, std::list<graph_type> glue_actions, edge_offset_t k, std::string fname){
    ordinal_t n = kmers.extent(0) / k;
    //there are issues compiling kernels if there is a std object in the function header
    edge_view_t sizes = sizes_init(n, k);
    auto glue_iter = glue_actions.begin();
    while(glue_iter != glue_actions.end()){
        write_unitigs(kmers, sizes, *glue_iter, fname);
        compress_unitigs(kmers, sizes, *glue_iter, k);
        glue_iter++;
    }
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

int main(int argc, char **argv) {

    if (argc < 5) {
        printf("You input %d args\n", argc);
        fprintf(stderr, "Usage: %s <k-mer file> <(k+1)-mer file> k <output file>\n", argv[0]);
        return EXIT_FAILURE;
    }
    char *kmer_fname = argv[1];
    char *kpmer_fname = argv[2];
    edge_offset_t k = atoi(argv[3]);
    std::string out_fname(argv[4]);
    Kokkos::initialize();
    {
        char_view_t kmers, kpmers;
        Kokkos::Timer t;
        load_kmers(kmers, kmer_fname, k);
        load_kmers(kpmers, kpmer_fname, k+1);
        printf("Read input data in %.3fs\n", t.seconds());
        t.reset();
        Kokkos::Timer t2;
        printf("kmer size: %lu, kmers: %lu\n", kmers.extent(0), kmers.extent(0)/k);
        printf("(k+1)-mer size: %lu, (k+1)mers: %lu\n", kpmers.extent(0), kpmers.extent(0)/(k+1));
        vtx_view_t vtx_map = generate_hashmap(kmers, k, kmers.extent(0)/k);
        //vtx_view_t edge_map = generate_hashmap(kpmers, k + 1, kpmers.extent(0)/(k + 1));
        printf("kmer hashmap size: %lu\n", vtx_map.extent(0));
        //printf("(k+1)-mer hashmap size: %lu\n", edge_map.extent(0));
        std::list<graph_type> glue_list;
        char_mirror_t kmer_copy;
        ExperimentLoggerUtil experiment;
        {
            vtx_view_t g;
            using coarsener_t = coarse_builder<ordinal_t, edge_offset_t, value_t, Device>;
            coarsener_t coarsener;
            {
                graph_type g_base = assemble_graph(kmers, kpmers, vtx_map, k);
                printf("entries: %lu\n", g_base.entries.extent(0));
                kmer_copy = move_to_main(kmers);
                //this is likely the peak memory usage point of the program
                //don't need these anymore, delete them
                //Kokkos::resize(edge_map, 0);
                Kokkos::resize(vtx_map, 0);
                Kokkos::resize(kpmers, 0);
                //will need this later but we made a copy
                Kokkos::resize(kmers, 0);
                g = coarsener.prune_edges(g_base);
            }
            printf("Time to assemble pruned graph: %.3fs\n", t.seconds());
            t.reset();
            glue_list = coarsener.coarsen_de_bruijn_full_cycle(g, experiment);
        }
        printf("glue list length: %lu\n", glue_list.size());
        printf("Time to generate glue list: %.3fs\n", t.seconds());
        printf("Aggregation time: %.3fs\n", experiment.getMeasurement(ExperimentLoggerUtil::Measurement::Map));
        printf("Coarse graph build time: %.3fs\n", experiment.getMeasurement(ExperimentLoggerUtil::Measurement::Build));
        printf("Interpolation graph transpose time: %.3fs\n", experiment.getMeasurement(ExperimentLoggerUtil::Measurement::InterpTranspose));
        t.reset();
        kmers = move_to_device(kmer_copy);
        t.reset();
        compress_unitigs_maximally(kmers, glue_list, k, out_fname);
        printf("Time to compact unitigs: %.3fs\n", t.seconds());
        t.reset();
        printf("Total time: %.3fs\n", t2.seconds());
        t2.reset();
    }
    Kokkos::finalize();
    return 0;
}
