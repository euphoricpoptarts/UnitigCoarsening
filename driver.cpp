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

char_view_t read_kmers(char *fname){

    FILE* infp = fopen(fname, "rb");
    if (infp == NULL) {
        std::cout << "Error: could not open output file " << fname << "! Exiting" << std::endl;
        exit(1);
    }

    ordinal_t n;
    edge_offset_t k;

    assert(fread(&n, sizeof(ordinal_t), 1, infp) != 0);
    assert(fread(&k, sizeof(edge_offset_t), 1, infp) != 0);
    char_view_t kmers("kmers", n*k);
    char_mirror_t kmers_m = Kokkos::create_mirror_view(kmers);
    size_t nitems_read = fread(kmers_m.data(), sizeof(char), n*k , infp);
    assert(nitems_read == ((size_t) n*k));
    fclose(infp);
    
    Kokkos::deep_copy(kmers, kmers_m);

    return kmers;
}

int save_kmers(char_view_t kmers, edge_offset_t k, char *fname){

    FILE* writeBinaryPtr = fopen(fname, "wb");
    if (writeBinaryPtr == NULL) {
        std::cout << "Error: could not open output file " << fname << "! Exiting" << std::endl;
        exit(1);
    }

    ordinal_t n = kmers.extent(0) / k;
    char_mirror_t kmers_m("kmers mirror", kmers.extent(0));
    Kokkos::deep_copy(kmers_m, kmers);

    fwrite(&n, sizeof(ordinal_t), 1, writeBinaryPtr);
    fwrite(&k, sizeof(edge_offset_t), 1, writeBinaryPtr);
    fwrite(kmers_m.data(), sizeof(char), kmers.extent(0), writeBinaryPtr);
    fclose(writeBinaryPtr);
    return 0;
}

vtx_view_t read_pruned_graph(char *fname){

    FILE* infp = fopen(fname, "rb");
    if (infp == NULL) {
        std::cout << "Error: could not open output file " << fname << "! Exiting" << std::endl;
        exit(1);
    }

    ordinal_t n;

    assert(fread(&n, sizeof(ordinal_t), 1, infp) != 0);
    vtx_view_t g("g", n);
    vtx_mirror_t g_m = Kokkos::create_mirror_view(g);
    size_t nitems_read = fread(g_m.data(), sizeof(ordinal_t), n , infp);
    assert(nitems_read == ((size_t) n));
    fclose(infp);
    
    Kokkos::deep_copy(g, g_m);

    return g;
}

int save_pruned_graph(vtx_view_t g, char *fname){

    FILE* writeBinaryPtr = fopen(fname, "wb");
    if (writeBinaryPtr == NULL) {
        std::cout << "Error: could not open output file " << fname << "! Exiting" << std::endl;
        exit(1);
    }

    ordinal_t n = g.extent(0);
    vtx_mirror_t g_m("g mirror", n);
    Kokkos::deep_copy(g_m, g);

    fwrite(&n, sizeof(ordinal_t), 1, writeBinaryPtr);
    fwrite(g_m.data(), sizeof(ordinal_t), n, writeBinaryPtr);
    fclose(writeBinaryPtr);
    return 0;
}

int load_kmers(char_view_t& out, char *fname, edge_offset_t k) {

    std::ifstream infp(fname);
    if (!infp.is_open()) {
        printf("Error: Could not open input file. Exiting ...\n");
        return EXIT_FAILURE;
    }
    edge_offset_t n;
    infp >> n;
    //+1 for final null terminator
    char_mirror_t char_mirror("char mirror", n*k + 1);
    for(long i = 0; i < n; i++){
        char* read_to = char_mirror.data() + i*k;
        int count = 0;
        //don't need to worry about reading null terminator into char_mirror
        //because it will be overwritten for all but the last string
        //we also allocated an extra char for the last string's null terminator
        infp >> read_to >> count;
    }
    infp.close();
    out = char_view_t("chars", n*k);
    //exclude the last character, which is the null terminator
    char_mirror_t mirror_sub = Kokkos::subview(char_mirror, std::make_pair((edge_offset_t)0, n*k));
    Kokkos::deep_copy(out, mirror_sub);
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

#ifdef COMPACT
int main(int argc, char **argv) {

    if (argc < 5) {
        printf("You input %d args\n", argc);
        fprintf(stderr, "Usage: %s <k-mer binary> <g binary> k <output file>\n", argv[0]);
        return EXIT_FAILURE;
    }
    char *kmer_fname = argv[1];
    char *g_fname = argv[2];
    edge_offset_t k = atoi(argv[3]);
    std::string out_fname(argv[4]);
    Kokkos::initialize();
    {
        Kokkos::Timer t;
        Kokkos::Timer t2;
        ExperimentLoggerUtil experiment;
        std::list<graph_type> glue_list;
        {
            vtx_view_t g = read_pruned_graph(g_fname);
            using coarsener_t = coarse_builder<ordinal_t, edge_offset_t, value_t, Device>;
            coarsener_t coarsener;
            glue_list = coarsener.coarsen_de_bruijn_full_cycle(g, experiment);
        }
        printf("glue list length: %lu\n", glue_list.size());
        printf("Time to generate glue list: %.3fs\n", t.seconds());
        printf("Aggregation time: %.3fs\n", experiment.getMeasurement(ExperimentLoggerUtil::Measurement::Map));
        printf("Coarse graph build time: %.3fs\n", experiment.getMeasurement(ExperimentLoggerUtil::Measurement::Build));
        printf("Interpolation graph transpose time: %.3fs\n", experiment.getMeasurement(ExperimentLoggerUtil::Measurement::InterpTranspose));
        t.reset();
        char_view_t kmers = read_kmers(kmer_fname);
        printf("Time to transfer kmers back to device: %.3fs\n", t.seconds());
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
#elif defined(PRUNE)
int main(int argc, char **argv) {

    if (argc < 6) {
        printf("You input %d args\n", argc);
        fprintf(stderr, "Usage: %s <k-mer file> <(k+1)-mer file> k <output kmer file> <output graph file>\n", argv[0]);
        return EXIT_FAILURE;
    }
    char *kmer_fname = argv[1];
    char *kpmer_fname = argv[2];
    edge_offset_t k = atoi(argv[3]);
    char *out_kname(argv[4]);
    char *out_gname(argv[5]);
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
                //this is likely the peak memory usage point of the program
                //don't need these anymore, delete them
                //Kokkos::resize(edge_map, 0);
                Kokkos::resize(vtx_map, 0);
                Kokkos::resize(kpmers, 0);
                //will need this later but we made a copy
                g = coarsener.prune_edges(g_base);
            }
            printf("Time to assemble pruned graph: %.3fs\n", t.seconds());
            t.reset();
            save_pruned_graph(g, out_gname);
            save_kmers(kmers, k, out_kname);
        }
    }
    Kokkos::finalize();
    return 0;
}
#endif
