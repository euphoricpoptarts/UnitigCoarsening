#include "coarseners.h"
#include "heuristics.h"
#include "ExperimentLoggerUtil.cpp"
#include "definitions_kokkos.h"
#include <assert.h>
#include <iostream>
#include <fstream>
#include <functional>

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

int load_kmers(char_view_t& out, char *fname, int k) {

    std::ifstream infp(fname);
    if (!infp.is_open()) {
        printf("Error: Could not open input file. Exiting ...\n");
        return EXIT_FAILURE;
    }
    long n;
    infp >> n;
    char_mirror_t char_mirror("char mirror", n*k);
    for(long i = 0; i < n; i++){
        char* read_to = char_mirror.data() + i*k;
        int count = 0;
        //may be a problem reading the null-terminator
        infp >> read_to >> count;
    }
    infp.close();
    out = char_view_t("chars", n*k);
    Kokkos::deep_copy(out, char_mirror);
    return 0;
}
  
//fnv hash algorithm
//we going deep
KOKKOS_INLINE_FUNCTION
uint32_t fnv(const char_view_t chars, size_t offset, size_t k){
    uint32_t hash = 2166136261U;
    for(uint32_t i = offset; i < offset + k; i++)
    {
        hash = hash ^ (chars(i)); // xor next byte into the bottom of the hash
        hash = hash * 16777619; // Multiply by prime number found to work well
    }
    return hash;
}

//gotta do everything myself
KOKKOS_INLINE_FUNCTION
bool cmp(const char_view_t s1_chars, const char_view_t s2_chars, const size_t s1_offset, const size_t s2_offset, const size_t k){
    for(size_t i = 0; i < k; i++){
        if(s1_chars(s1_offset + i) != s2_chars(s2_offset + i)){
            return false;
        }
    }
    return true;
}

//check if edge formed by appending extension to kmer at offset exists in edges using a hashmap
KOKKOS_INLINE_FUNCTION
bool find_edge(const char_view_t chars, const char_view_t edges, const vtx_view_t edge_map, size_t offset, size_t k, char extension){
    uint32_t hash = fnv(chars, offset, k);
    hash = hash ^ (extension); // xor next byte into the bottom of the hash
    hash = hash * 16777619; // Multiply by prime number found to work well
    uint32_t hash_cast = edge_map.extent(0) - 1;
    hash = hash & hash_cast;
    while(edge_map(hash) != ORD_MAX){
        size_t edge_offset = edge_map(hash)*(k + 1);
        if(cmp(chars, edges, offset, edge_offset, k) && extension == edges(edge_offset + k)){
            return true;
        }
        hash = (hash + 1) & hash_cast;
    }
    return false;
}

//check if vtx formed by appending extension to (k-1)mer at offset exists in vtxs using a hashmap
KOKKOS_INLINE_FUNCTION
ordinal_t find_vtx(const char_view_t chars, const vtx_view_t vtx_map, size_t offset, size_t k, char extension){
    uint32_t hash = fnv(chars, offset, k - 1);
    hash = hash ^ (extension); // xor next byte into the bottom of the hash
    hash = hash * 16777619; // Multiply by prime number found to work well
    uint32_t hash_cast = vtx_map.extent(0) - 1;
    hash = hash & hash_cast;
    while(vtx_map(hash) != ORD_MAX){
        size_t opposite_offset = vtx_map(hash)*k;
        if(cmp(chars, chars, offset, opposite_offset, k - 1) && extension == char(opposite_offset + (k - 1))){
            return vtx_map(hash);
        }
        hash = (hash + 1) & hash_cast;
    }
    return ORD_MAX;
}

vtx_view_t generate_hashmap(char_view_t kmers, int k, int size){
    size_t hashmap_size = 1;
    while(hashmap_size < 2*size) hashmap_size <<= 1;
    vtx_view_t out("hashmap", hashmap_size);
    Kokkos::parallel_for("init hashmap", hashmap_size, KOKKOS_LAMBDA(const ordinal_t i){
        out(i) = ORD_MAX;
    });
    size_t hash_cast = hashmap_size - 1;
    Kokkos::parallel_for("fill hashmap", size, KOKKOS_LAMBDA(const ordinal_t i){
        size_t hash = fnv(kmers, k*i, k) & hash_cast;
        bool success = Kokkos::atomic_compare_exchange_strong(&out(hash), ORD_MAX, i);
        //linear probing
        //all values are unique so no need to check
        while(!success){
            hash = (hash + 1) & hash_cast;
            success = Kokkos::atomic_compare_exchange_strong(&out(hash), ORD_MAX, i);
        }
    });
   return out; 
}

struct prefix_sum1
{
    vtx_view_t input;
    edge_view_t output;

    prefix_sum1(vtx_view_t input,
        edge_view_t output)
        : input(input)
        , output(output) {}

    KOKKOS_INLINE_FUNCTION
        void operator() (const ordinal_t i, edge_offset_t& update, const bool final) const {
        const edge_offset_t val_i = input(i);
        update += val_i;
        if (final) {
            output(i + 1) = update;
        }
    }
};

graph_type assemble_graph(char_view_t kmers, char_view_t kpmers, vtx_view_t edge_map, vtx_view_t vtx_map, int k){
    ordinal_t n = kmers.extent(0) / k;
    edge_view_t row_map("row map", n+1);
    edge_view_t edge_count("edge count", n);
    char edges[4] = {'A', 'T', 'G', 'C'};
    Kokkos::parallel_for("count edges", n, KOKKOS_LAMBDA(const ordinal_t i){
        for(int j = 0; j < 4; j++){
            //check if edge exists
            if(find_edge(kmers, kpmers, edge_map, i*k, k, edges[j])){
                edge_count(i)++;
            }
        }
    });
    Kokkos::parallel_scan("calc source offsets", n, prefix_sum1(edge_count, row_map));
    edge_subview_t rm_subview = Kokkos::subview(row_map, n);
    edge_offset_t total_e = 0;
    Kokkos::deep_copy(total_e, rm_subview);
    Kokkos::parallel_for("reset edge count", n, KOKKOS_LAMBDA(const ordinal_t i){
        edge_count(i) = 0;
    });
    vtx_view_t entries("pruned out entries", total_e);
    Kokkos::parallel_for("write transpose edges", n, KOKKOS_LAMBDA(const ordinal_t i){
        for(int j = 0; j < 4; j++){
            //check if edge exists
            if(find_edge(kmers, kpmers, edge_map, i*k, k, edges[j])){
                edge_offset_t insert = row_map(i) + edge_count(i)++;
                ordinal_t v = find_vtx(kmers, vtx_map, i*k + 1, k, edges[j]);
                entries(insert) = v;
            }
        }
    });
    graph_type g(entries, row_map);
    return g;
}

int main(int argc, char **argv) {

    if (argc < 4) {
        printf("You input %d args\n", argc);
        fprintf(stderr, "Usage: %s <k-mer file> <(k+1)-mer file> k\n", argv[0]);
        return EXIT_FAILURE;
    }
    char *kmer_fname = argv[1];
    char *kpmer_fname = argv[2];
    int k = atoi(argv[3]);
    Kokkos::initialize();
    {
        char_view_t kmers, kpmers;
        load_kmers(kmers, kmer_fname, k);
        load_kmers(kpmers, kpmer_fname, k+1);
        printf("kmer size: %lu\n", kmers.extent(0));
        printf("(k+1)-mer size: %lu\n", kpmers.extent(0));
        vtx_view_t vtx_map = generate_hashmap(kmers, k, kmers.extent(0)/k);
        vtx_view_t edge_map = generate_hashmap(kpmers, k + 1, kpmers.extent(0)/(k + 1));
        printf("kmer hashmap size: %lu\n", vtx_map.extent(0));
        printf("(k+1)-mer hashmap size: %lu\n", edge_map.extent(0));
        graph_type g = assemble_graph(kmers, kpmers, edge_map, vtx_map, k);
        printf("entries: %lu\n", g.entries.extent(0));
        //graph_type g;
        //load_graph(g, filename);
        //printf("vertices: %u; nnz: %lu\n", g.numRows(), g.entries.extent(0));
        //using coarsener_t = coarse_builder<ordinal_t, edge_offset_t, value_t, Device>;
        //coarsener_t coarsener;
        //ExperimentLoggerUtil experiment;
        //coarsener.coarsen_de_bruijn_full_cycle(g, experiment);
    }
    Kokkos::finalize();
    return 0;
}
