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
#include "lmin_buckets.h"

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
    t.reset();
    char_mirror_t char_mirror("char mirror", n*k);
    printf("Time to init chars host memory: %.3f\n", t.seconds());
    t.reset();
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
    printf("Time to write chars contiguously: %.3f\n", t.seconds());
    t.reset();
    out = char_view_t("chars", n*k);
    printf("Time to init chars device memory: %.3f\n", t.seconds());
    t.reset();
    Kokkos::deep_copy(out, char_mirror);
    printf("Time to copy chars to device memory: %.3f\n", t.seconds());
    t.reset();
    return 0;
}

vtx_view_t generate_permutation(ordinal_t n, pool_t rand_pool) {
    rand_view_t randoms("randoms", n);

    Kokkos::Timer t;
    Kokkos::parallel_for("create random entries", r_policy(0, n), KOKKOS_LAMBDA(ordinal_t i){
        gen_t generator = rand_pool.get_state();
        randoms(i) = generator.urand64();
        rand_pool.free_state(generator);
    });
    t.reset();

    ordinal_t t_buckets = 2*n;
    vtx_view_t buckets("buckets", t_buckets);
    Kokkos::parallel_for("init buckets", r_policy(0, t_buckets), KOKKOS_LAMBDA(ordinal_t i){
        buckets(i) = ORD_MAX;
    });

    uint64_t max = std::numeric_limits<uint64_t>::max();
    uint64_t bucket_size = max / t_buckets;
    Kokkos::parallel_for("insert buckets", r_policy(0, n), KOKKOS_LAMBDA(ordinal_t i){
        ordinal_t bucket = randoms(i) / bucket_size;
        //jesus take the wheel
        for(;; bucket++){
            if(bucket >= t_buckets) bucket -= t_buckets;
            if(buckets(bucket) == ORD_MAX){
                //attempt to insert into bucket
                if(Kokkos::atomic_compare_exchange_strong(&buckets(bucket), ORD_MAX, i)){
                    break;
                }
            }
        }
    });
    
    vtx_view_t permute("permutation", n);
    Kokkos::parallel_scan("extract permutation", r_policy(0, t_buckets), KOKKOS_LAMBDA(const ordinal_t i, ordinal_t& update, const bool final){
        if(buckets(i) != ORD_MAX){
            if(final){
                permute(update) = buckets(i);
            }
            update++;
        }
    });

    printf("sort time: %.4f\n", t.seconds());
    t.reset();
    return permute;
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
        fprintf(stderr, "Usage: %s <k-mer file> <(k+1)-mer file> k l <output file>\n", argv[0]);
        return EXIT_FAILURE;
    }
    char *kmer_fname = argv[1];
    char *kpmer_fname = argv[2];
    edge_offset_t k = atoi(argv[3]);
    edge_offset_t l = atoi(argv[4]);
    std::string out_fname(argv[5]);
    Kokkos::initialize();
    {
        char_view_t kmers, kpmers;
        Kokkos::Timer t;//, t2, t3;
        load_kmers(kmers, kmer_fname, k);
        load_kmers(kpmers, kpmer_fname, k+1);
        printf("Read input data in %.3fs\n", t.seconds());
        t.reset();
        pool_t rand_pool(std::time(nullptr));
        ordinal_t lmin_buckets = 1;
        lmin_buckets <<= 2*l;
        vtx_view_t lmin_bucket_map = generate_permutation(lmin_buckets, rand_pool);
        bucket_kmers kmer_b = find_l_minimizer(kmers, k, l, lmin_bucket_map);
        Kokkos::resize(kmers, 0);
        bucket_kmers kpmer_b = find_l_minimizer_edge(kpmers, k + 1, l, lmin_bucket_map);
        Kokkos::resize(kpmers, 0);
        printf("Computed l-minimizers in %.3f\n", t.seconds());
        t.reset();
        //t2.reset();
        //t3.reset();
        printf("kmer size: %lu, kmers: %lu\n", kmer_b.kmers.extent(0), kmer_b.size);
        printf("(k+1)-mer size: %lu, (k+1)mers: %lu\n", kpmer_b.kmers.extent(0), kpmer_b.size);
        vtx_view_t g("graph", kmer_b.size);
        Kokkos::parallel_for("init g", kmer_b.size, KOKKOS_LAMBDA(const ordinal_t i){
            g(i) = ORD_MAX;
        });
        ordinal_t largest_n = 0, largest_np = 0;;
        for(int i = 0; i < kmer_b.buckets; i++){
            ordinal_t kmer_count = kmer_b.buckets_row_map[i+1] - kmer_b.buckets_row_map[i];
            ordinal_t kpmer_count = kpmer_b.buckets_row_map[i+1] - kpmer_b.buckets_row_map[i];
            if(kmer_count > largest_n){
                largest_n = kmer_count;
            }
            if(kpmer_count > largest_np){
                largest_np = kpmer_count;
            }
        }
        vtx_view_t hashmap = init_hashmap(largest_n);
        assembler_data assembler = init_assembler(largest_n, largest_np);
        for(int i = 0; i < kmer_b.buckets; i++){
            Kokkos::Timer t2;
            ordinal_t kmer_count = kmer_b.buckets_row_map[i+1] - kmer_b.buckets_row_map[i];
            ordinal_t kpmer_count = kpmer_b.buckets_row_map[i+1] - kpmer_b.buckets_row_map[i];
            char_view_t kmer_s = Kokkos::subview(kmer_b.kmers, std::make_pair(kmer_b.buckets_row_map[i]*k, kmer_b.buckets_row_map[i+1]*k));
            char_view_t kpmer_s = Kokkos::subview(kpmer_b.kmers, std::make_pair(kpmer_b.buckets_row_map[i]*(k+1), kpmer_b.buckets_row_map[i+1]*(k+1)));
            generate_hashmap(hashmap, kmer_s, k, kmer_count);
            assemble_pruned_graph(assembler, kmer_s, kpmer_s, hashmap, k, g, kmer_b.buckets_row_map[i]);
            printf("Time to assemble bucket %i: %.4f\n", i, t2.seconds());
            printf("Bucket %i has %u kmers and %u k+1-mers\n", i, kmer_count, kpmer_count);
            t2.reset();
        }
        printf("Time to assemble pruned graph: %.3fs\n", t.seconds());
        t.reset();
        //vtx_view_t vtx_map = generate_hashmap(kmers, k, kmers.extent(0)/k);
        //printf("kmer hashmap size: %lu\n", vtx_map.extent(0));
        //printf("Time to generate hashmap: %.3f\n", t3.seconds());
        //t3.reset();
        //std::list<graph_type> glue_list;
        //char_mirror_t kmer_copy;
        //ExperimentLoggerUtil experiment;
        //{
        //    vtx_view_t g = assemble_pruned_graph(kmers, kpmers, vtx_map, k);
        //    using coarsener_t = coarse_builder<ordinal_t, edge_offset_t, value_t, Device>;
        //    coarsener_t coarsener;
        //    //{
        //    //    t3.reset();
        //    //    graph_type g_base = assemble_graph(kmers, kpmers, vtx_map, k);
        //    //    printf("entries: %lu\n", g_base.entries.extent(0));
        //    //    printf("Time to assemble base graph: %.3f\n", t3.seconds());
        //    //    t3.reset();
        //    //    //kmer_copy = move_to_main(kmers);
        //    //    //this is likely the peak memory usage point of the program
        //    //    //don't need these anymore, delete them
        //    //    //Kokkos::resize(edge_map, 0);
        //    //    Kokkos::resize(vtx_map, 0);
        //    //    Kokkos::resize(kpmers, 0);
        //    //    //will need this later but we made a copy
        //    //    //Kokkos::resize(kmers, 0);
        //    //    g = coarsener.prune_edges(g_base);
        //    //}
        //    printf("Time to assemble pruned graph: %.3fs\n", t.seconds());
        //    t.reset();
        //    glue_list = coarsener.coarsen_de_bruijn_full_cycle(g, experiment);
        //}
        //printf("glue list length: %lu\n", glue_list.size());
        //printf("Time to generate glue list: %.3fs\n", t.seconds());
        //printf("Aggregation time: %.3fs\n", experiment.getMeasurement(ExperimentLoggerUtil::Measurement::Map));
        //printf("Heavy edge time: %.3fs\n", experiment.getMeasurement(ExperimentLoggerUtil::Measurement::Heavy));
        //printf("Pairing time: %.3fs\n", experiment.getMeasurement(ExperimentLoggerUtil::Measurement::MapConstruct));
        //printf("Pairing time specific: %.3fs\n", experiment.getMeasurement(ExperimentLoggerUtil::Measurement::CoarsenPair));
        //printf("Coarse graph build time: %.3fs\n", experiment.getMeasurement(ExperimentLoggerUtil::Measurement::Build));
        //printf("Interpolation graph transpose time: %.3fs\n", experiment.getMeasurement(ExperimentLoggerUtil::Measurement::InterpTranspose));
        //printf("Glue compact time: %.3fs\n", experiment.getMeasurement(ExperimentLoggerUtil::Measurement::CompactGlues));
        //t.reset();
        ////kmers = move_to_device(kmer_copy);
        //t.reset();
        //compress_unitigs_maximally2(kmers, glue_list, k, out_fname);
        //printf("Time to compact unitigs: %.3fs\n", t.seconds());
        //t.reset();
        //printf("Total time: %.3fs\n", t2.seconds());
        //t2.reset();
    }
    Kokkos::finalize();
    return 0;
}
