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

//graph_type move_to_device(graph_m x){
//    edge_view_t row_map("row map", x.row_map.extent(0));
//    Kokkos::deep_copy(row_map, x.row_map);
//    vtx_view_t entries("entries", x.entries.extent(0));
//    Kokkos::deep_copy(entries, x.entries);
//    return graph_type(entries, row_map);
//}

struct final_partition {
    std::vector<graph_m> glues;
    std::vector<vtx_view_t> out_maps;
    std::vector<char_mirror_t> kmer_glues;
    std::vector<edge_mirror_t> kmer_rows;
};

final_partition collect_buckets(std::vector<bucket_glues> buckets, edge_offset_t k){
    ordinal_t bucket_count = buckets[0].buckets;
    vtx_mirror_t bucket_size("bucket size", bucket_count);
    vtx_mirror_t entries_bucket_size("bucket size", bucket_count);
    vtx_mirror_t kmers_bucket_size("bucket size", bucket_count);
    for(int i = 0; i < buckets.size(); i++){
        bucket_glues b = buckets[i];
        for(int j = 0; j < bucket_count; j++){
            bucket_size(j) += b.buckets_row_map[j+1] - b.buckets_row_map[j];
            edge_offset_t end = b.buckets_entries_row_map[j+1];
            edge_offset_t start = b.buckets_entries_row_map[j];
            entries_bucket_size(j) += end - start;
            kmers_bucket_size(j) += b.kmer_rows(end) - b.kmer_rows(start);
        }
    }
    std::vector<graph_m> glues;
    std::vector<vtx_view_t> out_maps;
    std::vector<char_mirror_t> kmer_glues;
    std::vector<edge_mirror_t> kmer_rows;
    for(int i = 0; i < bucket_count; i++){
        edge_mirror_t row_map("row map", bucket_size(i) + 1);
        vtx_view_t out_map("out map", bucket_size(i));
        edge_offset_t offset = 0;
        edge_offset_t write_sum = 0;
        for(int j = 0; j < bucket_count; j++){
            bucket_glues b = buckets[j];
            ordinal_t end = b.buckets_row_map[i+1];
            ordinal_t start = b.buckets_row_map[i];
            graph_m glue = b.glues;
            vtx_view_t out_map_in = b.output_map;
            Kokkos::parallel_for("move row map", host_policy(start, end), KOKKOS_LAMBDA(const ordinal_t x){
                ordinal_t insert = offset + x - start;
                ordinal_t write_val = write_sum + glue.row_map(x + 1) - glue.row_map(start);
                row_map(insert + 1) = write_val;
            });
            Kokkos::parallel_for("move out map", r_policy(start, end), KOKKOS_LAMBDA(const ordinal_t x){
                ordinal_t insert = offset + x - start;
                out_map(insert) = out_map_in(x);
            });
            offset += end - start;
            write_sum += b.buckets_entries_row_map[i+1] - b.buckets_entries_row_map[i];
        }
        vtx_mirror_t entries("entries", entries_bucket_size(i));
        Kokkos::parallel_for("move entries", host_policy(0, entries_bucket_size(i)), KOKKOS_LAMBDA(const ordinal_t x){
            entries(x) = x;
        });
        char_mirror_t kmers("kmers", kmers_bucket_size(i));
        edge_mirror_t kmer_row("kmer row", entries_bucket_size(i) + 1);
        offset = 0, write_sum = 0;
        for(int j = 0; j < bucket_count; j++){
            bucket_glues b = buckets[j];
            edge_offset_t end = b.buckets_entries_row_map[i+1];
            edge_offset_t start = b.buckets_entries_row_map[i];
            Kokkos::parallel_for("move kmer map", host_policy(start, end), KOKKOS_LAMBDA(const ordinal_t x){
                ordinal_t insert = offset + x - start;
                ordinal_t write_val = write_sum + b.kmer_rows(x + 1) - b.kmer_rows(start);
                kmer_row(insert + 1) = write_val;
            });
            edge_offset_t kmer_start = b.kmer_rows(start);
            edge_offset_t kmer_end = b.kmer_rows(end);
            char_mirror_t dest = Kokkos::subview(kmers, std::make_pair(write_sum, write_sum + kmer_end - kmer_start));
            char_mirror_t source = Kokkos::subview(b.kmers, std::make_pair(kmer_start, kmer_end));
            Kokkos::deep_copy(dest, source);
            offset += end - start;
            write_sum += kmer_end - kmer_start;
        }
        graph_m new_glue(entries, row_map);
        glues.push_back(new_glue);
        out_maps.push_back(out_map);
        kmer_glues.push_back(kmers);
        kmer_rows.push_back(kmer_row);
    }
    final_partition out;
    out.glues = glues;
    out.out_maps = out_maps;
    out.kmer_glues = kmer_glues;
    out.kmer_rows = kmer_rows;
    return out;
}

struct kmer_partitions {
    std::vector<char_mirror_t> kmers;
    vtx_mirror_t part_sizes;
    ordinal_t size;
};

kmer_partitions collect_buckets(std::vector<bucket_kmers> buckets, edge_offset_t k){
    kmer_partitions out;
    ordinal_t bucket_count = buckets[0].buckets;
    vtx_mirror_t bucket_size("bucket size", bucket_count);
    ordinal_t total_kmers = 0;
    for(int i = 0; i < buckets.size(); i++){
        bucket_kmers b = buckets[i];
        for(int j = 0; j < bucket_count; j++){
            bucket_size(j) += b.buckets_row_map[j+1] - b.buckets_row_map[j];
        }
    }
    for(int i = 0; i < bucket_count; i++){
        total_kmers += bucket_size(i);
    }
    std::vector<char_mirror_t> all_kmers;
    for(int j = 0; j < bucket_count; j++){
        char_mirror_t kmer_bucket("kmer bucket", bucket_size(j)*k);
        edge_offset_t transfer_loc = 0;
        for(int i = 0; i < buckets.size(); i++){
            edge_offset_t transfer_start = buckets[i].buckets_row_map[j]*k;
            edge_offset_t transfer_end = buckets[i].buckets_row_map[j + 1]*k;
            edge_offset_t transfer_size = transfer_end - transfer_start;
            if(transfer_size > 0){
                char_mirror_t dest = Kokkos::subview(kmer_bucket, std::make_pair(transfer_loc, transfer_loc + transfer_size));
                char_mirror_t source = Kokkos::subview(buckets[i].kmers, std::make_pair(transfer_start, transfer_end));
                Kokkos::deep_copy(dest, source);
                transfer_loc += transfer_size;
            }
        }
        all_kmers.push_back(kmer_bucket);
    }
    out.kmers = all_kmers;
    out.part_sizes = bucket_size;
    out.size = total_kmers;
    return out;
}

struct kpmer_partitions {
    std::vector<char_mirror_t> kmers;
    vtx_mirror_t part_sizes;
    ordinal_t size;
    char_mirror_t crosscut;
    vtx_mirror_t crosscut_row_map;
    vtx_mirror_t cross_ids;
    ordinal_t crosscut_buckets;
    ordinal_t crosscut_size;
};

kpmer_partitions collect_buckets(std::vector<bucket_kpmers> buckets, edge_offset_t k){
    kpmer_partitions out;
    ordinal_t bucket_count = buckets[0].buckets;
    ordinal_t cross_bucket_count = buckets[0].crosscut_buckets;
    vtx_mirror_t bucket_size("bucket size", bucket_count);
    vtx_mirror_t cross_bucket_size("bucket size", cross_bucket_count);
    vtx_mirror_t bucket_row_map("bucket size", bucket_count + 1);
    vtx_mirror_t cross_bucket_row_map("bucket size", cross_bucket_count + 1);
    for(int i = 0; i < buckets.size(); i++){
        bucket_kpmers b = buckets[i];
        for(int j = 0; j < bucket_count; j++){
            bucket_size(j) += b.buckets_row_map[j+1] - b.buckets_row_map[j];
        }
    }
    for(int i = 0; i < bucket_count; i++){
        bucket_row_map(i + 1) = bucket_row_map(i) + bucket_size(i);
    }
    ordinal_t total_size = bucket_row_map(bucket_count);
    std::vector<char_mirror_t> all_kmers;
    for(int j = 0; j < bucket_count; j++){
        char_mirror_t kmer_bucket("kmer bucket", bucket_size(j)*k);
        edge_offset_t transfer_loc = 0;
        ordinal_t bucket_kmer_count = 0;
        for(int i = 0; i < buckets.size(); i++){
            bucket_kpmers b = buckets[i];
            edge_offset_t transfer_start = buckets[i].buckets_row_map[j]*k;
            edge_offset_t transfer_end = buckets[i].buckets_row_map[j + 1]*k;
            edge_offset_t transfer_size = transfer_end - transfer_start;
            edge_offset_t local_bucket_size = buckets[i].buckets_row_map[j+1] - buckets[i].buckets_row_map[j];
            Kokkos::parallel_for("update cross ids", host_policy(b.crosscut_row_map[j*bucket_count], b.crosscut_row_map[(j+1)*bucket_count]), KOKKOS_LAMBDA(const ordinal_t i){
                if(b.cross_ids(i) != ORD_MAX){
                    b.cross_ids(i) += bucket_kmer_count;
                }
            });
            bucket_kmer_count += buckets[i].buckets_row_map[j+1] - buckets[i].buckets_row_map[j];
            if(transfer_size > 0){
                char_mirror_t dest = Kokkos::subview(kmer_bucket, std::make_pair(transfer_loc, transfer_loc + transfer_size));
                char_mirror_t source = Kokkos::subview(buckets[i].kmers, std::make_pair(transfer_start, transfer_end));
                Kokkos::deep_copy(dest, source);
                transfer_loc += transfer_size;
            }
        }
        all_kmers.push_back(kmer_bucket);
    }
    for(int i = 0; i < buckets.size(); i++){
        bucket_kpmers b = buckets[i];
        for(int j = 0; j < cross_bucket_count; j++){
            cross_bucket_size(j) += b.crosscut_row_map[j+1] - b.crosscut_row_map[j];
        }
    }
    for(int i = 0; i < cross_bucket_count; i++){
        cross_bucket_row_map(i + 1) = cross_bucket_row_map(i) + cross_bucket_size(i);
    }
    ordinal_t cross_size = cross_bucket_row_map(cross_bucket_count);
    char_mirror_t cross_kmers("all kmers", cross_size * k);
    vtx_mirror_t cross_ids("cross ids", cross_size);
    for(int j = 0; j < cross_bucket_count; j++){
        ordinal_t transfer_loc = cross_bucket_row_map(j);
        for(int i = 0; i < buckets.size(); i++){
            ordinal_t transfer_size = buckets[i].crosscut_row_map[j+1] - buckets[i].crosscut_row_map[j];
            if(transfer_size > 0){
                char_mirror_t transfer_to = Kokkos::subview(cross_kmers, std::make_pair(k*transfer_loc, k*(transfer_loc + transfer_size)));
                char_view_t transfer_from = Kokkos::subview(buckets[i].crosscut, std::make_pair(buckets[i].crosscut_row_map[j] * k, k * buckets[i].crosscut_row_map[j + 1]));
                vtx_mirror_t dest = Kokkos::subview(cross_ids, std::make_pair(transfer_loc, transfer_loc + transfer_size));
                vtx_mirror_t source = Kokkos::subview(buckets[i].cross_ids, std::make_pair(buckets[i].crosscut_row_map[j], buckets[i].crosscut_row_map[j+1]));
                Kokkos::deep_copy(transfer_to, transfer_from);
                Kokkos::deep_copy(dest, source);
                transfer_loc += transfer_size;
            }
        }
    }
    out.kmers = all_kmers;
    out.part_sizes = bucket_size;
    out.size = total_size;
    out.crosscut = cross_kmers;
    out.crosscut_row_map = cross_bucket_row_map;
    out.crosscut_size = cross_size;
    out.crosscut_buckets = cross_bucket_count;
    out.cross_ids = cross_ids;
    return out;
}

//input file looks like
//4
//AAAA
//GGGG
//CCCC
//TTTA
template <class bucket_t, class out_t>
out_t load_kmers(char *fname, edge_offset_t k, edge_offset_t l, vtx_view_t lmin_bucket_map) {

    Kokkos::Timer t;
    std::ifstream infp(fname);
    if (!infp.is_open()) {
        printf("Error: Could not open input file. Exiting ...\n");
        exit(1);
    }

    size_t sz = getFileLength(fname);

    size_t chunk_size = sz / 16;
    size_t offset = 0;
    // Create a buffer for file
    char* s = new char[chunk_size];
    edge_offset_t n = 0, total_read = 0;
    char_mirror_t char_mirror;
    char* read_to = 0;
    printf("Time to init buffer: %.3f\n", t.seconds());
    t.reset();
    char_view_t out = char_view_t("chars", chunk_size);
    char_mirror = Kokkos::create_mirror_view(out);
    double bucket_time = 0;
    std::vector<bucket_t> bucketed_kmers;
    while(offset < sz){
        t.reset();
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
        }
        read_to = char_mirror.data();
        size_t last_read = 0;
        ordinal_t kmers_read = 0;
        printf("read chunk in %.3f seconds\n", t.seconds());
        t.reset();
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
            kmers_read++;
            total_read++;
            //increment output buffer for next kmer
            read_to += k;
            //advance past the kmer
            f += k;
        }
        offset += last_read;
        char_view_t out_sub = Kokkos::subview(out, std::make_pair((edge_offset_t)0, (edge_offset_t)k*kmers_read));
        char_mirror_t out_m_sub = Kokkos::subview(char_mirror, std::make_pair((edge_offset_t)0, (edge_offset_t)k*kmers_read));
        Kokkos::deep_copy(out_sub, out_m_sub);
        printf("packed chunk in %.3f seconds\n", t.seconds());
        t.reset();
        bucket_t kmer_b = find_l_minimizer<bucket_t>(out, k, l, lmin_bucket_map, kmers_read);
        bucketed_kmers.push_back(kmer_b);
        bucket_time += t.seconds();
        t.reset();
        if(total_read == n){
            break;
        }
    }
    t.reset();
    out_t b = collect_buckets(bucketed_kmers, k);
    printf("Recombined k-mers in %.3f seconds\n", t.seconds());
    t.reset();
    printf("bucketed chunks in %.3f seconds\n", bucket_time);
    delete[] s;
    infp.close();
    return b;
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

edge_view_t move_to_device(edge_mirror_t x){
    edge_view_t y("device", x.extent(0));
    Kokkos::deep_copy(y, x);
    return y;
}

int main(int argc, char **argv) {

    if (argc != 5) {
        printf("You input %d args\n", argc);
        fprintf(stderr, "Usage: %s <k-mer file> k l <output file>\n", argv[0]);
        return EXIT_FAILURE;
    }
    char *kmer_fname = argv[1];
    edge_offset_t k = atoi(argv[2]);
    edge_offset_t l = atoi(argv[3]);
    std::string out_fname(argv[4]);
    Kokkos::initialize();
    {
        Kokkos::Timer t, t2;//, t3;
        pool_t rand_pool(std::time(nullptr));
        ordinal_t lmin_buckets = 1;
        lmin_buckets <<= 2*l;
        vtx_view_t lmin_bucket_map = generate_permutation(lmin_buckets, rand_pool);
        kpmer_partitions kmer_b = load_kmers<bucket_kpmers, kpmer_partitions>(kmer_fname, k, l, lmin_bucket_map);
        printf("Read input data in %.3fs\n", t.seconds());
        //t.reset();
        //t2.reset();
        //bucket_kmers kmer_b = find_l_minimizer(kmers, k, l, lmin_bucket_map);
        //Kokkos::resize(kmers, 0);
        //bucket_kpmers kpmer_b = find_l_minimizer_edge(kpmers, k + 1, l, lmin_bucket_map);
        //Kokkos::resize(kpmers, 0);
        //printf("Computed l-minimizers in %.3f\n", t.seconds());
        t.reset();
        ////t2.reset();
        ////t3.reset();
#ifdef HUGE
        printf("kmer size: %lu, kmers: %lu\n", kmer_b.size*k, kmer_b.size);
#elif defined(LARGE)
        printf("kmer size: %lu, kmers: %u\n", kmer_b.size*k, kmer_b.size);
#else
        printf("kmer size: %u, kmers: %u\n", kmer_b.size*k, kmer_b.size);
#endif
        ordinal_t largest_n = 0, largest_np = 0, largest_cross = 0;
        ordinal_t bucket_count = kmer_b.kmers.size();
        for(int i = 0; i < bucket_count; i++){
            ordinal_t kmer_count = kmer_b.part_sizes(i);
            ordinal_t cross_count = kmer_b.crosscut_row_map(bucket_count*(i + 1)) - kmer_b.crosscut_row_map(bucket_count*i);
            ordinal_t kpmer_count = kmer_count + cross_count;
            if(kmer_count > largest_n){
                largest_n = kmer_count;
            }
            if(kpmer_count > largest_np){
                largest_np = kpmer_count;
            }
            if(cross_count > largest_cross){
                largest_cross = cross_count;
            }
        }
        printf("largest_np: %u\n", largest_np);
        vtx_view_t hashmap = init_hashmap(largest_n);
        assembler_data assembler = init_assembler(largest_n, largest_np);
        ExperimentLoggerUtil experiment;
        coarsener_t coarsener;
        using c_output = typename coarsener_t::coarsen_output;
        std::vector<crosses> cross_list;
        std::vector<c_output> c_outputs;
        ordinal_t cross_offset = 0;
        for(int i = 0; i < bucket_count; i++){
            Kokkos::Timer t2;
            ordinal_t kmer_count = kmer_b.part_sizes(i);
            char_view_t kmer_s("kmer part", kmer_b.kmers[i].extent(0));
            Kokkos::deep_copy(kmer_s, kmer_b.kmers[i]);
            char_mirror_t cross_s_m = Kokkos::subview(kmer_b.crosscut, std::make_pair(kmer_b.crosscut_row_map(bucket_count*i)*k, kmer_b.crosscut_row_map(bucket_count*(i+1))*k));
            vtx_mirror_t cross_ids_m = Kokkos::subview(kmer_b.cross_ids, std::make_pair(kmer_b.crosscut_row_map(bucket_count*i), kmer_b.crosscut_row_map(bucket_count*(i+1))));
            char_view_t cross_s("cross s", cross_s_m.extent(0));
            vtx_view_t cross_ids("cross ids", cross_ids_m.extent(0));
            Kokkos::deep_copy(cross_ids, cross_ids_m);
            Kokkos::deep_copy(cross_s, cross_s_m);
            generate_hashmap(hashmap, kmer_s, k, kmer_count);
            vtx_view_t g_s("graph portion", kmer_count);
            Kokkos::parallel_for("init g", kmer_count, KOKKOS_LAMBDA(const ordinal_t i){
                g_s(i) = ORD_MAX;
            });
            crosses c = assemble_pruned_graph(assembler, kmer_s, hashmap, cross_s, cross_ids, k, g_s);
            cross_list.push_back(c);
            printf("Time to assemble bucket %i: %.4f\n", i, t2.seconds());
            //printf("Bucket %i has %u kmers and %u k+1-mers\n", i, kmer_count, kpmer_count);
            t2.reset();
            c_output x = coarsener.coarsen_de_bruijn_full_cycle(g_s, c, cross_offset, experiment);
            c_outputs.push_back(x);
            printf("Time to coarsen bucket %i: %.4f\n", i, t2.seconds());
            t2.reset();
        }
        Kokkos::resize(kmer_b.crosscut, 0);
        //vtx_view_t in_cross_buf("in cross buffer", largest_cross);
        ordinal_t cross_written_count = 0;
        printf("Cross edges written: %u\n", cross_written_count);
        printf("Time to assemble pruned graph: %.3fs\n", t.seconds());
        t.reset();
        vtx_view_t small_g("small g", cross_offset);
        Kokkos::parallel_for("init g", cross_offset, KOKKOS_LAMBDA(const ordinal_t i){
            small_g(i) = ORD_MAX;
        });
        for(int i = 0; i < bucket_count; i++){
            for(int j = 0; j < bucket_count; j++){
                if(i != j){
                    ordinal_t out_bucket_begin = kmer_b.crosscut_row_map(bucket_count*i + j) - kmer_b.crosscut_row_map(bucket_count*i);
                    ordinal_t in_bucket_begin = kmer_b.crosscut_row_map(bucket_count*j + i) - kmer_b.crosscut_row_map(bucket_count*j);
                    ordinal_t bucket_size = kmer_b.crosscut_row_map(bucket_count*i + j + 1) - kmer_b.crosscut_row_map(bucket_count*i + j);
                    vtx_view_t out_cross = cross_list[i].out;
                    vtx_view_t in_cross = cross_list[j].in;
                    ordinal_t local_count = 0;
                    Kokkos::parallel_reduce("fill crosses", bucket_size, KOKKOS_LAMBDA(const ordinal_t x, ordinal_t& update){
                        ordinal_t u = out_cross(out_bucket_begin + x);
                        ordinal_t v = in_cross(in_bucket_begin + x);
                        if(u != ORD_MAX && v != ORD_MAX){
                            small_g(u) = v;
                            update++;
                        }
                    }, local_count);
                    cross_written_count += local_count;
                }
            }
        }
        graph_type small_g_result = coarsener.coarsen_de_bruijn_full_cycle_final(small_g, experiment);
        vtx_view_t repartition_map("repartition", cross_offset);
        vtx_view_t output_mapping("output mapping", cross_offset);
        ordinal_t part_size = (small_g_result.entries.extent(0) + bucket_count) / bucket_count;
        vtx_view_t part_offsets_dev("part offsets", bucket_count + 1);
        vtx_mirror_t part_offsets = Kokkos::create_mirror(part_offsets_dev);
        Kokkos::parallel_for("compute partitions", policy(small_g_result.numRows(), Kokkos::AUTO), KOKKOS_LAMBDA(const member& thread){
            ordinal_t i = thread.league_rank();
            ordinal_t start = small_g_result.row_map(i);
            ordinal_t end = small_g_result.row_map(i + 1);
            ordinal_t midpoint = (start + end) / 2;
            if(start / part_size != end / part_size){
                //write first row that is in the partition of end
                //it is either the current row or the next row
                if(midpoint / part_size == end / part_size){
                    part_offsets_dev(end / part_size) = i;
                } else {
                    part_offsets_dev(end / part_size) = i + 1;
                }
            }
            Kokkos::parallel_for(Kokkos::TeamThreadRange(thread, small_g_result.row_map(i), small_g_result.row_map(i + 1)), [=] (const ordinal_t j){
                repartition_map(small_g_result.entries(j)) = midpoint / part_size;
                output_mapping(small_g_result.entries(j)) = j;
            });
        });
        Kokkos::deep_copy(part_offsets, part_offsets_dev);
        part_offsets(bucket_count) = small_g_result.numRows();
        std::vector<bucket_glues> output_glues;
        ordinal_t glue_offset = 0;
        Kokkos::Timer t3;
        //repartition kmers according to output unitig
        for(int i = 0; i < bucket_count; i++) {
            Kokkos::Timer t4;
            char_view_t kmer_s("kmer part", kmer_b.kmers[i].extent(0));
            Kokkos::deep_copy(kmer_s, kmer_b.kmers[i]);
            c_output c = c_outputs[i];
            ordinal_t glue_size = c.cross.numRows();
            vtx_view_t repart_s = Kokkos::subview(repartition_map, std::make_pair(glue_offset, glue_offset + glue_size));
            vtx_view_t output_s = Kokkos::subview(output_mapping, std::make_pair(glue_offset, glue_offset + glue_size));
            glue_offset += glue_size;
            graph_type non_cross_glue = move_to_device(c.glue);
            graph_type cross_glue = move_to_device(c.cross);
            printf("Time to move bucket %i's stuff to device: %.3fs\n", i, t4.seconds());
            t4.reset();
            write_unitigs2(kmer_s, k, non_cross_glue, out_fname);
            printf("Time to write bucket %i's non-crossing output: %.3fs\n", i, t4.seconds());
            t4.reset();
            bucket_glues glue_b = partition_for_output(bucket_count, cross_glue, repart_s, output_s);
            printf("Time to calculate bucket %i's crossing output partitions: %.3fs\n", i, t4.seconds());
            t4.reset();
            glue_b = partition_kmers_for_glueing(glue_b, kmer_s, k);
            printf("Time to repartition bucket %i's kmers: %.3fs\n", i, t4.seconds());
            t4.reset();
            output_glues.push_back(glue_b);
        }
        c_outputs.clear();
        kmer_b.kmers.clear();
        printf("Coarsened vertices reordered: %u\n", glue_offset);
        printf("Time to repartition kmers: %.3fs\n", t3.seconds());
        t3.reset();
        final_partition f_p = collect_buckets(output_glues, k);
        ordinal_t glue_start = 0, glue_end = 0;
        for(int i = 0; i < bucket_count; i++){
            glue_start = part_offsets(i);
            glue_end = part_offsets(i + 1);
            vtx_view_t output_s = f_p.out_maps[i];
            Kokkos::parallel_for("modify result entries", output_s.extent(0), KOKKOS_LAMBDA(const ordinal_t i){
                small_g_result.entries(output_s(i)) = i;
            });
            Kokkos::View<const edge_offset_t> result_start_s = Kokkos::subview(small_g_result.row_map, glue_start);
            Kokkos::View<const edge_offset_t> result_end_s = Kokkos::subview(small_g_result.row_map, glue_end);
            edge_offset_t result_start = 0, result_end = 0;
            Kokkos::deep_copy(result_start, result_start_s);
            Kokkos::deep_copy(result_end, result_end_s);
            edge_view_t row_map("row map", 1 + glue_end - glue_start);
            printf("result size: %u; graph size: %u\n", result_end - result_start, f_p.glues[i].numRows());
            Kokkos::parallel_for("write row map", glue_end - glue_start, KOKKOS_LAMBDA(const ordinal_t i){
                row_map(i + 1) = small_g_result.row_map(glue_start + i + 1) - small_g_result.row_map(glue_start);
            });
            vtx_view_t entries = Kokkos::subview(small_g_result.entries, std::make_pair(result_start, result_end));
            graph_type out_g(entries, row_map);
            graph_type blah = coarsener.compacter.collect_unitigs(move_to_device(f_p.glues[i]), out_g);
            write_unitigs3(move_to_device(f_p.kmer_glues[i]), move_to_device(f_p.kmer_rows[i]), k, blah, out_fname);
        }
        //printf("glue list length: %lu\n", glue_list.size());
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
        ////kmers = move_to_device(kmer_copy);
        //t.reset();
        //compress_unitigs_maximally2(kmer_b.kmers, glue_list, k, out_fname);
        printf("Time to compact unitigs: %.3fs\n", t.seconds());
        t.reset();
        printf("Total time: %.3fs\n", t2.seconds());
        t2.reset();
    }
    Kokkos::finalize();
    return 0;
}
