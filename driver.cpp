#include "coarseners.h"
#include "heuristics.h"
#include "assembler.h"
#include "compact.h"
#include "ExperimentLoggerUtil.cpp"
#include "defs.h"
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

size_t getFileLength(const char *fname){
    // Obtain the size of the file.
    return fs::file_size(fname);
}

struct minitig_partition {
    std::vector<minitigs> minis;
    std::vector<vtx_view_t> out_maps;
};

minitig_partition collect_buckets(std::vector<bucket_minitigs> buckets){
    ordinal_t bucket_count = buckets[0].buckets;
    vtx_mirror_t bucket_size("bucket size", bucket_count);
    edge_mirror_t minitig_bucket_size("bucket size", bucket_count);
    for(int i = 0; i < buckets.size(); i++){
        bucket_minitigs b = buckets[i];
        for(int j = 0; j < bucket_count; j++){
            edge_offset_t end = b.part_offsets[j+1];
            edge_offset_t start = b.part_offsets[j];
            bucket_size(j) += end - start;
            minitig_bucket_size(j) += b.m.row_map(end) - b.m.row_map(start);
        }
    }
    std::vector<vtx_view_t> out_maps;
    std::vector<minitigs> minis;
    for(int i = 0; i < bucket_count; i++){
        edge_mirror_t row_map("row map", bucket_size(i) + 1);
        vtx_view_t out_map(Kokkos::ViewAllocateWithoutInitializing("out map"), bucket_size(i));
        vtx_mirror_t lengths(Kokkos::ViewAllocateWithoutInitializing("lengths"), bucket_size(i));
        comp_mt chars(Kokkos::ViewAllocateWithoutInitializing("chars"), minitig_bucket_size(i));
        edge_offset_t offset = 0;
        edge_offset_t write_sum = 0;
        for(int j = 0; j < bucket_count; j++){
            bucket_minitigs b = buckets[j];
            ordinal_t end = b.part_offsets[i+1];
            ordinal_t start = b.part_offsets[i];
            edge_offset_t kmer_start = b.m.row_map(start);
            edge_offset_t kmer_end = b.m.row_map(end);
            vtx_view_t out_map_in = b.output_map;
            Kokkos::parallel_for("move row map", host_policy(start, end), KOKKOS_LAMBDA(const ordinal_t x){
                ordinal_t insert = offset + x - start;
                ordinal_t write_val = write_sum + b.m.row_map(x + 1) - b.m.row_map(start);
                row_map(insert + 1) = write_val;
            });
            Kokkos::parallel_for("move out map", r_policy(start, end), KOKKOS_LAMBDA(const ordinal_t x){
                ordinal_t insert = offset + x - start;
                out_map(insert) = out_map_in(x);
            });
            Kokkos::parallel_for("move out map", host_policy(start, end), KOKKOS_LAMBDA(const ordinal_t x){
                ordinal_t insert = offset + x - start;
                lengths(insert) = b.m.lengths(x);
            });
            comp_mt dest = Kokkos::subview(chars, std::make_pair(write_sum, write_sum + kmer_end - kmer_start));
            comp_mt source = Kokkos::subview(b.m.chars, std::make_pair(kmer_start, kmer_end));
            Kokkos::deep_copy(dest, source);
            offset += end - start;
            write_sum += b.m.row_map(end) - b.m.row_map(start);
        }
        minitigs x;
        x.lengths = lengths;
        x.size = bucket_size(i);
        x.chars = chars;
        x.row_map = row_map;
        out_maps.push_back(out_map);
        minis.push_back(x);
    }
    minitig_partition out;
    out.out_maps = out_maps;
    out.minis = minis;
    return out;
}

struct kpmer_partitions {
    std::vector<comp_mt> kmers;
    vtx_mirror_t part_sizes;
    ordinal_t size;
    comp_mt crosscut;
    vtx_mirror_t crosscut_row_map;
    vtx_mirror_t cross_lend, cross_borrow;
    ordinal_t crosscut_buckets;
    ordinal_t crosscut_size;
};

char_view_t decompress_kmers(comp_vt comped, edge_offset_t k){
    edge_offset_t k_pad = ((k + 15) / 16) * 16;
    edge_offset_t comp_size = k_pad / 16;
    ordinal_t kmers_in_bucket = comped.extent(0) / comp_size;
    printf("kmers in bucket: %u\n", kmers_in_bucket);
    char_view_t kmers(Kokkos::ViewAllocateWithoutInitializing("kmers"), kmers_in_bucket * k);
    char_mirror_t char_map_mirror("char map mirror", 4);
    char_map_mirror(0) = 'A';
    char_map_mirror(1) = 'C';
    char_map_mirror(2) = 'G';
    char_map_mirror(3) = 'T';
    char_view_t char_map("char map", 4);
    Kokkos::deep_copy(char_map, char_map_mirror);
    Kokkos::parallel_for("decompress kmers", kmers_in_bucket, KOKKOS_LAMBDA(const ordinal_t x){
        for(edge_offset_t i = 0; i < comp_size; i++){
            uint32_t bytes = comped(x*comp_size + i);
            for(edge_offset_t j = 0; j < 16; j++){
                char b = (bytes >> (2*j)) & 3;
                b = char_map(b);
                if(16*i + j < k) kmers(x*k + 16*i + j) = b;
            }
        }
    });
    return kmers;
}

kpmer_partitions collect_buckets(std::vector<bucket_kpmers>& buckets, edge_offset_t k){
    kpmer_partitions out;
    ordinal_t bucket_count = buckets[0].buckets;
    ordinal_t cross_bucket_count = buckets[0].crosscut_buckets;
    vtx_mirror_t bucket_size("bucket size", bucket_count);
    vtx_mirror_t cross_bucket_size("bucket size", cross_bucket_count);
    vtx_mirror_t bucket_row_map("bucket size", bucket_count + 1);
    vtx_mirror_t cross_bucket_row_map("bucket size", cross_bucket_count + 1);
    edge_offset_t k_pad = ((k + 15) / 16) * 16;
    edge_offset_t comp_size = k_pad / 16;
    for(int i = 0; i < buckets.size(); i++){
        bucket_kpmers b = buckets[i];
        auto b_kmers = b.kmers.begin();
        for(int j = 0; j < bucket_count; j++){
            bucket_size(j) += b_kmers->extent(0)/comp_size;
            b_kmers++;
        }
    }
    for(int i = 0; i < bucket_count; i++){
        bucket_row_map(i + 1) = bucket_row_map(i) + bucket_size(i);
    }
    ordinal_t total_size = bucket_row_map(bucket_count);
    std::vector<comp_mt> all_kmers;
    for(int j = 0; j < bucket_count; j++){
        Kokkos::Timer t;
        edge_offset_t bytes = bucket_size(j)*comp_size;
        comp_mt kmer_bucket(Kokkos::ViewAllocateWithoutInitializing("kmer bucket"), bytes);
        edge_offset_t transfer_loc = 0;
        ordinal_t bucket_kmer_count = 0;
        for(int i = 0; i < buckets.size(); i++){
            bucket_kpmers& b = buckets[i];
            comp_mt source = b.kmers.front();
            b.kmers.pop_front();
            edge_offset_t transfer_size = source.extent(0);
            Kokkos::parallel_for("update cross ids", host_policy(b.crosscut_row_map[j*bucket_count], b.crosscut_row_map[(j+1)*bucket_count]), KOKKOS_LAMBDA(const ordinal_t x){
                if(b.cross_lend(x) != ORD_MAX){
                    b.cross_lend(x) += bucket_kmer_count;
                }
                if(b.cross_borrow(x) != ORD_MAX){
                    b.cross_borrow(x) += bucket_kmer_count;
                }
            });
            bucket_kmer_count += transfer_size / comp_size;
            if(transfer_size > 0){
                comp_mt dest = Kokkos::subview(kmer_bucket, std::make_pair(transfer_loc, transfer_loc + transfer_size));
                Kokkos::deep_copy(dest, source);
                transfer_loc += transfer_size;
            }
        }
        all_kmers.push_back(kmer_bucket);
        printf("Formed bucket %i (containing %li bytes) in %.3f (%.2f GB/s)\n", j, 4*bytes, t.seconds(), static_cast<double>(4*bytes) / (t.seconds()*1000000000.0));
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
    vtx_mirror_t cross_lend(Kokkos::ViewAllocateWithoutInitializing("cross ids"), cross_size);
    vtx_mirror_t cross_borrow(Kokkos::ViewAllocateWithoutInitializing("cross ids"), cross_size);
    for(int j = 0; j < cross_bucket_count; j++){
        ordinal_t transfer_loc = cross_bucket_row_map(j);
        for(int i = 0; i < buckets.size(); i++){
            ordinal_t transfer_size = buckets[i].crosscut_row_map[j+1] - buckets[i].crosscut_row_map[j];
            if(transfer_size > 0){
                vtx_mirror_t dest = Kokkos::subview(cross_lend, std::make_pair(transfer_loc, transfer_loc + transfer_size));
                vtx_mirror_t source = Kokkos::subview(buckets[i].cross_lend, std::make_pair(buckets[i].crosscut_row_map[j], buckets[i].crosscut_row_map[j+1]));
                Kokkos::deep_copy(dest, source);
                dest = Kokkos::subview(cross_borrow, std::make_pair(transfer_loc, transfer_loc + transfer_size));
                source = Kokkos::subview(buckets[i].cross_borrow, std::make_pair(buckets[i].crosscut_row_map[j], buckets[i].crosscut_row_map[j+1]));
                Kokkos::deep_copy(dest, source);
                transfer_loc += transfer_size;
            }
        }
    }
    out.kmers = all_kmers;
    out.part_sizes = bucket_size;
    out.size = total_size;
    out.crosscut_row_map = cross_bucket_row_map;
    out.crosscut_size = cross_size;
    out.crosscut_buckets = cross_bucket_count;
    out.cross_lend = cross_lend;
    out.cross_borrow = cross_borrow;
    return out;
}

comp_mt compact_kmers(comp_mt buffer, char_mirror_t kmers, edge_offset_t k, ordinal_t n){
    vtx_mirror_t char_map("char map", 256);
    char_map('A') = 0;
    char_map('C') = 1;
    char_map('G') = 2;
    char_map('T') = 3;
    edge_offset_t k_pad = ((k + 15) / 16) * 16;
    edge_offset_t comp_size = k_pad / 16;
    comp_mt kmer_compress = Kokkos::subview(buffer, std::make_pair((edge_offset_t) 0, n * comp_size));
    Kokkos::parallel_for("compress kmers", host_policy(0, n), KOKKOS_LAMBDA(const edge_offset_t j){
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

    size_t chunk_size = sz / 64;
    size_t offset = 0;
    // Create a buffer for file
    char* s = new char[chunk_size];
    edge_offset_t n = 0, total_read = 0;
    char* read_to = 0;
    printf("Time to init buffer: %.3f\n", t.seconds());
    t.reset();
    edge_offset_t k_pad = ((k + 15) / 16) * 16;
    edge_offset_t comp_size = k_pad / 16;
    size_t buff_size = chunk_size / k;
    buff_size *= comp_size;
    comp_vt out("chars", buff_size);
    comp_mt comp_buf("chars", buff_size);
    char_mirror_t packed_chars(Kokkos::ViewAllocateWithoutInitializing("packed chars"), chunk_size);
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
        read_to = packed_chars.data();
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
        printf("packed chunk in %.3f seconds\n", t.seconds());
        t.reset();
        comp_mt out_m_sub = compact_kmers(comp_buf, packed_chars, k, kmers_read);
        comp_vt out_sub = Kokkos::subview(out, std::make_pair((edge_offset_t)0, (edge_offset_t)comp_size*kmers_read));
        Kokkos::deep_copy(out_sub, out_m_sub);
        printf("transferred chunk to device in %.3f seconds\n", t.seconds());
        t.reset();
        bucket_t kmer_b = find_l_minimizer<bucket_t>(out_sub, k, l, lmin_bucket_map, kmers_read);
        bucketed_kmers.push_back(kmer_b);
        printf("organized chunk by lmins in %.3f seconds\n", t.seconds());
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
        //generate a random priority table for the lmers
        vtx_view_t lmin_bucket_map = generate_permutation(lmin_buckets, rand_pool);
        //load and partition kmers into buckets
        kpmer_partitions kmer_b = load_kmers<bucket_kpmers, kpmer_partitions>(kmer_fname, k, l, lmin_bucket_map);
        printf("Read input data and bucketed by l-minimizers in %.3fs\n", t.seconds());
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
        ordinal_t largest_n = 0;
        ordinal_t bucket_count = kmer_b.kmers.size();
        //calculate the largest bucket
        for(int i = 0; i < bucket_count; i++){
            ordinal_t kmer_count = kmer_b.part_sizes(i);
            if(kmer_count > largest_n){
                largest_n = kmer_count;
            }
        }
        printf("largest_n: %u\n", largest_n);
        //init hashmap and related datastructures
        vtx_view_t hashmap = init_hashmap(largest_n);
        assembler_data assembler = init_assembler(largest_n);
        ExperimentLoggerUtil experiment;
        coarsener_t coarsener;
        using c_output = typename coarsener_t::coarsen_output;
        //contains the inter-bucket edges
        std::vector<crosses> cross_list;
        //contains the output of the coarsenings
        std::vector<minitigs> c_outputs;
        ordinal_t cross_offset = 0;
        //for each bucket, assemble the local graph, and coarsen it
        //track vertices of inter-bucket edges in each bucket, and translate them to coarse vtx ids
        edge_offset_t k_pad = ((k + 15) / 16) * 16;
        edge_offset_t comp_size = k_pad / 16;
        for(int i = 0; i < bucket_count; i++){
            Kokkos::Timer t2;
            t2.reset();
            //move data to device
            ordinal_t kmer_count = kmer_b.part_sizes(i);
            comp_vt kmer_compress(Kokkos::ViewAllocateWithoutInitializing("kmer part"), kmer_b.kmers[i].extent(0));
            Kokkos::deep_copy(kmer_compress, kmer_b.kmers[i]);
            printf("Time to move bucket %i to device: %.3fs\n", i, t2.seconds());
            t2.reset();
            vtx_mirror_t cross_lend_m = Kokkos::subview(kmer_b.cross_lend, std::make_pair(kmer_b.crosscut_row_map(bucket_count*i), kmer_b.crosscut_row_map(bucket_count*(i+1))));
            vtx_view_t cross_lend("cross ids", cross_lend_m.extent(0));
            Kokkos::deep_copy(cross_lend, cross_lend_m);
            vtx_mirror_t cross_borrow_m = Kokkos::subview(kmer_b.cross_borrow, std::make_pair(kmer_b.crosscut_row_map(bucket_count*i), kmer_b.crosscut_row_map(bucket_count*(i+1))));
            vtx_view_t cross_borrow("cross ids", cross_borrow_m.extent(0));
            Kokkos::deep_copy(cross_borrow, cross_borrow_m);
            printf("Time to move crossers to device: %.3fs\n", t2.seconds());
            t2.reset();
            Kokkos::Timer t5;
            //insert k-1 prefixes into hashmap
            generate_hashmap(hashmap, kmer_compress, comp_size, kmer_count);
            printf("Time to generate hashmap: %.3fs\n", t5.seconds());
            t5.reset();
            vtx_view_t g_s(Kokkos::ViewAllocateWithoutInitializing("graph portion"), kmer_count);
            Kokkos::deep_copy(g_s, ORD_MAX);
            //assemble local graph by looking up k-1 suffixes inside hashmap
            crosses c = assemble_pruned_graph(assembler, kmer_compress, hashmap, cross_lend, cross_borrow, comp_size, g_s);
            cross_list.push_back(c);
            printf("Time to assemble bucket %i: %.4f\n", i, t2.seconds());
            t2.reset();
            //coarsen local graph and relabel vertices of inter-bucket edges
            c_output x = coarsener.coarsen_de_bruijn_full_cycle(g_s, c, cross_offset, experiment);
            printf("Time to coarsen bucket %i: %.4f\n", i, t2.seconds());
            t2.reset();
            write_intra_bucket_outputs(kmer_compress, k, comp_size, x.glue, out_fname);
            //write output of this bucket that does not require knowledge of other buckets
            printf("Time to write bucket %i's non-crossing output: %.3fs\n", i, t2.seconds());
            t2.reset();
            //generate partial unitigs from kmers
            minitigs y = generate_minitigs(x.cross, kmer_compress, k, comp_size);
            c_outputs.push_back(y);
            printf("Time to repartition bucket %i's kmers: %.3fs\n", i, t2.seconds());
            t2.reset();
        }
        kmer_b.kmers.clear();
        Kokkos::resize(kmer_b.crosscut, 0);
        ordinal_t cross_written_count = 0;
        printf("Time to assemble pruned graph: %.3fs\n", t.seconds());
        t.reset();
        vtx_view_t small_g(Kokkos::ViewAllocateWithoutInitializing("small g"), cross_offset);
        Kokkos::deep_copy(small_g, ORD_MAX);
        //assemble the graph induced by the inter-bucket edges, maximally coarsened within each bucket
        for(int i = 0; i < bucket_count; i++){
            for(int j = 0; j < bucket_count; j++){
                //handle edges from bucket i to bucket j
                if(i != j){
                    ordinal_t out_bucket_begin = kmer_b.crosscut_row_map(bucket_count*i + j) - kmer_b.crosscut_row_map(bucket_count*i);
                    ordinal_t in_bucket_begin = kmer_b.crosscut_row_map(bucket_count*j + i) - kmer_b.crosscut_row_map(bucket_count*j);
                    ordinal_t bucket_size = kmer_b.crosscut_row_map(bucket_count*i + j + 1) - kmer_b.crosscut_row_map(bucket_count*i + j);
                    //source vertices in bucket i
                    vtx_view_t out_cross = cross_list[i].out;
                    //destination vertices in bucket j
                    vtx_view_t in_cross = cross_list[j].in;
                    ordinal_t local_count = 0;
                    Kokkos::parallel_reduce("fill crosses", bucket_size, KOKKOS_LAMBDA(const ordinal_t x, ordinal_t& update){
                        ordinal_t u = out_cross(out_bucket_begin + x);
                        ordinal_t v = in_cross(in_bucket_begin + x);
                        //both vertices must be known in order to assign an edge
                        //one may be known but not the other if one vertex had multiple edges 
                        if(u != ORD_MAX && v != ORD_MAX){
                            small_g(u) = v;
                            update++;
                        }
                    }, local_count);
                    cross_written_count += local_count;
                }
            }
        }
        printf("Cross edges written: %u\n", cross_written_count);
        //coarsen the final graph
        //for k = 31, this will have about 6-7% as many vertices as kmers in the original input
        //for k = 63, this is about 3-4%
        graph_type small_g_result = coarsener.coarsen_de_bruijn_full_cycle_final(small_g, experiment);
        vtx_view_t repartition_map("repartition", cross_offset);
        vtx_view_t output_mapping("output mapping", cross_offset);
        ordinal_t part_size = (small_g_result.entries.extent(0) + bucket_count) / bucket_count;
        vtx_view_t part_offsets_dev("part offsets", bucket_count + 1);
        vtx_mirror_t part_offsets = Kokkos::create_mirror(part_offsets_dev);
        //assign each vertex in small_g to an output partition according to its fully coarsened id
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
        std::vector<bucket_minitigs> output_minitigs;
        ordinal_t glue_offset = 0;
        Kokkos::Timer t3;
        //repartition kmers according to output unitig
        for(int i = 0; i < bucket_count; i++) {
            Kokkos::Timer t4;
            //move data to device
            minitigs x = c_outputs[i];
            ordinal_t glue_size = x.size;
            vtx_view_t repart_s = Kokkos::subview(repartition_map, std::make_pair(glue_offset, glue_offset + glue_size));
            vtx_view_t output_s = Kokkos::subview(output_mapping, std::make_pair(glue_offset, glue_offset + glue_size));
            glue_offset += glue_size;
            printf("Time to move bucket %i's stuff to device: %.3fs\n", i, t4.seconds());
            t4.reset();
            //determine output partition of each kmer that is part of an inter-bucket unitig
            bucket_minitigs minitig_b = partition_for_output(bucket_count, x, k, repart_s, output_s);
            printf("Time to calculate bucket %i's crossing output partitions: %.3fs\n", i, t4.seconds());
            t4.reset();
            output_minitigs.push_back(minitig_b);
        }
        c_outputs.clear();
        kmer_b.kmers.clear();
        printf("Coarsened vertices reordered: %u\n", glue_offset);
        printf("Time to repartition kmers: %.3fs\n", t3.seconds());
        t3.reset();
        //form output partitions by collecting partitions from each bucket
        minitig_partition f_p = collect_buckets(output_minitigs);
        ordinal_t glue_start = 0, glue_end = 0;
        //for each output partition, form each maximal unitig and write to output
        for(int i = 0; i < bucket_count; i++){
            glue_start = part_offsets(i);
            glue_end = part_offsets(i + 1);
            vtx_view_t output_s = f_p.out_maps[i];
            //move data to device
            Kokkos::parallel_for("modify result entries", output_s.extent(0), KOKKOS_LAMBDA(const ordinal_t i){
                small_g_result.entries(output_s(i)) = i;
            });
            Kokkos::View<const edge_offset_t> result_start_s = Kokkos::subview(small_g_result.row_map, glue_start);
            Kokkos::View<const edge_offset_t> result_end_s = Kokkos::subview(small_g_result.row_map, glue_end);
            edge_offset_t result_start = 0, result_end = 0;
            Kokkos::deep_copy(result_start, result_start_s);
            Kokkos::deep_copy(result_end, result_end_s);
            edge_view_t row_map("row map", 1 + glue_end - glue_start);
            printf("result size: %u; graph size: %u\n", result_end - result_start, f_p.minis[i].size);
            Kokkos::parallel_for("write row map", glue_end - glue_start, KOKKOS_LAMBDA(const ordinal_t i){
                row_map(i + 1) = small_g_result.row_map(glue_start + i + 1) - small_g_result.row_map(glue_start);
            });
            vtx_view_t entries = Kokkos::subview(small_g_result.entries, std::make_pair(result_start, result_end));
            graph_type out_g(entries, row_map);
            //write to file
            write_unitigs4(f_p.minis[i].chars, f_p.minis[i].row_map, f_p.minis[i].lengths, k, Kokkos::create_mirror(out_g), out_fname);
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
