#include "defs.h"

template< class ScalarType, int N >
struct array_type {
    ScalarType the_array[N];

    KOKKOS_INLINE_FUNCTION   // Default constructor - Initialize to 0's
    array_type() {
        for (int i = 0; i < N; i++ ) { the_array[i] = 0; }
    }

    KOKKOS_INLINE_FUNCTION   // Copy Constructor
    array_type(const array_type & rhs) {
        for (int i = 0; i < N; i++ ){
            the_array[i] = rhs.the_array[i];
        }
    }

    KOKKOS_INLINE_FUNCTION   // add operator
    array_type& operator += (const array_type& src) {
        for ( int i = 0; i < N; i++ ) {
            the_array[i]+=src.the_array[i];
        }
        return *this;
    }

    KOKKOS_INLINE_FUNCTION   // volatile add operator
    void operator += (const volatile array_type& src) volatile {
        for ( int i = 0; i < N; i++ ) {
            the_array[i]+=src.the_array[i];
        }
    }
};

const int large_buckets = 64;
using reduce_t = array_type<unitig_compact::ordinal_t, large_buckets>;
namespace Kokkos { //reduction identity must be defined in Kokkos namespace
   template<>
   struct reduction_identity< reduce_t > {
      KOKKOS_FORCEINLINE_FUNCTION static reduce_t sum() {
         return reduce_t();
      }
   };
}

namespace unitig_compact{

KOKKOS_INLINE_FUNCTION
void get_double_lmin(const char_view_t& chars, const vtx_view_t& lmin_map, const vtx_view_t& char_map, edge_offset_t offset, edge_offset_t k, edge_offset_t l, ordinal_t& l1, ordinal_t& l2){
    const ordinal_t lmer_mask = (1 << (2*l)) - 1;
    ordinal_t lmer_id = 0;
    l1 = ORD_MAX;
    l2 = ORD_MAX;
    for(edge_offset_t i = offset; i < offset + k; i++)
    {
        char c = chars(i);
        ordinal_t c_val = char_map(c);
        //emplace c_val into the least significant two bits and trim the most significant bits
        lmer_id <<= 2;
        lmer_id ^= c_val;
        lmer_id &= lmer_mask;

        if(i + 1 - offset >= l && i + 1 < offset + k){
            if(l1 > lmin_map(lmer_id)){
                l1 = lmin_map(lmer_id);
            }
        }
        if(i + 1 - offset >= l + 1){
            if(l2 > lmin_map(lmer_id)){
                l2 = lmin_map(lmer_id);
            }
        }
    }
}

KOKKOS_INLINE_FUNCTION
ordinal_t get_lmin(const char_view_t chars, const vtx_view_t lmin_map, const vtx_view_t char_map, edge_offset_t offset, edge_offset_t k, edge_offset_t l){
    const ordinal_t lmer_mask = (1 << (2*l)) - 1;
    ordinal_t lmer_id = 0;
    //ordinal_t lmin = 0;
    ordinal_t lmin_order = ORD_MAX;
    for(edge_offset_t i = offset; i < offset + k; i++)
    {
        char c = chars(i);
        ordinal_t c_val = char_map(c);
        //emplace c_val into the least significant two bits and trim the most significant bits
        lmer_id <<= 2;
        lmer_id ^= c_val;
        lmer_id &= lmer_mask;

        if(i + 1 - offset >= l){
            if(lmin_order > lmin_map(lmer_id)){
                lmin_order = lmin_map(lmer_id);
                //lmin = lmer_id;
            }
        }
    }
    return lmin_order;
}

struct minitigs {
    edge_mirror_t row_map;
    char_mirror_t chars;
    ordinal_t size;
};

struct minitigs_d {
    edge_view_t row_map;
    char_view_t chars;
    ordinal_t size;
};

struct bucket_kmers {
    char_mirror_t kmers;
    vtx_mirror_t buckets_row_map;
    ordinal_t buckets;
    ordinal_t size;
};

struct bucket_kpmers {
    std::list<char_mirror_t> kmers;
    ordinal_t buckets;
    ordinal_t size;
    char_mirror_t crosscut;
    vtx_mirror_t cross_ids;
    vtx_mirror_t crosscut_row_map;
    ordinal_t crosscut_buckets;
    ordinal_t crosscut_size;
};

struct bucket_glues {
    char_mirror_t kmers;
    edge_mirror_t kmer_rows;
    graph_m glues;
    vtx_mirror_t buckets_row_map;
    vtx_mirror_t buckets_entries_row_map;
    ordinal_t buckets;
    vtx_view_t output_map;
};

struct bucket_minitigs {
    minitigs m;
    vtx_mirror_t part_offsets;
    ordinal_t buckets;
    vtx_view_t output_map;
};

minitigs_d move_to_device(minitigs x){
    minitigs_d y;
    y.row_map = edge_view_t("row map", x.row_map.extent(0));
    Kokkos::deep_copy(y.row_map, x.row_map);
    y.chars = char_view_t("chars", x.chars.extent(0));
    Kokkos::deep_copy(y.chars, x.chars);
    y.size = x.size;
    return y;
}

graph_type move_to_device(graph_m x){
    edge_view_t row_map("row map", x.row_map.extent(0));
    Kokkos::deep_copy(row_map, x.row_map);
    vtx_view_t entries("entries", x.entries.extent(0));
    Kokkos::deep_copy(entries, x.entries);
    return graph_type(entries, row_map);
}

minitigs generate_minitigs(graph_type glues, char_view_t kmers, edge_offset_t k){
    ordinal_t n = glues.numRows();
    edge_view_t row_map("minitig row map", n + 1);
    edge_offset_t total_chars = 0;
    Kokkos::parallel_scan("write minitig row map", n, KOKKOS_LAMBDA(const ordinal_t i, edge_offset_t& update, const bool final){
        update += k - 1 + glues.row_map(i + 1) - glues.row_map(i);
        if(final){
            row_map(i + 1) = update;
        }
    }, total_chars);
    char_view_t chars("minitig chars", total_chars);
    Kokkos::parallel_for("write minitig chars", policy(n, Kokkos::AUTO), KOKKOS_LAMBDA(const member& thread){
        ordinal_t i = thread.league_rank();
        edge_offset_t write_to = row_map(i);
        Kokkos::parallel_for(Kokkos::TeamThreadRange(thread, glues.row_map(i), glues.row_map(i + 1)), [=] (const edge_offset_t j){
            ordinal_t u = glues.entries(j);
            edge_offset_t local_write = write_to + j - glues.row_map(i);
            if(j + 1 < glues.row_map(i + 1)){
                chars(local_write) = kmers(u*k);
            } else {
                for(edge_offset_t l = u*k; l < (u+1)*k; l++){
                    chars(local_write++) = kmers(l);
                }
            }
        });
    });
    minitigs out;
    out.row_map = edge_mirror_t(Kokkos::ViewAllocateWithoutInitializing("minitig row map host"), n + 1);
    Kokkos::deep_copy(out.row_map, row_map);
    out.chars = char_mirror_t(Kokkos::ViewAllocateWithoutInitializing("minitig chars host"), total_chars);
    Kokkos::deep_copy(out.chars, chars);
    out.size = n;
    return out;
}

bucket_glues partition_kmers_for_glueing(bucket_glues glues, char_view_t kmers, edge_offset_t k){
    graph_type g_g = move_to_device(glues.glues);
    ordinal_t n = g_g.entries.extent(0);
    edge_view_t kmer_rows("kmer rows", n + 1);
    Kokkos::parallel_scan("name", g_g.numRows(), KOKKOS_LAMBDA(const ordinal_t i, edge_offset_t& update, const bool final){
        if(final){
            ordinal_t u = g_g.row_map(i);
            kmer_rows(u) = update;
            edge_offset_t write_val = update + k;
            u++;
            for(; u < g_g.row_map(i + 1); u++){
                kmer_rows(u) = write_val;
                write_val++;
            }
            if(i + 1 == g_g.numRows()){
                kmer_rows(u) = write_val;
            } 
        }
        update += k - 1 + (g_g.row_map(i + 1) - g_g.row_map(i));
    });
    edge_subview_t kmer_glue_size_s = Kokkos::subview(kmer_rows, n);
    edge_offset_t kmer_glue_size = 0;
    Kokkos::deep_copy(kmer_glue_size, kmer_glue_size_s);
    char_view_t kmer_glues(Kokkos::ViewAllocateWithoutInitializing("kmers glue"), kmer_glue_size);
    Kokkos::parallel_for("shuffle kmers for glueing", g_g.entries.extent(0), KOKKOS_LAMBDA(const ordinal_t i){
        edge_offset_t read_idx = ((g_g.entries(i) + 1)*k) - 1;
        //j + 1 >= kmer_rows(i) + 1 because kmer_rows(i) can be zero and its datatype is unsigned
        for(edge_offset_t j = kmer_rows(i + 1) - 1; j + 1 >= kmer_rows(i) + 1; j--){
            kmer_glues(j) = kmers(read_idx);
            read_idx--;
        }
    });
    glues.kmers = char_mirror_t(Kokkos::ViewAllocateWithoutInitializing("kmers host"), kmer_glue_size);
    Kokkos::deep_copy(glues.kmers, kmer_glues);
    glues.kmer_rows = edge_mirror_t(Kokkos::ViewAllocateWithoutInitializing("kmer rows host"), n + 1);
    Kokkos::deep_copy(glues.kmer_rows, kmer_rows);
    return glues;
}

minitigs permute_minitigs(minitigs x_host, vtx_view_t permute){
    minitigs_d x = move_to_device(x_host);
    ordinal_t n = x.size;
    edge_view_t row_map("minitig row map", n + 1);
    edge_offset_t total_chars = x.chars.extent(0);
    Kokkos::parallel_scan("write minitig row map", n, KOKKOS_LAMBDA(const ordinal_t i, edge_offset_t& update, const bool final){
        ordinal_t u = permute(i);
        update += x.row_map(u + 1) - x.row_map(u);
        if(final){
            row_map(i + 1) = update;
        }
    });
    char_view_t chars("minitig chars", total_chars);
    Kokkos::parallel_for("write minitig chars", policy(n, Kokkos::AUTO), KOKKOS_LAMBDA(const member& thread){
        ordinal_t i = thread.league_rank();
        ordinal_t u = permute(i);
        edge_offset_t write_to = row_map(i);
        Kokkos::parallel_for(Kokkos::TeamThreadRange(thread, x.row_map(u), x.row_map(u + 1)), [=] (const edge_offset_t j){
            edge_offset_t local_write = write_to + j - x.row_map(u);
            chars(local_write) = x.chars(j);
        });
    });
    minitigs out;
    out.row_map = edge_mirror_t(Kokkos::ViewAllocateWithoutInitializing("minitig row map host"), n + 1);
    Kokkos::deep_copy(out.row_map, row_map);
    out.chars = char_mirror_t(Kokkos::ViewAllocateWithoutInitializing("minitig chars host"), total_chars);
    Kokkos::deep_copy(out.chars, chars);
    out.size = n;
    return out;
}

bucket_minitigs partition_for_output(ordinal_t buckets, minitigs x, edge_offset_t k, vtx_view_t partition, vtx_view_t output_mapping){
    vtx_view_t bucket_counts("buckets", buckets);
    ordinal_t n = x.size;
    Kokkos::parallel_for("count partitions", n, KOKKOS_LAMBDA(const ordinal_t i){
        Kokkos::atomic_increment(&bucket_counts(partition(i)));
    });
    vtx_mirror_t bucket_counts_m = Kokkos::create_mirror(bucket_counts);
    Kokkos::deep_copy(bucket_counts_m, bucket_counts);
    vtx_view_t bucket_offsets("bucket offsets", buckets + 1);
    vtx_mirror_t bucket_offsets_m = Kokkos::create_mirror(bucket_offsets);
    for(ordinal_t i = 0; i < buckets; i++){
        bucket_offsets_m(i + 1) = bucket_offsets_m(i) + bucket_counts_m(i);
        bucket_counts_m(i) = 0;
    }
    Kokkos::deep_copy(bucket_offsets, bucket_offsets_m);
    Kokkos::deep_copy(bucket_counts, bucket_counts_m);
    vtx_view_t bucketed_rows("bucketed rows", n);
    vtx_view_t bucketed_output_map("bucketed output map", n);
    Kokkos::parallel_for("move to partitions", n, KOKKOS_LAMBDA(const ordinal_t i){
        ordinal_t insert = Kokkos::atomic_fetch_add(&bucket_counts(partition(i)), 1) + bucket_offsets(partition(i));
        bucketed_rows(insert) = i;
        bucketed_output_map(insert) = output_mapping(i);
    });
    minitigs y = permute_minitigs(x, bucketed_rows);
    bucket_minitigs out;
    out.part_offsets = bucket_offsets_m;
    out.output_map = bucketed_output_map;
    out.m = y;
    out.buckets = buckets;
    return out;
}

bucket_glues partition_for_output(ordinal_t buckets, graph_type glues, vtx_view_t partition, vtx_view_t output_mapping){
    vtx_view_t bucket_counts("buckets", buckets);
    Kokkos::parallel_for("count partitions", glues.numRows(), KOKKOS_LAMBDA(const ordinal_t i){
        Kokkos::atomic_increment(&bucket_counts(partition(i)));
    });
    vtx_mirror_t bucket_counts_m = Kokkos::create_mirror(bucket_counts);
    Kokkos::deep_copy(bucket_counts_m, bucket_counts);
    vtx_view_t bucket_offsets("bucket offsets", buckets + 1);
    vtx_mirror_t bucket_offsets_m = Kokkos::create_mirror(bucket_offsets);
    for(ordinal_t i = 0; i < buckets; i++){
        bucket_offsets_m(i + 1) = bucket_offsets_m(i) + bucket_counts_m(i);
        bucket_counts_m(i) = 0;
    }
    Kokkos::deep_copy(bucket_offsets, bucket_offsets_m);
    Kokkos::deep_copy(bucket_counts, bucket_counts_m);
    vtx_view_t bucketed_rows("bucketed rows", glues.numRows());
    vtx_view_t bucketed_output_map("bucketed output map", glues.numRows());
    Kokkos::parallel_for("move to partitions", glues.numRows(), KOKKOS_LAMBDA(const ordinal_t i){
        ordinal_t insert = Kokkos::atomic_fetch_add(&bucket_counts(partition(i)), 1) + bucket_offsets(partition(i));
        bucketed_rows(insert) = i;
        bucketed_output_map(insert) = output_mapping(i);
    });
    edge_view_t reordered_glue_row_map("reordered glue row map", glues.numRows() + 1);
    Kokkos::parallel_scan("reorder scan", glues.numRows(), KOKKOS_LAMBDA(const ordinal_t i, edge_offset_t& update, const bool final){
        ordinal_t u = bucketed_rows(i);
        update += glues.row_map(u + 1) - glues.row_map(u);
        if(final) {
            reordered_glue_row_map(i + 1) = update;
        }
    });
    vtx_view_t reordered_glue_entries("reordered glue entries", glues.entries.extent(0));
    Kokkos::parallel_for("reorder entries", policy(glues.numRows(), Kokkos::AUTO), KOKKOS_LAMBDA(const member& thread){
        ordinal_t i = thread.league_rank();
        ordinal_t u = bucketed_rows(i);
        edge_offset_t write_offset = reordered_glue_row_map(i);
        Kokkos::parallel_for(Kokkos::TeamThreadRange(thread, glues.row_map(u), glues.row_map(u + 1)), [=] (const ordinal_t j){
            ordinal_t local_write_offset = write_offset + (j - glues.row_map(u));
            reordered_glue_entries(local_write_offset) = glues.entries(j);
        });
    });
    vtx_view_t buckets_entries_row_map("buckets entries row map", buckets + 1);
    Kokkos::parallel_for("fill bucket entries row map from reordered row map", buckets, KOKKOS_LAMBDA(const ordinal_t i){
        edge_offset_t x = bucket_offsets(i + 1);
        buckets_entries_row_map(i + 1) = reordered_glue_row_map(x);
    });
    vtx_mirror_t buckets_entries_row_map_m = Kokkos::create_mirror(buckets_entries_row_map);
    Kokkos::deep_copy(buckets_entries_row_map_m, buckets_entries_row_map);
    bucket_glues out;
    graph_type reordered_glues(reordered_glue_entries, reordered_glue_row_map);
    out.glues = Kokkos::create_mirror(reordered_glues);
    out.buckets_row_map = bucket_offsets_m;
    out.buckets = buckets;
    out.output_map = bucketed_output_map;
    out.buckets_entries_row_map = buckets_entries_row_map_m;
    return out;
}

template <class T>
T find_l_minimizer(char_view_t& kmers, edge_offset_t k, edge_offset_t l, vtx_view_t lmin_bucket_map, ordinal_t size);

template <>
bucket_kmers find_l_minimizer<bucket_kmers>(char_view_t& kmers, edge_offset_t k, edge_offset_t l, vtx_view_t lmin_bucket_map, ordinal_t size){
    ordinal_t lmin_buckets = 1;
    lmin_buckets <<= 2*l;
    //ordinal_t large_buckets_mask = large_buckets - 1;
    vtx_mirror_t char_map_mirror("char map mirror", 256);
    char_map_mirror('A') = 0;
    char_map_mirror('C') = 1;
    char_map_mirror('G') = 2;
    char_map_mirror('T') = 3;
    vtx_view_t char_map("char map", 256);
    Kokkos::deep_copy(char_map, char_map_mirror);
    vtx_view_t lmin_counter("lmin counter", large_buckets);//lmin_buckets);
    vtx_view_t lmins("lmins", size);
    Kokkos::Timer t;
    Kokkos::parallel_for("calc lmins", size, KOKKOS_LAMBDA(const ordinal_t i){
        lmins(i) = get_lmin(kmers, lmin_bucket_map, char_map, k*i, k, l);
    });
    //printf("Found lmins in %.3f seconds\n", t.seconds());
    t.reset();
    vtx_view_t small_bucket_counts("small buckets", lmin_buckets);
    Kokkos::parallel_for("count lmins", size, KOKKOS_LAMBDA(const ordinal_t i){
        ordinal_t lmin = lmins(i);
        //lmin = lmin & large_buckets_mask;
        //update.the_array[lmin] += 1;
        Kokkos::atomic_increment(&small_bucket_counts(lmin));
    });
    //printf("Counted lmins in %.3f seconds\n", t.seconds());
    t.reset();
    vtx_mirror_t small_bucket_counts_m = Kokkos::create_mirror(small_bucket_counts);
    Kokkos::deep_copy(small_bucket_counts_m, small_bucket_counts);
    vtx_mirror_t buckets_m("buckets mirror", large_buckets + 1);
    vtx_mirror_t small_buckets_m("buckets mirror", lmin_buckets + 1);
    ordinal_t offset = 0;
    for(ordinal_t i = 0; i < large_buckets; i++){
        for(ordinal_t j = i; j < lmin_buckets; j += large_buckets){
            small_buckets_m(j) = offset;
            offset += small_bucket_counts_m(j);
        }
        buckets_m(i + 1) = offset;
    }
    small_buckets_m(lmin_buckets) = offset;
    Kokkos::parallel_for("reset small bucket counts", lmin_buckets, KOKKOS_LAMBDA(const ordinal_t i){
        small_bucket_counts(i) = 0;
    });
    vtx_view_t small_buckets("buckets", lmin_buckets + 1);
    vtx_view_t kmer_ids("buckets", size);
    Kokkos::deep_copy(small_buckets, small_buckets_m);
    Kokkos::parallel_for("partition by lmins", size, KOKKOS_LAMBDA(const ordinal_t i){
        ordinal_t lmin = lmins(i);
        //lmin = lmin & large_buckets_mask;
        ordinal_t insert = small_buckets(lmin) + Kokkos::atomic_fetch_add(&small_bucket_counts(lmin), 1);
        kmer_ids(insert) = i;
    });
    //printf("Partitioned kmers in %.3f seconds\n", t.seconds());
    t.reset();
    char_view_t kmers_partitioned(Kokkos::ViewAllocateWithoutInitializing("kmers partitioned"), size*k);
    Kokkos::View<const char*, Kokkos::MemoryTraits<Kokkos::RandomAccess>> kmers_read = kmers;
    if(typeid(Kokkos::DefaultExecutionSpace::memory_space) != typeid(Kokkos::HostSpace)){
        Kokkos::parallel_for("write kmers device", policy(size, 1, 32), KOKKOS_LAMBDA(const member& thread){
            ordinal_t write_id = thread.league_rank();
            ordinal_t i = kmer_ids(write_id);
            edge_offset_t write_idx = k*write_id;
            edge_offset_t start = k*i;
            edge_offset_t end = start + k;
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(thread, start, end), [=](const edge_offset_t j){
                kmers_partitioned(write_idx + (j - start)) = kmers_read(j);
            });
        });
    } else {
        Kokkos::parallel_for("write kmers host", size, KOKKOS_LAMBDA(const ordinal_t write_id){
            ordinal_t i = kmer_ids(write_id);
            edge_offset_t write_idx = k*write_id;
            edge_offset_t start = k*i;
            edge_offset_t end = start + k;
            for(edge_offset_t j = start; j < end; j++){
                kmers_partitioned(write_idx + (j - start)) = kmers_read(j);
            }
        });
    }
    //printf("Wrote partitioned kmers in %.3f seconds\n", t.seconds());
    t.reset();
    bucket_kmers output;
    output.kmers = Kokkos::create_mirror_view(kmers_partitioned);
    Kokkos::deep_copy(output.kmers, kmers_partitioned);
    output.buckets_row_map = buckets_m;
    output.buckets = large_buckets;
    output.size = buckets_m(large_buckets);
    return output;
}


template <>
bucket_kpmers find_l_minimizer<bucket_kpmers>(char_view_t& kmers, edge_offset_t k, edge_offset_t l, vtx_view_t lmin_bucket_map, ordinal_t size){
    ordinal_t lmin_buckets = 1;
    lmin_buckets <<= 2*l;
    ordinal_t large_buckets_mask = large_buckets - 1;
    vtx_mirror_t char_map_mirror("char map mirror", 256);
    char_map_mirror('A') = 0;
    char_map_mirror('C') = 1;
    char_map_mirror('G') = 2;
    char_map_mirror('T') = 3;
    vtx_view_t char_map("char map", 256);
    Kokkos::deep_copy(char_map, char_map_mirror);
    vtx_view_t lmin_counter("lmin counter", large_buckets);//lmin_buckets);
    vtx_view_t out_lmins("lmins", size);
    vtx_view_t in_lmins("lmins", size);
    Kokkos::Timer t;
    Kokkos::parallel_for("calc lmins", size, KOKKOS_LAMBDA(const ordinal_t i){
        get_double_lmin(kmers, lmin_bucket_map, char_map, k*i, k, l, out_lmins(i), in_lmins(i));
    });
    //printf("Found lmins in %.3f seconds\n", t.seconds());
    t.reset();
    vtx_view_t cross_buckets_count("buckets", large_buckets*large_buckets);
    vtx_view_t small_bucket_counts("small buckets", lmin_buckets);
    Kokkos::parallel_for("count lmins", size, KOKKOS_LAMBDA(const ordinal_t i){
        ordinal_t lmin_out = out_lmins(i);
        lmin_out = lmin_out & large_buckets_mask;
        ordinal_t lmin_in = in_lmins(i);
        lmin_in = lmin_in & large_buckets_mask;
        Kokkos::atomic_increment(&small_bucket_counts(out_lmins(i)));
        //ensure they are in the same large bucket
        if(lmin_in != lmin_out){
            Kokkos::atomic_increment(&cross_buckets_count(lmin_out*large_buckets + lmin_in));
            Kokkos::atomic_increment(&cross_buckets_count(lmin_in*large_buckets + lmin_out));
        }
    });
    //printf("Counted lmins in %.3f seconds\n", t.seconds());
    t.reset();
    vtx_mirror_t small_bucket_counts_m = Kokkos::create_mirror(small_bucket_counts);
    Kokkos::deep_copy(small_bucket_counts_m, small_bucket_counts);
    vtx_mirror_t buckets_m("buckets mirror", large_buckets + 1);
    vtx_mirror_t small_buckets_m("buckets mirror", lmin_buckets + 1);
    ordinal_t offset = 0;
    for(ordinal_t i = 0; i < large_buckets; i++){
        for(ordinal_t j = i; j < lmin_buckets; j += large_buckets){
            small_buckets_m(j) = offset;
            offset += small_bucket_counts_m(j);
        }
        buckets_m(i + 1) = offset;
    }
    small_buckets_m(lmin_buckets) = offset;
    Kokkos::parallel_for("reset small bucket counts", lmin_buckets, KOKKOS_LAMBDA(const ordinal_t i){
        small_bucket_counts(i) = 0;
    });
    vtx_view_t small_buckets("small buckets", lmin_buckets + 1);
    vtx_view_t kmer_writes("kmer writes", buckets_m(large_buckets));
    vtx_view_t buckets("buckets device", large_buckets + 1);
    Kokkos::deep_copy(buckets, buckets_m);
    Kokkos::deep_copy(small_buckets, small_buckets_m);
    vtx_view_t cross_buckets("cross buckets", large_buckets*large_buckets + 1);
    Kokkos::parallel_scan("prefix sum crosscut", large_buckets*large_buckets, KOKKOS_LAMBDA(const ordinal_t i, ordinal_t& update, const bool final){
        update += cross_buckets_count(i);
        if(final){
            cross_buckets(i + 1) = update;
            cross_buckets_count(i) = 0;
        }
    });
    vtx_mirror_t crosscut_buckets_m = Kokkos::create_mirror(cross_buckets);
    Kokkos::deep_copy(crosscut_buckets_m, cross_buckets);
    ordinal_t total_crosscut = crosscut_buckets_m(large_buckets*large_buckets);
    vtx_view_t cross_writes("cross writes", total_crosscut);
    vtx_view_t cross_ids("cross ids", total_crosscut);
    Kokkos::parallel_for("init cross ids", total_crosscut, KOKKOS_LAMBDA(const ordinal_t i){
        cross_ids(i) = ORD_MAX;
    });
    Kokkos::parallel_for("partition by lmins", size, KOKKOS_LAMBDA(const ordinal_t i){
        ordinal_t lmin_out = out_lmins(i);
        ordinal_t insert = small_buckets(lmin_out) + Kokkos::atomic_fetch_add(&small_bucket_counts(lmin_out), 1);
        kmer_writes(insert) = i;
        lmin_out = lmin_out & large_buckets_mask;
        ordinal_t lmin_in = in_lmins(i);
        lmin_in = lmin_in & large_buckets_mask;
        if(lmin_in != lmin_out){
            ordinal_t out_bucket = lmin_out*large_buckets + lmin_in;
            ordinal_t in_bucket = lmin_in*large_buckets + lmin_out;
            ordinal_t less_bucket = in_bucket < out_bucket ? in_bucket : out_bucket;
            ordinal_t offset = Kokkos::atomic_fetch_add(&cross_buckets_count(less_bucket), 1);
            ordinal_t cross_insert = cross_buckets(out_bucket) + offset;
            cross_writes(cross_insert) = i;
            cross_ids(cross_insert) = insert - buckets(lmin_out);
            cross_insert = cross_buckets(in_bucket) + offset;
            cross_writes(cross_insert) = i;
        }
    });
    //printf("Partitioned kmers in %.3f seconds\n", t.seconds());
    t.reset();
    edge_offset_t k_pad = ((k + 3) / 4) * 4;
    edge_offset_t comp_size = k_pad / 4;
    char_view_t kmer_compress(Kokkos::ViewAllocateWithoutInitializing("kmers compressed nonpart"), buckets_m(large_buckets) * comp_size);
    Kokkos::parallel_for("compress kmers", buckets_m(large_buckets), KOKKOS_LAMBDA(const edge_offset_t j){
        char byte = 0;
        for(edge_offset_t x = 0; x < k_pad; x++){
            byte <<= 2;
            if((x & 3) == 0) byte = 0;
            if(x < k) byte = byte | char_map(kmers(j*k + x));
            if((x & 3) == 3) kmer_compress(j*comp_size + (x / 4)) = byte;
        }
    });
    char_view_t kmers_partitioned(Kokkos::ViewAllocateWithoutInitializing("kmers partitioned"), buckets_m(large_buckets) * comp_size);
    if(typeid(Kokkos::DefaultExecutionSpace::memory_space) != typeid(Kokkos::HostSpace)){
        Kokkos::parallel_for("write kmers", policy(buckets_m(large_buckets), 32), KOKKOS_LAMBDA(const member& thread){
            ordinal_t write_id = thread.league_rank();
            ordinal_t i = kmer_writes(write_id);
            edge_offset_t write_idx = comp_size*write_id;
            edge_offset_t start = comp_size*i;
            edge_offset_t end = start + comp_size;
            Kokkos::parallel_for(Kokkos::TeamThreadRange(thread, start, end), [=](const edge_offset_t j){
                kmers_partitioned(write_idx + (j - start)) = kmer_compress(j);
            });
        });
    } else {
        Kokkos::parallel_for("write kmers host", buckets_m(large_buckets), KOKKOS_LAMBDA(const ordinal_t write_id){
            ordinal_t i = kmer_writes(write_id);
            edge_offset_t write_idx = comp_size*write_id;
            edge_offset_t start = comp_size*i;
            edge_offset_t end = start + comp_size;
            for(edge_offset_t j = start; j < end; j++){
                kmers_partitioned(write_idx + (j - start)) = kmer_compress(j);
            }
        });
    }
    char_view_t crosscut_partitioned(Kokkos::ViewAllocateWithoutInitializing("crosscut partitioned"), total_crosscut * k);
    if(typeid(Kokkos::DefaultExecutionSpace::memory_space) != typeid(Kokkos::HostSpace)){
        Kokkos::parallel_for("write kmers", policy(total_crosscut, 32), KOKKOS_LAMBDA(const member& thread){
            ordinal_t write_id = thread.league_rank();
            ordinal_t read_id = cross_writes(write_id);
            edge_offset_t write_idx = k*write_id;
            edge_offset_t start = k*read_id;
            edge_offset_t end = start + k;
            Kokkos::parallel_for(Kokkos::TeamThreadRange(thread, start, end), [=](const edge_offset_t j){
                crosscut_partitioned(write_idx + (j - start)) = kmers(j);
            });
        });
    } else {
        Kokkos::parallel_for("write kmers host", total_crosscut, KOKKOS_LAMBDA(const ordinal_t write_id){
            ordinal_t read_id = cross_writes(write_id);
            edge_offset_t write_idx = k*write_id;
            edge_offset_t start = k*read_id;
            edge_offset_t end = start + k;
            for(edge_offset_t j = start; j < end; j++){
                crosscut_partitioned(write_idx + (j - start)) = kmers(j);
            }
        });
    }
    //printf("Wrote partitioned kmers in %.3f seconds\n", t.seconds());
    t.reset();
    {
        //prevent lambdas from trying to capture output.kmers (a std::vector)
        bucket_kpmers output;
        for(int i = 0; i < large_buckets; i++){
            ordinal_t kmers_in_bucket = buckets_m(i + 1) - buckets_m(i);
            char_view_t kmer_bucket_dev = Kokkos::subview(kmers_partitioned, std::make_pair(comp_size*buckets_m(i), comp_size*buckets_m(i+1)));
            char_mirror_t kmer_bucket = Kokkos::create_mirror_view(kmer_bucket_dev);
            Kokkos::deep_copy(kmer_bucket, kmer_bucket_dev);
            output.kmers.push_back(kmer_bucket);
        }
        output.buckets = large_buckets;
        output.size = buckets_m(large_buckets);
        output.crosscut = Kokkos::create_mirror_view(crosscut_partitioned);
        Kokkos::deep_copy(output.crosscut, crosscut_partitioned);
        output.cross_ids = Kokkos::create_mirror_view(cross_ids);
        Kokkos::deep_copy(output.cross_ids, cross_ids);
        output.crosscut_row_map = crosscut_buckets_m;
        output.crosscut_buckets = large_buckets*large_buckets;
        output.crosscut_size = crosscut_buckets_m(large_buckets*large_buckets);
        return output;
    }
}

}
