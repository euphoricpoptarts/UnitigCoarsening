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
char get_char2(const hash_vt& source, edge_offset_t offset, edge_offset_t char_id){
    offset += (char_id / 16);
    char_id = char_id % 16;
    uint32_t bytes = source(offset);
    char byte = bytes >> (2*char_id);
    byte = byte & 3;
    return byte;
}

KOKKOS_INLINE_FUNCTION
void write_char(comp_vt dest, edge_offset_t offset, edge_offset_t char_id, char val){
    offset += (char_id / 16);
    char_id = char_id % 16;
    uint32_t bytes = val;
    bytes = bytes << (2*char_id);
    Kokkos::atomic_or(&dest(offset), bytes);
}

KOKKOS_INLINE_FUNCTION
void get_double_lmin(const hash_vt& chars, const lmin_vt& lmin_map, edge_offset_t offset, edge_offset_t k, edge_offset_t l, ordinal_t& l1, ordinal_t& l2){
    const ordinal_t lmer_mask = (1 << (2*l)) - 1;
    ordinal_t lmer_id = 0;
    l1 = ORD_MAX;
    l2 = ORD_MAX;
    for(edge_offset_t i = 0; i < k; i++)
    {
        ordinal_t c_val = get_char2(chars, offset, i);
        //emplace c_val into the least significant two bits and trim the most significant bits
        lmer_id <<= 2;
        lmer_id ^= c_val;
        lmer_id &= lmer_mask;

        //check if we have a full lmer
        if(i + 1 >= l){
            ordinal_t lx = lmin_map(lmer_id);
            //check if we are in the k-1 prefix
            if(i + 1 < k && l1 > lx){
                l1 = lx;
            }
            //check if we are in the k-1 suffix
            if(i + 1 >= l + 1 && l2 > lx){
                l2 = lx;
            }
        }
    }
}

KOKKOS_INLINE_FUNCTION
ordinal_t get_lmin(const char_view_t chars, const lmin_vt lmin_map, const vtx_view_t char_map, edge_offset_t offset, edge_offset_t k, edge_offset_t l){
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
    vtx_mirror_t lengths;
    edge_mirror_t row_map;
    comp_mt chars;
    ordinal_t size;
};

struct minitigs_d {
    vtx_view_t lengths;
    edge_view_t row_map;
    comp_vt chars;
    ordinal_t size;
};

struct bucket_kpmers {
    std::list<comp_mt> kmers;
    ordinal_t buckets;
    ordinal_t size;
    vtx_mirror_t cross_lend, cross_borrow;
    vtx_mirror_t crosscut_row_map;
    ordinal_t crosscut_buckets;
    ordinal_t crosscut_size;
};

struct bucket_minitigs {
    minitigs m;
    vtx_mirror_t part_offsets;
    ordinal_t buckets;
    vtx_view_t output_map;
};

minitigs_d move_to_device(minitigs x){
    minitigs_d y;
    y.lengths = vtx_view_t("lengths", x.lengths.extent(0));
    Kokkos::deep_copy(y.lengths, x.lengths);
    y.row_map = edge_view_t("row map", x.row_map.extent(0));
    Kokkos::deep_copy(y.row_map, x.row_map);
    y.chars = comp_vt("chars", x.chars.extent(0));
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

minitigs generate_minitigs(graph_type glues, const hash_vt kmers, edge_offset_t k, edge_offset_t comp_size){
    ordinal_t n = glues.numRows();
    vtx_view_t lengths("minitig lengths", n);
    Kokkos::parallel_for("write minitig lengths", n, KOKKOS_LAMBDA(const ordinal_t i){
        lengths(i) = k - 1 + glues.row_map(i + 1) - glues.row_map(i);
    });
    edge_view_t row_map("minitig row map", n + 1);
    edge_offset_t total_chars = 0;
    Kokkos::parallel_scan("write minitig row map", n, KOKKOS_LAMBDA(const ordinal_t i, edge_offset_t& update, const bool final){
        ordinal_t padded = ((lengths(i) + 15) / 16);
        update += padded;
        if(final){
            row_map(i + 1) = update;
        }
    }, total_chars);
    comp_vt chars("minitig chars", total_chars);
    Kokkos::parallel_for("write minitig chars", policy(n, Kokkos::AUTO), KOKKOS_LAMBDA(const member& thread){
        const ordinal_t i = thread.league_rank();
        edge_offset_t write_to = row_map(i);
        Kokkos::parallel_for(Kokkos::TeamThreadRange(thread, glues.row_map(i), glues.row_map(i + 1)), [=] (const edge_offset_t j){
            ordinal_t u = glues.entries(j);
            edge_offset_t local_write = j - glues.row_map(i);
            if(j + 1 < glues.row_map(i + 1)){
                char x = get_char2(kmers, u*comp_size, 0);
                write_char(chars, write_to, local_write, x);
            } else {
                for(edge_offset_t l = 0; l < k; l++){
                    char x = get_char2(kmers, u*comp_size, l);
                    write_char(chars, write_to, local_write++, x);
                }
            }
        });
    });
    minitigs out;
    out.lengths = vtx_mirror_t(Kokkos::ViewAllocateWithoutInitializing("minitig lengths"), n);
    Kokkos::deep_copy(out.lengths, lengths);
    out.row_map = edge_mirror_t(Kokkos::ViewAllocateWithoutInitializing("minitig row map host"), n + 1);
    Kokkos::deep_copy(out.row_map, row_map);
    out.chars = comp_mt(Kokkos::ViewAllocateWithoutInitializing("minitig chars host"), total_chars);
    Kokkos::deep_copy(out.chars, chars);
    out.size = n;
    return out;
}

minitigs permute_minitigs(minitigs x_host, vtx_view_t permute){
    minitigs_d x = move_to_device(x_host);
    ordinal_t n = x.size;
    vtx_view_t lengths("minitig lengths", n);
    edge_view_t row_map("minitig row map", n + 1);
    edge_offset_t total_chars = x.chars.extent(0);
    Kokkos::parallel_for("permute lengths", n, KOKKOS_LAMBDA(const ordinal_t i){
        lengths(i) = x.lengths(permute(i));
    });
    Kokkos::parallel_scan("permut minitig row map", n, KOKKOS_LAMBDA(const ordinal_t i, edge_offset_t& update, const bool final){
        ordinal_t u = permute(i);
        update += x.row_map(u + 1) - x.row_map(u);
        if(final){
            row_map(i + 1) = update;
        }
    });
    comp_vt chars("minitig chars", total_chars);
    Kokkos::parallel_for("permute minitig chars", policy(n, Kokkos::AUTO), KOKKOS_LAMBDA(const member& thread){
        ordinal_t i = thread.league_rank();
        ordinal_t u = permute(i);
        edge_offset_t write_to = row_map(i);
        Kokkos::parallel_for(Kokkos::TeamThreadRange(thread, x.row_map(u), x.row_map(u + 1)), [=] (const edge_offset_t j){
            edge_offset_t local_write = write_to + j - x.row_map(u);
            chars(local_write) = x.chars(j);
        });
    });
    minitigs out;
    out.lengths = vtx_mirror_t(Kokkos::ViewAllocateWithoutInitializing("minitig lengths"), n);
    Kokkos::deep_copy(out.lengths, lengths);
    out.row_map = edge_mirror_t(Kokkos::ViewAllocateWithoutInitializing("minitig row map host"), n + 1);
    Kokkos::deep_copy(out.row_map, row_map);
    out.chars = comp_mt(Kokkos::ViewAllocateWithoutInitializing("minitig chars host"), total_chars);
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

template <class T>
T find_l_minimizer(const hash_vt& kmers, edge_offset_t k, edge_offset_t l, const lmin_vt lmin_bucket_map, ordinal_t size);

template <>
bucket_kpmers find_l_minimizer<bucket_kpmers>(const hash_vt& kmer_compress, edge_offset_t k, edge_offset_t l, const lmin_vt lmin_bucket_map, ordinal_t size){
    ordinal_t lmin_buckets = 1;
    lmin_buckets <<= 2*l;
    ordinal_t large_buckets_mask = large_buckets - 1;
    vtx_view_t lmin_counter("lmin counter", large_buckets);//lmin_buckets);
    vtx_view_t out_lmins("lmins", size);
    vtx_view_t in_lmins("lmins", size);
    edge_offset_t k_pad = ((k + 15) / 16) * 16;
    edge_offset_t comp_size = k_pad / 16;
    Kokkos::Timer t;
    Kokkos::parallel_for("calc lmins", size, KOKKOS_LAMBDA(const ordinal_t i){
        get_double_lmin(kmer_compress, lmin_bucket_map, comp_size*i, k, l, out_lmins(i), in_lmins(i));
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
            Kokkos::atomic_increment(&small_bucket_counts(in_lmins(i)));
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
    vtx_view_t cross_lend("cross lends", total_crosscut);
    vtx_view_t cross_borrow("cross borrow", total_crosscut);
    Kokkos::deep_copy(cross_lend, ORD_MAX);
    Kokkos::deep_copy(cross_borrow, ORD_MAX);
    Kokkos::parallel_for("partition by lmins", size, KOKKOS_LAMBDA(const ordinal_t i){
        ordinal_t lmin_out = out_lmins(i);
        ordinal_t insert = small_buckets(lmin_out) + Kokkos::atomic_fetch_add(&small_bucket_counts(lmin_out), 1);
        kmer_writes(insert) = i;
        lmin_out = lmin_out & large_buckets_mask;
        ordinal_t lmin_in = in_lmins(i);
        lmin_in = lmin_in & large_buckets_mask;
        if(lmin_in != lmin_out){
            ordinal_t local_id = insert - buckets(lmin_out);
            ordinal_t write_lmin = in_lmins(i);
            insert = small_buckets(write_lmin) + Kokkos::atomic_fetch_add(&small_bucket_counts(write_lmin), 1);
            kmer_writes(insert) = i;
            ordinal_t out_bucket = lmin_out*large_buckets + lmin_in;
            ordinal_t in_bucket = lmin_in*large_buckets + lmin_out;
            ordinal_t less_bucket = in_bucket < out_bucket ? in_bucket : out_bucket;
            ordinal_t offset = Kokkos::atomic_fetch_add(&cross_buckets_count(less_bucket), 1);
            ordinal_t cross_insert = cross_buckets(out_bucket) + offset;
            //the lmin_out -> lmin_in crossbucket tracks the id of the out-side kmer of this edge
            //and will eventually know the coarse id
            //the lmin_in -> lmin_out crossbucket will be written with the in-side kmer of the edge
            //when it is discovered later
            cross_lend(cross_insert) = local_id;
            cross_insert = cross_buckets(in_bucket) + offset;
            local_id = insert - buckets(lmin_in);
            cross_borrow(cross_insert) = local_id;
        }
    });
    t.reset();
    comp_vt kmers_partitioned(Kokkos::ViewAllocateWithoutInitializing("kmers partitioned"), buckets_m(large_buckets) * comp_size);
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
    //printf("Wrote partitioned kmers in %.3f seconds\n", t.seconds());
    t.reset();
    {
        //prevent lambdas from trying to capture output.kmers (a std::vector)
        bucket_kpmers output;
        comp_mt partition_copy = Kokkos::create_mirror_view(kmers_partitioned);
        Kokkos::deep_copy(partition_copy, kmers_partitioned);
        for(int i = 0; i < large_buckets; i++){
            ordinal_t kmers_in_bucket = buckets_m(i + 1) - buckets_m(i);
            comp_mt kmer_bucket_subview = Kokkos::subview(partition_copy, std::make_pair(comp_size*buckets_m(i), comp_size*buckets_m(i+1)));
            comp_mt kmer_bucket(Kokkos::ViewAllocateWithoutInitializing("kmer bucket"), kmers_in_bucket*comp_size);
            Kokkos::deep_copy(kmer_bucket, kmer_bucket_subview);
            output.kmers.push_back(kmer_bucket);
        }
        output.buckets = large_buckets;
        output.size = buckets_m(large_buckets);
        output.cross_lend = Kokkos::create_mirror_view(cross_lend);
        Kokkos::deep_copy(output.cross_lend, cross_lend);
        output.cross_borrow = Kokkos::create_mirror_view(cross_borrow);
        Kokkos::deep_copy(output.cross_borrow, cross_borrow);
        output.crosscut_row_map = crosscut_buckets_m;
        output.crosscut_buckets = large_buckets*large_buckets;
        output.crosscut_size = crosscut_buckets_m(large_buckets*large_buckets);
        return output;
    }
}

}
