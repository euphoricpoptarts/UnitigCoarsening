#include "definitions_kokkos.h"

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
void get_double_lmin(const char_view_t chars, const vtx_view_t lmin_map, const vtx_view_t char_map, edge_offset_t offset, edge_offset_t k, edge_offset_t l, ordinal_t& l1, ordinal_t& l2){
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


struct bucket_kmers {
    char_view_t kmers;
    vtx_mirror_t buckets_row_map;
    ordinal_t buckets;
    ordinal_t size;
};

struct bucket_kpmers {
    char_view_t kmers;
    vtx_mirror_t buckets_row_map;
    ordinal_t buckets;
    ordinal_t size;
    char_view_t crosscut;
    vtx_mirror_t crosscut_row_map;
    ordinal_t crosscut_buckets;
    ordinal_t crosscut_size;
};

bucket_kmers find_l_minimizer(char_view_t& kmers, edge_offset_t k, edge_offset_t l, vtx_view_t lmin_bucket_map, ordinal_t size){
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
    vtx_view_t lmins("lmins", size);
    Kokkos::Timer t;
    Kokkos::parallel_for("calc lmins", size, KOKKOS_LAMBDA(const ordinal_t i){
        lmins(i) = get_lmin(kmers, lmin_bucket_map, char_map, k*i, k, l);
    });
    printf("Found lmins in %.3f seconds\n", t.seconds());
    t.reset();
    reduce_t r;
    Kokkos::parallel_reduce("count lmins", size, KOKKOS_LAMBDA(const ordinal_t i, reduce_t& update){
        ordinal_t lmin = lmins(i);
        lmin = lmin & large_buckets_mask;
        update.the_array[lmin] += 1;
    }, Kokkos::Sum<reduce_t>(r));
    printf("Counted lmins in %.3f seconds\n", t.seconds());
    t.reset();
    vtx_mirror_t buckets_m("buckets mirror", large_buckets + 1);
    for(ordinal_t i = 0; i < large_buckets; i++){
        //printf("bucket %u contains %u\n", i, r.the_array[i]);
        buckets_m(i + 1) = buckets_m(i) + r.the_array[i];
    }
    vtx_view_t buckets("buckets", large_buckets + 1);
    vtx_view_t buckets_count("buckets", large_buckets);
    vtx_view_t kmer_ids("buckets", size);
    Kokkos::deep_copy(buckets, buckets_m);
    Kokkos::parallel_for("partition by lmins", size, KOKKOS_LAMBDA(const ordinal_t i){
        ordinal_t lmin = lmins(i);
        lmin = lmin & large_buckets_mask;
        ordinal_t insert = buckets(lmin) + Kokkos::atomic_fetch_add(&buckets_count(lmin), 1);
        kmer_ids(insert) = i;
    });
    printf("Partitioned kmers in %.3f seconds\n", t.seconds());
    t.reset();
    char_view_t kmers_partitioned(Kokkos::ViewAllocateWithoutInitializing("kmers partitioned"), kmers.extent(0));
    Kokkos::View<const char*, Kokkos::MemoryTraits<Kokkos::RandomAccess>> kmers_read = kmers;
    Kokkos::parallel_for("write kmers", policy(size, 32), KOKKOS_LAMBDA(const member& thread){
        ordinal_t write_id = thread.league_rank();
        ordinal_t i = kmer_ids(write_id);
        edge_offset_t write_idx = k*write_id;
        edge_offset_t start = k*i;
        edge_offset_t end = start + k;
        Kokkos::parallel_for(Kokkos::TeamThreadRange(thread, start, end), [=](const edge_offset_t j){
            kmers_partitioned(write_idx + (j - start)) = kmers_read(j);
        });
    });
    printf("Wrote partitioned kmers in %.3f seconds\n", t.seconds());
    t.reset();
    bucket_kmers output;
    output.kmers = kmers_partitioned;
    output.buckets_row_map = buckets_m;
    output.buckets = large_buckets;
    output.size = buckets_m(large_buckets);
    return output;
}

bucket_kpmers find_l_minimizer_edge(char_view_t& kmers, edge_offset_t k, edge_offset_t l, vtx_view_t lmin_bucket_map, ordinal_t size){
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
    reduce_t r;
    vtx_view_t cross_buckets_count("buckets", large_buckets*large_buckets);
    Kokkos::parallel_reduce("count lmins", size, KOKKOS_LAMBDA(const ordinal_t i, reduce_t& update){
        ordinal_t lmin_out = out_lmins(i);
        lmin_out = lmin_out & large_buckets_mask;
        ordinal_t lmin_in = in_lmins(i);
        lmin_in = lmin_in & large_buckets_mask;
        if(lmin_in == lmin_out){
            update.the_array[lmin_out] += 1;
        } else {
            Kokkos::atomic_increment(&cross_buckets_count(lmin_out*large_buckets + lmin_in));
            Kokkos::atomic_increment(&cross_buckets_count(lmin_in*large_buckets + lmin_out));
        }
    }, Kokkos::Sum<reduce_t>(r));
    //printf("Counted lmins in %.3f seconds\n", t.seconds());
    t.reset();
    vtx_mirror_t buckets_m("buckets mirror", large_buckets + 1);
    for(ordinal_t i = 0; i < large_buckets; i++){
        //printf("bucket %u contains %u\n", i, r.the_array[i]);
        buckets_m(i + 1) = buckets_m(i) + r.the_array[i];
    }
    vtx_view_t buckets("buckets", large_buckets + 1);
    vtx_view_t buckets_count("buckets", large_buckets);
    vtx_view_t kmer_writes("buckets", buckets_m(large_buckets));
    Kokkos::deep_copy(buckets, buckets_m);
    Kokkos::parallel_for("partition by lmins", size, KOKKOS_LAMBDA(const ordinal_t i){
        ordinal_t lmin_out = out_lmins(i);
        lmin_out = lmin_out & large_buckets_mask;
        ordinal_t lmin_in = in_lmins(i);
        lmin_in = lmin_in & large_buckets_mask;
        if(lmin_in == lmin_out){
            ordinal_t insert = buckets(lmin_out) + Kokkos::atomic_fetch_add(&buckets_count(lmin_out), 1);
            kmer_writes(insert) = i;
        }
    });
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
    Kokkos::parallel_for("partition by lmins", size, KOKKOS_LAMBDA(const ordinal_t i){
        ordinal_t lmin_out = out_lmins(i);
        lmin_out = lmin_out & large_buckets_mask;
        ordinal_t lmin_in = in_lmins(i);
        lmin_in = lmin_in & large_buckets_mask;
        if(lmin_in != lmin_out){
            ordinal_t out_bucket = lmin_out*large_buckets + lmin_in;
            ordinal_t in_bucket = lmin_in*large_buckets + lmin_out;
            ordinal_t less_bucket = in_bucket < out_bucket ? in_bucket : out_bucket;
            ordinal_t offset = Kokkos::atomic_fetch_add(&cross_buckets_count(less_bucket), 1);
            ordinal_t insert = cross_buckets(out_bucket) + offset;
            cross_writes(insert) = i;
            insert = cross_buckets(in_bucket) + offset;
            cross_writes(insert) = i;
        }
    });
    //printf("Partitioned kmers in %.3f seconds\n", t.seconds());
    t.reset();
    char_view_t kmers_partitioned(Kokkos::ViewAllocateWithoutInitializing("kmers partitioned"), buckets_m(large_buckets) * k);
    Kokkos::parallel_for("write kmers", policy(buckets_m(large_buckets), 32), KOKKOS_LAMBDA(const member& thread){
        ordinal_t write_id = thread.league_rank();
        ordinal_t i = kmer_writes(write_id);
        edge_offset_t write_idx = k*write_id;
        edge_offset_t start = k*i;
        edge_offset_t end = start + k;
        Kokkos::parallel_for(Kokkos::TeamThreadRange(thread, start, end), [=](const edge_offset_t j){
            kmers_partitioned(write_idx + (j - start)) = kmers(j);
        });
    });
    char_view_t crosscut_partitioned(Kokkos::ViewAllocateWithoutInitializing("crosscut partitioned"), total_crosscut * k);
    Kokkos::parallel_for("write kmers", policy(total_crosscut, 32), KOKKOS_LAMBDA(const member& thread){
        ordinal_t write_id = thread.league_rank();
        ordinal_t i = cross_writes(write_id);
        edge_offset_t write_idx = k*write_id;
        edge_offset_t start = k*i;
        edge_offset_t end = start + k;
        Kokkos::parallel_for(Kokkos::TeamThreadRange(thread, start, end), [=](const edge_offset_t j){
            crosscut_partitioned(write_idx + (j - start)) = kmers(j);
        });
    });
    printf("Wrote partitioned kmers in %.3f seconds\n", t.seconds());
    t.reset();
    bucket_kpmers output;
    output.kmers = kmers_partitioned;
    output.buckets_row_map = buckets_m;
    output.buckets = large_buckets;
    output.size = buckets_m(large_buckets);
    output.crosscut = crosscut_partitioned;
    output.crosscut_row_map = crosscut_buckets_m;
    output.crosscut_buckets = large_buckets*large_buckets;
    output.crosscut_size = crosscut_buckets_m(large_buckets*large_buckets);
    return output;
}

}
