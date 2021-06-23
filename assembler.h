#include "definitions_kokkos.h"

namespace unitig_compact {

//fnv hash algorithm
//we going deep
KOKKOS_INLINE_FUNCTION
uint32_t fnv(const char_view_t chars, edge_offset_t offset, edge_offset_t k){
    uint32_t hash = 2166136261U;
    for(edge_offset_t i = offset; i < offset + k; i++)
    {
        hash = hash ^ (chars(i)); // xor next byte into the bottom of the hash
        hash = hash * 16777619; // Multiply by prime number found to work well
    }
    return hash;
}

//gotta do everything myself
KOKKOS_INLINE_FUNCTION
bool cmp(const char_view_t s1_chars, const char_view_t s2_chars, const edge_offset_t s1_offset, const edge_offset_t s2_offset, const edge_offset_t k){
    for(edge_offset_t i = 0; i < k; i++){
        if(s1_chars(s1_offset + i) != s2_chars(s2_offset + i)){
            return false;
        }
    }
    return true;
}

//check if vtx found in edge_chars at offset exists in vtx_chars using a hashmap
KOKKOS_INLINE_FUNCTION
edge_offset_t find_vtx_from_edge(const char_view_t vtx_chars, const edge_view_t vtx_map, edge_offset_t offset, edge_offset_t k, edge_offset_t size){
    size_t hash = fnv(vtx_chars, offset, k - 1);
    size_t hash_cast = vtx_map.extent(0) - 1;
    hash = hash & hash_cast;
    while(vtx_map(hash) != EDGE_MAX){
        edge_offset_t addr = vtx_map(hash);
        bool nullified = false;
        if(addr >= size){
            addr -= size;
            nullified = true;
        }
        edge_offset_t hash_offset = addr*k;
        if(cmp(vtx_chars, vtx_chars, offset, hash_offset, k - 1)){
            if(!nullified){
                return addr;
            } else {
                return addr + size;
            }
        }
        hash = (hash + 1) & hash_cast;
    }
    return EDGE_MAX;
}

edge_view_t generate_hashmap(char_view_t kmers, edge_offset_t k, edge_offset_t size){
    size_t hashmap_size = 1;
    size_t preferred_size = size;
    while(hashmap_size < preferred_size) hashmap_size <<= 1;
    edge_view_t out("hashmap", hashmap_size);
    Kokkos::parallel_for("init hashmap", hashmap_size, KOKKOS_LAMBDA(const ordinal_t i){
        out(i) = EDGE_MAX;
    });
    size_t hash_cast = hashmap_size - 1;
    Kokkos::parallel_for("fill hashmap", size, KOKKOS_LAMBDA(const edge_offset_t i){
        size_t hash = fnv(kmers, k*i, k - 1);
        hash = hash & hash_cast;
        bool success = Kokkos::atomic_compare_exchange_strong(&out(hash), EDGE_MAX, i);
        //linear probing
        while(!success){
            edge_offset_t written = out(hash);
            if(written >= size){
                written = written - size;
                if(cmp(kmers, kmers, k*i, k*written, k - 1)){
                    //hash value matches k-1 mer
                    //but has been nullified
                    //do nothing
                    break;
                }
            } else if(cmp(kmers, kmers, k*i, k*written, k - 1)){
                //hash value matches k-1 mer
                //nullify it
                Kokkos::atomic_compare_exchange_strong(&out(hash), written, i + size);
                break;
            } 
            hash = (hash + 1) & hash_cast;
            success = Kokkos::atomic_compare_exchange_strong(&out(hash), EDGE_MAX, i);
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

vtx_view_t assemble_pruned_graph(char_view_t kmers, edge_view_t vtx_map, edge_offset_t k){
    ordinal_t n = kmers.extent(0) / k;
    //both the in and out edge counts for each vertex are packed into one char
    char_view_t edge_count("edge count", n);
    edge_view_t in("in vertex", n);
    Kokkos::parallel_for("translate edges", n, KOKKOS_LAMBDA(const ordinal_t i){
        edge_offset_t v = find_vtx_from_edge(kmers, vtx_map, i*k + 1, k, n);
        in(i) = v;
    });
    Kokkos::parallel_for("count edges", n, KOKKOS_LAMBDA(const ordinal_t i){
        edge_offset_t v = in(i);
        if(v != EDGE_MAX){
            if(v > n){
                v = v - n;
            }
            Kokkos::atomic_add(&edge_count(v), (char)1);
        }
    });
    vtx_view_t g("pruned out entries", n);
    Kokkos::parallel_for("init g", n, KOKKOS_LAMBDA(const ordinal_t i){
        g(i) = ORD_MAX;
    });
    Kokkos::parallel_for("write edges", n, KOKKOS_LAMBDA(const ordinal_t i){
        ordinal_t u = i;
        ordinal_t v = in(i);
        //u has one out edge (if v < n)
        //and v has one in edge (if edge_count(v) == 1)
        if(v < n && edge_count(v) == 1){
            g(u) = v;
        }
    });
    return g;
}

}
