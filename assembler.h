#include "defs.h"

namespace unitig_compact {

//fnv hash algorithm
//for use on k-1 prefix
KOKKOS_INLINE_FUNCTION
uint32_t fnv_pref(const hash_vt chars, edge_offset_t offset, edge_offset_t comp){
    uint32_t hash = 2166136261U;
    const uint32_t pref_mask = 268435455;
    for(edge_offset_t i = offset; i < offset + comp; i++)
    {
        uint32_t x = chars(i);
        if(i + 1 == offset + comp) x = x & pref_mask;
        for(int i = 0; i < 4; i++){
            char y = (x >> i*8) & 255;
            hash = hash ^ y; // xor next byte into the bottom of the hash
            hash = hash * 16777619; // Multiply by prime number found to work well
        }
    }
    return hash;
}

//for use on k-1 suffix
KOKKOS_INLINE_FUNCTION
uint32_t fnv_suf(const hash_vt chars, edge_offset_t offset, edge_offset_t comp){
    uint32_t hash = 2166136261U;
    const uint32_t pref_mask = 268435455;
    for(edge_offset_t i = offset; i < offset + comp; i++)
    {
        uint32_t x = chars(i);
        x = x >> 2;
        if(i + 1 == offset + comp) x = x & pref_mask;
        else{
            uint32_t addon = chars(i + 1) & 3;
            x = x | (addon << 30);
        }
        for(int i = 0; i < 4; i++){
            char y = (x >> i*8) & 255;
            hash = hash ^ y; // xor next byte into the bottom of the hash
            hash = hash * 16777619; // Multiply by prime number found to work well
        }
    }
    return hash;
}

//compares k-1 prefixes
KOKKOS_INLINE_FUNCTION
bool cmp_pref(const hash_vt s1_chars, const hash_vt s2_chars, const edge_offset_t s1_offset, const edge_offset_t s2_offset, const edge_offset_t comp){
    const uint32_t pref_mask = 268435455;
    for(edge_offset_t i = 0; i < comp; i++){
        if(i + 1 == comp){
            if((s1_chars(s1_offset + i) & pref_mask) != (s2_chars(s2_offset + i) & pref_mask)){
                return false;
            }
        } else {
            if(s1_chars(s1_offset + i) != s2_chars(s2_offset + i)){
                return false;
            }
        }
    }
    return true;
}

//compares a k-1 suffix to a k-1 prefix
KOKKOS_INLINE_FUNCTION
bool cmp_suf(const hash_vt s1_chars, const hash_vt s2_chars, const edge_offset_t s1_offset, const edge_offset_t s2_offset, const edge_offset_t comp){
    const uint32_t pref_mask = 268435455;
    for(edge_offset_t i = 0; i < comp; i++){
        uint32_t suf = s1_chars(s1_offset + i);
        suf = suf >> 2;
        if(i + 1 == comp){
            if((suf & pref_mask) != (s2_chars(s2_offset + i) & pref_mask)){
                return false;
            }
        } else {
            uint32_t suf_addon = s1_chars(s1_offset + i + 1) & 3;
            suf = suf | (suf_addon << 30);
            if(suf != s2_chars(s2_offset + i)){
                return false;
            }
        }
    }
    return true;
}

//check if vtx found in edge_chars at offset exists in vtx_chars using a hashmap
KOKKOS_INLINE_FUNCTION
ordinal_t find_vtx_from_edge(const hash_vt& vtx_chars, const vtx_view_t& vtx_map, const hash_vt& vtx_chars2, const edge_offset_t offset, const edge_offset_t comp, const ordinal_t size){
    uint32_t hash = fnv_suf(vtx_chars2, offset, comp);
    uint32_t hash_cast = vtx_map.extent(0) - 1;
    hash = hash & hash_cast;
    while(vtx_map(hash) != ORD_MAX){
        ordinal_t addr = vtx_map(hash);
        bool nullified = false;
        if(addr >= size){
            addr -= size;
            nullified = true;
        }
        edge_offset_t hash_offset = addr*comp;
        if(cmp_suf(vtx_chars2, vtx_chars, offset, hash_offset, comp)){
            if(!nullified){
                return addr;
            } else {
                return addr + size;
            }
        }
        hash = (hash + 1) & hash_cast;
    }
    return ORD_MAX;
}

vtx_view_t init_hashmap(ordinal_t max_size){
    size_t hashmap_size = 1;
    while(hashmap_size < 2*max_size) hashmap_size <<= 1;
    vtx_view_t hashmap("hashmap", hashmap_size);
    return hashmap;
}

void generate_hashmap(vtx_view_t hashmap, const hash_vt kmers, edge_offset_t comp, ordinal_t size){
    size_t hashmap_size = hashmap.extent(0);
    Kokkos::parallel_for("init hashmap", hashmap_size, KOKKOS_LAMBDA(const ordinal_t i){
        hashmap(i) = ORD_MAX;
    });
    size_t hash_cast = hashmap_size - 1;
    Kokkos::parallel_for("fill hashmap", size, KOKKOS_LAMBDA(const ordinal_t i){
        uint32_t hash = fnv_pref(kmers, comp*i, comp) & hash_cast;
        bool success = Kokkos::atomic_compare_exchange_strong(&hashmap(hash), ORD_MAX, i);
        //linear probing
        while(!success){
            ordinal_t written = hashmap(hash);
            if(written >= size){
                written = written - size;
                if(cmp_pref(kmers, kmers, comp*i, comp*written, comp)){
                    //hash value matches k-1 prefix
                    //but has been nullified
                    //do nothing
                    break;
                }
            } else if(cmp_pref(kmers, kmers, comp*i, comp*written, comp)){
                //hash value matches k-1 prefix
                //nullify it
                Kokkos::atomic_compare_exchange_strong(&hashmap(hash), written, i + size);
                break;
            }
            hash = (hash + 1) & hash_cast;
            success = Kokkos::atomic_compare_exchange_strong(&hashmap(hash), ORD_MAX, i);
        }
    });
}

struct assembler_data {
    vtx_view_t in;
    vtx_view_t out;
    char_view_t edge_count;
};

assembler_data init_assembler(ordinal_t max_n, ordinal_t max_np){
    assembler_data d;
    d.in = vtx_view_t("in vertex", max_np);
    d.out = vtx_view_t("out vertex", max_np);
    d.edge_count = char_view_t("edge count", max_n);
    return d;
}

crosses assemble_pruned_graph(assembler_data assembler, const hash_vt kmers, vtx_view_t vtx_map, const hash_vt cross, vtx_view_t cross_ids, edge_offset_t comp, vtx_view_t g){
    ordinal_t n = kmers.extent(0) / comp;
    ordinal_t n_cross = cross.extent(0) / comp;
    Kokkos::parallel_for("reset edge count", n, KOKKOS_LAMBDA(const ordinal_t i){
        assembler.edge_count(i) = 0;
    });
    //both the in and out edge counts for each vertex are packed into one char
    Kokkos::parallel_for("translate edges", n, KOKKOS_LAMBDA(const ordinal_t i){
        ordinal_t v = find_vtx_from_edge(kmers, vtx_map, kmers, i*comp, comp, n);
        assembler.in(i) = v;
    });
    Kokkos::parallel_for("translate cross edges", n_cross, KOKKOS_LAMBDA(const ordinal_t i){
        ordinal_t v = find_vtx_from_edge(kmers, vtx_map, cross, i*comp, comp, n);
        assembler.in(n + i) = v;
    });
    Kokkos::parallel_for("count edges", n + n_cross, KOKKOS_LAMBDA(const ordinal_t i){
        ordinal_t v = assembler.in(i);
        if(v != ORD_MAX){
            if(v > n){
                v = v - n;
            }
            Kokkos::atomic_add(&assembler.edge_count(v), (char)1);
        }
    });
    Kokkos::parallel_for("write edges", n, KOKKOS_LAMBDA(const ordinal_t i){
        ordinal_t u = i;
        ordinal_t v = assembler.in(i);
        //u has one out edge (if v < n)
        //and v has one in edge (if edge_count(v) == 1)
        if(v < n && assembler.edge_count(v) == 1){
            g(u) = v;
        }
    });
    vtx_view_t out_cross_id("out cross ids", n_cross);
    Kokkos::parallel_for("init out cross ids", n_cross, KOKKOS_LAMBDA(const ordinal_t i){
        ordinal_t u = cross_ids(i);
        out_cross_id(i) = u;
    });
    vtx_view_t in_cross_id("in cross ids", n_cross);
    Kokkos::parallel_for("init in cross ids", n_cross, KOKKOS_LAMBDA(const ordinal_t i){
        ordinal_t v = assembler.in(n + i);
        if(v < n && (assembler.edge_count(v) == 1)){
            in_cross_id(i) = v;
        } else {
            in_cross_id(i) = ORD_MAX;
        }
    });
    crosses c;
    c.in = in_cross_id;
    c.out = out_cross_id;
    return c;
}

}
