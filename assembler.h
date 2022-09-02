#include "definitions_kokkos.h"

namespace unitig_compact {

//fnv hash algorithm
//for use on k-1 prefix
KOKKOS_INLINE_FUNCTION
uint32_t fnv_pref(const comp_vt chars, edge_offset_t offset, edge_offset_t comp){
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
uint32_t fnv_suf(const comp_vt chars, edge_offset_t offset, edge_offset_t comp){
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
bool cmp_pref(const comp_vt s1_chars, const comp_vt s2_chars, const edge_offset_t s1_offset, const edge_offset_t s2_offset, const edge_offset_t comp){
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
bool cmp_suf(const comp_vt s1_chars, const comp_vt s2_chars, const edge_offset_t s1_offset, const edge_offset_t s2_offset, const edge_offset_t comp){
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
edge_offset_t find_vtx_from_edge(const comp_vt in_chars, const comp_vt map1_chars, const comp_vt map2_chars, const vtx_view_t vtx_map, edge_offset_t offset, edge_offset_t comp, edge_offset_t size){
    size_t hash = fnv_suf(in_chars, offset, comp);
    size_t hash_cast = vtx_map.extent(0) - 1;
    hash = hash & hash_cast;
    edge_offset_t null_marker = 2*size;
    while(vtx_map(hash) != HASH_NULL){
        edge_offset_t addr = vtx_map(hash);
        edge_offset_t actual = addr;
        //bool nullified = false;
        if(addr >= null_marker){
            addr -= null_marker;
            //nullified = true;
        }
        comp_vt cmp_chars = map1_chars;
        if(addr >= size){
            cmp_chars = map2_chars;
            addr -= size;
        }
        edge_offset_t hash_offset = addr*comp;
        if(cmp_suf(in_chars, cmp_chars, offset, hash_offset, comp)){
            return actual;
        }
        hash = (hash + 1) & hash_cast;
    }
    return HASH_NULL;
}

vtx_view_t generate_hashmap(comp_vt kmers, comp_vt rcomps, edge_offset_t comp, edge_offset_t size){
    size_t hashmap_size = 1;
    //we need to insert 2*size entries, and we want 50% extra space, so multiply by 3
    size_t preferred_size = 3*size;
    edge_offset_t null_marker = 2*size;
    while(hashmap_size < preferred_size) hashmap_size <<= 1;
    vtx_view_t out(Kokkos::ViewAllocateWithoutInitializing("hashmap"), hashmap_size);
    Kokkos::deep_copy(out, HASH_NULL);
    size_t hash_cast = hashmap_size - 1;
    Kokkos::parallel_for("fill hashmap", size, KOKKOS_LAMBDA(const edge_offset_t i){
        size_t hash = fnv_pref(kmers, comp*i, comp);
        hash = hash & hash_cast;
        bool success = Kokkos::atomic_compare_exchange_strong(&out(hash), HASH_NULL, i);
        //linear probing
        while(!success){
            edge_offset_t written = out(hash);
            if(written >= null_marker){
                written = written - null_marker;
                if(cmp_pref(kmers, kmers, comp*i, comp*written, comp)){
                    //hash value matches k-1 mer
                    //but has been nullified
                    //do nothing
                    break;
                }
            } else if(cmp_pref(kmers, kmers, comp*i, comp*written, comp)){
                //hash value matches k-1 mer
                //nullify it
                Kokkos::atomic_compare_exchange_strong(&out(hash), written, i + null_marker);
                break;
            } 
            hash = (hash + 1) & hash_cast;
            success = Kokkos::atomic_compare_exchange_strong(&out(hash), HASH_NULL, i);
        }
    });
    Kokkos::parallel_for("fill hashmap", size, KOKKOS_LAMBDA(const edge_offset_t i){
        size_t hash = fnv_pref(rcomps, comp*i, comp);
        hash = hash & hash_cast;
        bool success = Kokkos::atomic_compare_exchange_strong(&out(hash), HASH_NULL, i + size);
        //linear probing
        while(!success){
            comp_vt compare = kmers;
            edge_offset_t written = out(hash);
            edge_offset_t write_check = written;
            if(written >= null_marker){
                written = written - null_marker;
                if(written >= size){
                    written = written - size;
                    compare = rcomps;
                }
                if(cmp_pref(rcomps, compare, comp*i, comp*written, comp)){
                    //hash value matches k-1 mer
                    //but has been nullified
                    //do nothing
                    break;
                }
            } else {
                if(written >= size){
                    written = written - size;
                    compare = rcomps;
                }
                if(cmp_pref(rcomps, compare, comp*i, comp*written, comp)){
                    //hash value matches k-1 mer
                    //nullify it
                    Kokkos::atomic_compare_exchange_strong(&out(hash), write_check, i + size + null_marker);
                    break;
                }
            } 
            hash = (hash + 1) & hash_cast;
            success = Kokkos::atomic_compare_exchange_strong(&out(hash), HASH_NULL, i + size);
        }
    });
    return out; 
}

char_view_t generate_rcomps(char_view_t kmers, edge_offset_t k, edge_offset_t size){
    char_view_t rcomps(Kokkos::ViewAllocateWithoutInitializing("reverse complements"), k * size);
    char_mirror_t char_map_mirror("char map mirror", 256);
    char_map_mirror('A') = 'T';
    char_map_mirror('C') = 'G';
    char_map_mirror('G') = 'C';
    char_map_mirror('T') = 'A';
    char_view_t char_map("char map", 256);
    Kokkos::deep_copy(char_map, char_map_mirror);
    Kokkos::parallel_for("fill rcomps", size, KOKKOS_LAMBDA(const ordinal_t i){
        edge_offset_t b = i*k;
        edge_offset_t e = (i + 1)*k - 1;
        for(edge_offset_t j = 0; j < k; j++){
            rcomps(b + j) = char_map(kmers(e - j));
        }
    });
    return rcomps;
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

void dump_graph(graph_t g){
    long n = g.numRows();
    long t_e = g.entries.extent(0);
    edge_mirror_t row_map_x("row map", n + 1);
    Kokkos::deep_copy(row_map_x, g.row_map);
    vtx_mirror_t entries_x("entries", t_e);
    Kokkos::deep_copy(entries_x, g.entries);
    Kokkos::View<unsigned int*>::HostMirror row_map("row map", n + 1);
    Kokkos::parallel_for("row map to int", host_policy(0, n + 1), KOKKOS_LAMBDA(const ordinal_t i){
        row_map(i) = row_map_x(i);
    });
    Kokkos::View<unsigned int*>::HostMirror entries("entries", t_e);
    Kokkos::parallel_for("entries to int", host_policy(0, t_e), KOKKOS_LAMBDA(const ordinal_t i){
        entries(i) = entries_x(i);
    });
    FILE* f = fopen("debruijn.csr", "wb");
    fwrite(&n, sizeof(long), 1, f);
    fwrite(&t_e, sizeof(long), 1, f);
    fwrite(row_map.data(), sizeof(unsigned int), n + 1, f);
    fwrite(entries.data(), sizeof(unsigned int), t_e, f);
    fclose(f);
}

graph_t convert_to_graph(vtx_view_t g){
    ordinal_t n = g.extent(0);
    vtx_view_t edge_count("edge count", n);
    Kokkos::parallel_for("count edges", n, KOKKOS_LAMBDA(const ordinal_t i){
        ordinal_t v = g(i);
        if(v != ORD_MAX){
            Kokkos::atomic_add(&edge_count(v), 1);
            Kokkos::atomic_add(&edge_count(i), 1);
        }
    });
    edge_view_t row_map("row map", n + 1);
    prefix_sum1 f(edge_count, row_map);
    Kokkos::parallel_scan("exclusive prefix sum", n, f);
    edge_subview_t total_edges_s = Kokkos::subview(row_map, n);
    edge_offset_t total_edges = 0;
    Kokkos::deep_copy(total_edges, total_edges_s);
    vtx_view_t entries("entries", total_edges);
    Kokkos::parallel_for("write edges", n, KOKKOS_LAMBDA(const ordinal_t i){
        ordinal_t v = g(i);
        if(v != ORD_MAX){
            entries(row_map(v + 1) - 1) = i;
            entries(row_map(i)) = v;
        }
    });
    return graph_t(entries, row_map);
}

canon_graph assemble_pruned_graph(comp_vt kmers, comp_vt rcomps, vtx_view_t vtx_map, edge_offset_t comp){
    ordinal_t n = kmers.extent(0) / comp;
    //both the in and out edge counts for each vertex are packed into one char
    char_view_t edge_count("edge count", n);
    edge_view_t in1("in vertex", n);
    edge_view_t in2("in vertex", n);
    Kokkos::parallel_for("translate edges", n, KOKKOS_LAMBDA(const ordinal_t i){
        edge_offset_t v = find_vtx_from_edge(kmers, kmers, rcomps, vtx_map, i*comp, comp, n);
        in1(i) = v;
    });
    Kokkos::parallel_for("translate rcomp edges", n, KOKKOS_LAMBDA(const ordinal_t i){
        edge_offset_t v = find_vtx_from_edge(rcomps, kmers, rcomps, vtx_map, i*comp, comp, n);
        in2(i) = v;
    });
    vtx_view_t g1("pruned out entries", n);
    vtx_view_t g2("pruned out entries", n);
    Kokkos::parallel_for("init g", n, KOKKOS_LAMBDA(const ordinal_t i){
        g1(i) = ORD_MAX;
        g2(i) = ORD_MAX;
    });
    Kokkos::parallel_for("write edges", n, KOKKOS_LAMBDA(const ordinal_t i){
        ordinal_t u = i;
        edge_offset_t v = in1(i);
        //u has one out edge (if v < 2*n)
        if(v < 2*n){
            g1(u) = v;
        }
    });
    Kokkos::parallel_for("write edges", n, KOKKOS_LAMBDA(const ordinal_t i){
        ordinal_t u = i;
        edge_offset_t v = in2(i);
        //u has one out edge (if v < 2*n)
        if(v < 2*n){
            g2(u) = v;
        }
    });
    vtx_view_t reset_left("reset left", n);
    Kokkos::parallel_for("confirm reverse edge", n, KOKKOS_LAMBDA(const ordinal_t i){
        ordinal_t v = g1(i);
        if(v < n){
            if(g2(v) - n != i){
                reset_left(i) = 1;
            }
        } else if(v != ORD_MAX) {
            if(g1(v - n) - n != i) {
                reset_left(i) = 1;
            }
        }
    });
    vtx_view_t reset_right("reset right", n);
    Kokkos::parallel_for("confirm reverse edge", n, KOKKOS_LAMBDA(const ordinal_t i){
        ordinal_t v = g2(i);
        if(v < n){
            if(g2(v) != i){
                reset_right(i) = 1;
            }
        } else if(v != ORD_MAX) {
            if(g1(v - n) != i) {
                reset_right(i) = 1;
            }
        }
    });
    Kokkos::parallel_for("remove markers", n, KOKKOS_LAMBDA(const ordinal_t i){
        if(reset_left(i) == 1){
            g1(i) = ORD_MAX;
        } else if(g1(i) >= n && g1(i) != ORD_MAX){
            g1(i) -= n;
        }
        if(g1(i) == i){
            //remove self-loop edges
            g1(i) = ORD_MAX;
        }
    });
    Kokkos::parallel_for("remove markers", n, KOKKOS_LAMBDA(const ordinal_t i){
        if(reset_right(i) == 1){
            g2(i) = ORD_MAX;
        } else if(g2(i) >= n && g2(i) != ORD_MAX){
            g2(i) -= n;
        }
        if(g2(i) == i){
            //remove self-loop edges
            g2(i) = ORD_MAX;
        }
    });
    canon_graph g;
    g.right_edges = g1;
    g.left_edges = g2;
    g.size = n;
    return g;
}

}
