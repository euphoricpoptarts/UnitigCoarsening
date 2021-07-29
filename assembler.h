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
edge_offset_t find_vtx_from_edge(const char_view_t in_chars, const char_view_t map1_chars, const char_view_t map2_chars, const edge_view_t vtx_map, edge_offset_t offset, edge_offset_t k, edge_offset_t size){
    size_t hash = fnv(in_chars, offset, k - 1);
    size_t hash_cast = vtx_map.extent(0) - 1;
    hash = hash & hash_cast;
    edge_offset_t null_marker = 2*size;
    while(vtx_map(hash) != EDGE_MAX){
        edge_offset_t addr = vtx_map(hash);
        edge_offset_t actual = addr;
        bool nullified = false;
        if(addr >= null_marker){
            addr -= null_marker;
            nullified = true;
        }
        char_view_t cmp_chars = map1_chars;
        if(addr >= size){
            cmp_chars = map2_chars;
            addr -= size;
        }
        edge_offset_t hash_offset = addr*k;
        if(cmp(in_chars, cmp_chars, offset, hash_offset, k - 1)){
            return actual;
        }
        hash = (hash + 1) & hash_cast;
    }
    return EDGE_MAX;
}

edge_view_t generate_hashmap(char_view_t kmers, char_view_t rcomps, edge_offset_t k, edge_offset_t size){
    size_t hashmap_size = 1;
    size_t preferred_size = 2*size;
    edge_offset_t null_marker = 2*size;
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
            if(written >= null_marker){
                written = written - null_marker;
                if(cmp(kmers, kmers, k*i, k*written, k - 1)){
                    //hash value matches k-1 mer
                    //but has been nullified
                    //do nothing
                    break;
                }
            } else if(cmp(kmers, kmers, k*i, k*written, k - 1)){
                //hash value matches k-1 mer
                //nullify it
                Kokkos::atomic_compare_exchange_strong(&out(hash), written, i + null_marker);
                break;
            } 
            hash = (hash + 1) & hash_cast;
            success = Kokkos::atomic_compare_exchange_strong(&out(hash), EDGE_MAX, i);
        }
    });
    Kokkos::parallel_for("fill hashmap", size, KOKKOS_LAMBDA(const edge_offset_t i){
        size_t hash = fnv(rcomps, k*i, k - 1);
        hash = hash & hash_cast;
        bool success = Kokkos::atomic_compare_exchange_strong(&out(hash), EDGE_MAX, i + size);
        //linear probing
        while(!success){
            edge_offset_t written = out(hash);
            if(written >= null_marker){
                written = written - null_marker;
                if(written >= size){
                    writen = written - size;
                }
                if(cmp(rcomps, rcomps, k*i, k*written, k - 1)){
                    //hash value matches k-1 mer
                    //but has been nullified
                    //do nothing
                    break;
                }
            } else {
                if(written >= size){
                    writen = written - size;
                }
                if(cmp(rcomps, rcomps, k*i, k*written, k - 1)){
                    //hash value matches k-1 mer
                    //nullify it
                    Kokkos::atomic_compare_exchange_strong(&out(hash), written, i + size + null_marker);
                    break;
                }
            } 
            hash = (hash + 1) & hash_cast;
            success = Kokkos::atomic_compare_exchange_strong(&out(hash), EDGE_MAX, i + size);
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

void dump_graph(graph_type g){
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

graph_type convert_to_graph(vtx_view_t g){
    ordinal_t n = g.extent(0);
    vtx_view_t edge_count("edge count", n);
    Kokkos::parallel_for("count edges", n, KOKKOS_LAMBDA(const ordinal_t i){
        ordinal_t v = g(i);
        if(v != ORD_MAX){
            Kokkos::atomic_add(&edge_count(v), 1u);
            Kokkos::atomic_add(&edge_count(i), 1u);
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
    return graph_type(entries, row_map);
}

struct canon_graph {
    vtx_view_t right_edges;
    vtx_view_t left_edges;
};

canon_graph assemble_pruned_graph(char_view_t kmers, char_view_t rcomps, edge_view_t vtx_map, edge_offset_t k){
    ordinal_t n = kmers.extent(0) / k;
    //both the in and out edge counts for each vertex are packed into one char
    char_view_t edge_count("edge count", n);
    edge_view_t in1("in vertex", n);
    edge_view_t in2("in vertex", n);
    Kokkos::parallel_for("translate edges", n, KOKKOS_LAMBDA(const ordinal_t i){
        edge_offset_t v = find_vtx_from_edge(kmers, kmers, rcomps, vtx_map, i*k + 1, k, n);
        in1(i) = v;
    });
    Kokkos::parallel_for("translate rcomp edges", n, KOKKOS_LAMBDA(const ordinal_t i){
        edge_offset_t v = find_vtx_from_edge(rcomps, kmers, rcomps, vtx_map, i*k + 1, k, n);
        in2(i) = v;
    });
    //Kokkos::parallel_for("count edges", n, KOKKOS_LAMBDA(const ordinal_t i){
    //    edge_offset_t v = in(i);
    //    if(v != EDGE_MAX){
    //        if(v > n){
    //            v = v - n;
    //        }
    //        Kokkos::atomic_add(&edge_count(v), (char)1);
    //    }
    //});
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
    Kokkos::parallel_for("confirm reverse edge", n, KOKKOS_LAMBDA(const ordinal_t i){
        edge_offset_t v = g1(i);
        if(v < n){
            if(g2(v) - n != i){
                g1(i) = ORD_MAX;
            }
        } else if(v != ORD_MAX) {
            if(g1(v - n) != i) {
                g1(i) = ORD_MAX;
            }
        }
    });
    Kokkos::parallel_for("confirm reverse edge", n, KOKKOS_LAMBDA(const ordinal_t i){
        edge_offset_t v = g2(i);
        if(v < n){
            if(g2(v) - n != i){
                g2(i) = ORD_MAX;
            }
        } else if(v != ORD_MAX) {
            if(g1(v - n) != i) {
                g2(i) = ORD_MAX;
            }
        }
    });
    canon_graph g;
    g.right_edges = g1;
    g.left_edges = g2;
    return g;
}

}
