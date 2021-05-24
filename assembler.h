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

//check if edge formed by appending extension to kmer at offset exists in edges using a hashmap
KOKKOS_INLINE_FUNCTION
bool find_edge(const char_view_t chars, const char_view_t edges, const vtx_view_t edge_map, edge_offset_t offset, edge_offset_t k, char extension){
    uint32_t hash = fnv(chars, offset, k);
    hash = hash ^ (extension); // xor next byte into the bottom of the hash
    hash = hash * 16777619; // Multiply by prime number found to work well
    uint32_t hash_cast = edge_map.extent(0) - 1;
    hash = hash & hash_cast;
    while(edge_map(hash) != ORD_MAX){
        edge_offset_t hash_offset = edge_map(hash)*(k + 1);
        if(cmp(chars, edges, offset, hash_offset, k) && extension == edges(hash_offset + k)){
            return true;
        }
        hash = (hash + 1) & hash_cast;
    }
    return false;
}

//check if vtx formed by appending extension to (k-1)mer at offset exists in vtxs using a hashmap
KOKKOS_INLINE_FUNCTION
ordinal_t find_vtx(const char_view_t chars, const vtx_view_t vtx_map, edge_offset_t offset, edge_offset_t k, char extension){
    uint32_t hash = fnv(chars, offset, k - 1);
    hash = hash ^ (extension); // xor next byte into the bottom of the hash
    hash = hash * 16777619; // Multiply by prime number found to work well
    uint32_t hash_cast = vtx_map.extent(0) - 1;
    hash = hash & hash_cast;
    while(vtx_map(hash) != ORD_MAX){
        edge_offset_t hash_offset = vtx_map(hash)*k;
        if(cmp(chars, chars, offset, hash_offset, k - 1) && (extension == chars(hash_offset + (k - 1))) ){
            return vtx_map(hash);
        }
        hash = (hash + 1) & hash_cast;
    }
    return ORD_MAX;
}

//check if vtx found in edge_chars at offset exists in vtx_chars using a hashmap
//KOKKOS_INLINE_FUNCTION
//ordinal_t find_vtx_from_edge(const char_view_t vtx_chars, const vtx_view_t vtx_map, const char_view_t edge_chars, edge_offset_t offset, edge_offset_t k){
//    uint32_t hash = fnv(edge_chars, offset, k);
//    uint32_t hash_cast = vtx_map.extent(0) - 1;
//    hash = hash & hash_cast;
//    while(vtx_map(hash) != ORD_MAX){
//        edge_offset_t hash_offset = vtx_map(hash)*k;
//        if(cmp(edge_chars, vtx_chars, offset, hash_offset, k)){
//            return vtx_map(hash);
//        }
//        hash = (hash + 1) & hash_cast;
//    }
//    return ORD_MAX;
//}

KOKKOS_INLINE_FUNCTION
ordinal_t find_vtx_from_edge(const vtx_view_t char_map, const vtx_view_t vtx_map, const char_view_t edge_chars, edge_offset_t offset, edge_offset_t k){
    edge_offset_t tree_pos = 0;
    for(edge_offset_t j = offset; j < offset + k; j++){
        char c = edge_chars(j);
        edge_offset_t key = char_map(c);
        tree_pos += key;
        tree_pos = vtx_map(tree_pos);
    }
    return vtx_map(tree_pos);
}

//vtx_view_t generate_hashmap(char_view_t kmers, edge_offset_t k, ordinal_t size){
//    size_t hashmap_size = 1;
//    while(hashmap_size < 2*size) hashmap_size <<= 1;
//    vtx_view_t out("hashmap", hashmap_size);
//    Kokkos::parallel_for("init hashmap", hashmap_size, KOKKOS_LAMBDA(const ordinal_t i){
//        out(i) = ORD_MAX;
//    });
//    size_t hash_cast = hashmap_size - 1;
//    Kokkos::parallel_for("fill hashmap", size, KOKKOS_LAMBDA(const ordinal_t i){
//        uint32_t hash = fnv(kmers, k*i, k) & hash_cast;
//        bool success = Kokkos::atomic_compare_exchange_strong(&out(hash), ORD_MAX, i);
//        //linear probing
//        //all values are unique so no need to check
//        while(!success){
//            hash = (hash + 1) & hash_cast;
//            success = Kokkos::atomic_compare_exchange_strong(&out(hash), ORD_MAX, i);
//        }
//    });
//    return out; 
//}

vtx_view_t generate_hashmap(char_view_t kmers, edge_offset_t k, ordinal_t size){
    edge_offset_t out_size = kmers.extent(0) * 4;
    vtx_view_t out("hashmap", out_size);
    edge_subview_t tree_count("tree count");
    Kokkos::parallel_for("init hashmap", out_size, KOKKOS_LAMBDA(const edge_offset_t i){
        out(i) = ORD_MAX;
    });
    //init first 4 levels as a dense 4-tree
    edge_offset_t allocate = 4 + 16 + 64 + 256;
    Kokkos::parallel_for("init tree", allocate, KOKKOS_LAMBDA(const edge_offset_t i){
        out(i) = 4*i + 4;
    });
    allocate += 1024;
    Kokkos::deep_copy(tree_count, allocate);
    vtx_mirror_t char_map_mirror("char map mirror", 256);
    char_map_mirror('A') = 0;
    char_map_mirror('C') = 1;
    char_map_mirror('G') = 2;
    char_map_mirror('T') = 3;
    vtx_view_t char_map("char map", 256);
    Kokkos::deep_copy(char_map, char_map_mirror);
    vtx_view_t succeeded("successes", size);
    ordinal_t success_count = 0;
    while(success_count < size){
    Kokkos::parallel_reduce("fill hashmap", size, KOKKOS_LAMBDA(const ordinal_t i, ordinal_t& update){
        if(succeeded(i) == 1){
            update++;
        } else {
            bool failed = false;
            edge_offset_t tree_pos = 0;
            for(edge_offset_t j = k*i; j < k*(i + 1); j++){
                char c = kmers(j);
                edge_offset_t key = char_map(c);
                tree_pos += key;
                if(out(tree_pos) < ORD_MAX - 1){
                    tree_pos = out(tree_pos);
                } else {
                    //allocate 4 new leaves
                    if(Kokkos::atomic_compare_exchange_strong(&out(tree_pos), ORD_MAX, ORD_MAX - 1)){
                        edge_offset_t new_pos = Kokkos::atomic_fetch_add(&tree_count(), 4);
                        out(tree_pos) = new_pos;
                        //Kokkos::atomic_compare_exchange_strong(&out(tree_pos), ORD_MAX - 1, (ordinal_t) new_pos);
                        tree_pos = new_pos;
                    } else {
                        failed = true;
                        break;
                        //somebody else is in the process of allocating them
                        //this is technically a lock
                        //while(out(tree_pos) == ORD_MAX - 1){ }
                        //tree_pos = out(tree_pos);
                    }
                }
            }
            if(!failed){
                out(tree_pos) = i;
                succeeded(i) = 1;
                update++;
            }
        }
    }, success_count);
    printf("Total successes: %u\n", success_count);
    }
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

vtx_view_t assemble_pruned_graph(char_view_t kmers, char_view_t kpmers, vtx_view_t vtx_map, edge_offset_t k){
    ordinal_t n = kmers.extent(0) / k;
    ordinal_t np = kpmers.extent(0) / (k + 1);
    //both the in and out edge counts for each vertex are packed into one char
    char_view_t edge_count("edge count", n);
    vtx_view_t in("in vertex", np);
    vtx_view_t out("out vertex", np);
    vtx_mirror_t char_map_mirror("char map mirror", 256);
    char_map_mirror('A') = 0;
    char_map_mirror('C') = 1;
    char_map_mirror('G') = 2;
    char_map_mirror('T') = 3;
    vtx_view_t char_map("char map", 256);
    Kokkos::deep_copy(char_map, char_map_mirror);
    Kokkos::parallel_for("translate edges", np, KOKKOS_LAMBDA(const ordinal_t i){
        ordinal_t u = find_vtx_from_edge(char_map, vtx_map, kpmers, i*(k+1), k);
        out(i) = u;
    });
    Kokkos::parallel_for("translate edges", np, KOKKOS_LAMBDA(const ordinal_t i){
        ordinal_t v = find_vtx_from_edge(char_map, vtx_map, kpmers, i*(k+1) + 1, k);
        in(i) = v;
    });
    Kokkos::parallel_for("count edges", np, KOKKOS_LAMBDA(const ordinal_t i){
        ordinal_t u = out(i);
        //u can have at most 4 out edges
        Kokkos::atomic_add(&edge_count(u), (char)1);
    });
    Kokkos::parallel_for("count edges", np, KOKKOS_LAMBDA(const ordinal_t i){
        ordinal_t v = in(i);
        //v can have at most 4 in edges
        //so we count the in edges as multiples of 8 (first power of 2 greater than 4, easy to do bitwise ops)
        Kokkos::atomic_add(&edge_count(v), (char)8);
    });
    vtx_view_t g("pruned out entries", n);
    Kokkos::parallel_for("init g", n, KOKKOS_LAMBDA(const ordinal_t i){
        g(i) = ORD_MAX;
    });
    Kokkos::parallel_for("write edges", np, KOKKOS_LAMBDA(const ordinal_t i){
        ordinal_t u = out(i);
        ordinal_t v = in(i);
        //u has one out edge
        //and v has one in edge
        if((edge_count(u) & 7) == 1 && (edge_count(v) >> 3) == 1){
            g(u) = v;
        }
    });
    return g;
}

graph_type assemble_graph(char_view_t kmers, char_view_t kpmers, vtx_view_t vtx_map, edge_offset_t k){
    ordinal_t n = kmers.extent(0) / k;
    ordinal_t np = kpmers.extent(0) / (k + 1);
    edge_view_t row_map("row map", n+1);
    vtx_view_t in("in vertex", np);
    vtx_view_t out("out vertex", np);
    vtx_mirror_t char_map_mirror("char map mirror", 256);
    char_map_mirror('A') = 0;
    char_map_mirror('C') = 1;
    char_map_mirror('G') = 2;
    char_map_mirror('T') = 3;
    vtx_view_t char_map("char map", 256);
    Kokkos::deep_copy(char_map, char_map_mirror);
    Kokkos::parallel_for("translate edges", np, KOKKOS_LAMBDA(const ordinal_t i){
        ordinal_t u = find_vtx_from_edge(char_map, vtx_map, kpmers, i*(k+1), k);
        out(i) = u;
    });
    Kokkos::parallel_for("translate edges", np, KOKKOS_LAMBDA(const ordinal_t i){
        ordinal_t v = find_vtx_from_edge(char_map, vtx_map, kpmers, i*(k+1) + 1, k);
        in(i) = v;
    });
    vtx_view_t edge_count("edge count", n);
    Kokkos::parallel_for("count edges", np, KOKKOS_LAMBDA(const ordinal_t i){
        ordinal_t u = out(i);
        Kokkos::atomic_increment(&edge_count(u));
    });
    Kokkos::parallel_scan("calc source offsets", n, prefix_sum1(edge_count, row_map));
    edge_subview_t rm_subview = Kokkos::subview(row_map, n);
    edge_offset_t total_e = 0;
    Kokkos::deep_copy(total_e, rm_subview);
    Kokkos::parallel_for("reset edge count", n, KOKKOS_LAMBDA(const ordinal_t i){
        edge_count(i) = 0;
    });
    vtx_view_t entries("pruned out entries", total_e);
    Kokkos::parallel_for("write edges", np, KOKKOS_LAMBDA(const ordinal_t i){
        ordinal_t u = out(i);
        ordinal_t v = in(i);
        edge_offset_t insert = row_map(u) + Kokkos::atomic_fetch_add(&edge_count(u), 1);
        entries(insert) = v;
    });
    graph_type g(entries, row_map);
    return g;
}

}
