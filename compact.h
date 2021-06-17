#include "definitions_kokkos.h"
#include <iostream>
#include <fstream>

namespace unitig_compact{

void write_to_f(char_view_t unitigs, std::string fname){
    char_mirror_t chars("chars", unitigs.extent(0));
    Kokkos::deep_copy(chars, unitigs);
    std::ofstream of(fname, std::ofstream::out | std::ofstream::app);
    //FILE *of = fopen(fname.c_str(), "a");
    if (!of.is_open()) {
        printf("Error: Could not open input file. Exiting ...\n");
        exit(1);
    }
    //string is already formatted, dump it into file
    //need to be cautious about non-integer type of chars.extent(0)
    //fprintf(of, "%.*s", chars.extent(0), chars.data());
    of.write(chars.data(), chars.extent(0));
    //fclose(of);
    of.close();
}

void write_unitigs3(char_view_t kmers, edge_view_t kmer_rows, edge_offset_t k, graph_type glue_action, std::string fname){
    edge_offset_t null_size = glue_action.numRows();
    edge_view_t write_sizes("write sizes", null_size + 1);
    Kokkos::parallel_scan("count writes", r_policy(0, null_size), KOKKOS_LAMBDA(const edge_offset_t i, edge_offset_t& update, const bool final){
        //+1 for '\n'
        //k-1 for prefix of first k-mer
        edge_offset_t size = glue_action.row_map(i + 1) - glue_action.row_map(i) + 1 + (k - 1);
        if(final){
            write_sizes(i) = update;
            if(i + 1 == null_size){
                write_sizes(null_size) = update + size;
            }
        }
        update += size;
    });
    edge_offset_t write_size = 0;
    edge_subview_t write_size_sub = Kokkos::subview(write_sizes, null_size);
    Kokkos::deep_copy(write_size, write_size_sub);
    char_view_t writes("writes", write_size);
    ordinal_t max_rows = kmer_rows.extent(0);
    Kokkos::parallel_for("move writes", policy(null_size, Kokkos::AUTO), KOKKOS_LAMBDA(const member& thread){
        ordinal_t i = thread.league_rank();
        edge_offset_t write_offset = write_sizes(i);
        edge_offset_t start = glue_action.row_map(i);
        edge_offset_t end = glue_action.row_map(i + 1);
        Kokkos::parallel_for(Kokkos::TeamThreadRange(thread, start, end), [=] (const edge_offset_t j){
            ordinal_t u = glue_action.entries(j);
            edge_offset_t kmer_start = kmer_rows(u);
            edge_offset_t kmer_end = kmer_rows(u + 1);
            if(j > start){
                edge_offset_t offset = write_offset + (k - 1) + (j - start);
                writes(offset) = kmers(kmer_end - 1);
            } else {
                edge_offset_t offset = write_offset;
                for(edge_offset_t x = kmer_start; x < kmer_end; x++){
                    writes(offset) = kmers(x);
                    offset++;
                }
            }
            if(j + 1 == end){
                writes(write_sizes(i + 1) - 1) = '\n';
            }
        });
    });
    write_to_f(writes, fname);
}

void write_unitigs2(char_view_t kmers, edge_offset_t k, graph_type glue_action, std::string fname){
    edge_offset_t null_size = glue_action.numRows();
    edge_view_t write_sizes("write sizes", null_size + 1);
    Kokkos::parallel_scan("count writes", r_policy(0, null_size), KOKKOS_LAMBDA(const edge_offset_t i, edge_offset_t& update, const bool final){
        //+1 for '\n'
        //k-1 for prefix of first k-mer
        edge_offset_t size = glue_action.row_map(i + 1) - glue_action.row_map(i) + 1 + (k - 1);
        if(final){
            write_sizes(i) = update;
            if(i + 1 == null_size){
                write_sizes(null_size) = update + size;
            }
        }
        update += size;
    });
    edge_offset_t write_size = 0;
    edge_subview_t write_size_sub = Kokkos::subview(write_sizes, null_size);
    Kokkos::deep_copy(write_size, write_size_sub);
    char_view_t writes("writes", write_size);
    Kokkos::parallel_for("move writes", policy(null_size, Kokkos::AUTO), KOKKOS_LAMBDA(const member& thread){
        ordinal_t i = thread.league_rank();
        edge_offset_t write_offset = write_sizes(i);
        edge_offset_t start = glue_action.row_map(i);
        edge_offset_t end = glue_action.row_map(i + 1);
        Kokkos::parallel_for(Kokkos::TeamThreadRange(thread, start, end), [=] (const edge_offset_t j){
            ordinal_t u = glue_action.entries(j);
            if(j > start){
                edge_offset_t offset = write_offset + (k - 1) + (j - start);
                writes(offset) = kmers((u + 1)*k - 1);
            } else {
                edge_offset_t offset = write_offset;
                for(edge_offset_t x = u*k; x < k*(u+1); x++){
                    writes(offset) = kmers(x);
                    offset++;
                }
            }
            if(j + 1 == end){
                writes(write_sizes(i + 1) - 1) = '\n';
            }
        });
    });
    write_to_f(writes, fname);
}

void write_unitigs(char_view_t kmers, edge_view_t kmer_offsets, graph_type glue_action, std::string fname){
    edge_offset_t write_size = 0;
    c_edge_subview_t start_writes_sub = Kokkos::subview(glue_action.row_map, 0);
    c_edge_subview_t end_writes_sub = Kokkos::subview(glue_action.row_map, 1);
    edge_offset_t start_writes = 0, end_writes = 0;
    Kokkos::deep_copy(start_writes, start_writes_sub);
    Kokkos::deep_copy(end_writes, end_writes_sub);
    edge_view_t write_sizes("write sizes", end_writes - start_writes + 1);
    Kokkos::parallel_scan("count writes", r_policy(start_writes, end_writes), KOKKOS_LAMBDA(const edge_offset_t i, edge_offset_t& update, const bool final){
        ordinal_t u = glue_action.entries(i);
        //+1 for '\n'
        edge_offset_t size = kmer_offsets(u + 1) - kmer_offsets(u) + 1;
        if(final){
            write_sizes(i - start_writes) = update;
            if(i + 1 == end_writes){
                write_sizes(end_writes - start_writes) = update + size;
            }
        }
        update += size;
    });
    edge_subview_t write_size_sub = Kokkos::subview(write_sizes, end_writes - start_writes);
    Kokkos::deep_copy(write_size, write_size_sub);
    char_view_t writes("writes", write_size);
#if defined(HUGE) || defined(LARGE)
    printf("write out unitigs size: %lu\n", write_size);
    printf("write out unitigs count: %lu\n", end_writes - start_writes);
#else
    printf("write out unitigs size: %u\n", write_size);
    printf("write out unitigs count: %u\n", end_writes - start_writes);
#endif
    Kokkos::parallel_for("move writes", policy(end_writes - start_writes, Kokkos::AUTO), KOKKOS_LAMBDA(const member& thread){
        const edge_offset_t i = thread.league_rank() + start_writes;
        ordinal_t u = glue_action.entries(i);
        edge_offset_t write_offset = write_sizes(i - start_writes);
        Kokkos::parallel_for(Kokkos::TeamThreadRange(thread, kmer_offsets(u), kmer_offsets(u + 1)), [=] (const edge_offset_t j){
            edge_offset_t offset = write_offset + (j - kmer_offsets(u));
            writes(offset) = kmers(j);
            if(offset + 2 == write_sizes(i + 1 - start_writes)){
                writes(offset + 1) = '\n';
            }
        });
    });
    write_to_f(writes, fname);
}

void compress_unitigs(char_view_t& kmers, edge_view_t& kmer_offsets, graph_type glue_action, edge_offset_t k, int iteration){
    edge_offset_t write_size = 0;
    //minus 2 because 0 is not processed, and row_map is one bigger than number of rows
    ordinal_t n = glue_action.row_map.extent(0) - 2;
    edge_view_t next_offsets("next offsets", n + 1);
    Kokkos::parallel_scan("compute offsets", r_policy(1, n+1), KOKKOS_LAMBDA(const ordinal_t u, edge_offset_t& update, const bool final){
        edge_offset_t size = 0;
        bool first = true;
        for(edge_offset_t i = glue_action.row_map(u); i < glue_action.row_map(u + 1); i++){
            ordinal_t f = glue_action.entries(i);
            size += kmer_offsets(f + 1) - kmer_offsets(f);
            if(!first){
                //subtract for overlap of k-mers/unitigs
                size -= (k - 1);
            }
            first = false;
        }
        if(final){
            next_offsets(u - 1) = update;
            if(u == n){
                next_offsets(n) = update + size;
            }
        }
        update += size;
    });
    edge_subview_t write_size_sub = Kokkos::subview(next_offsets, n);
    Kokkos::deep_copy(write_size, write_size_sub);
    char_view_t writes("writes", write_size);
#if defined(HUGE) || defined(LARGE)
    printf("compressed unitigs size: %lu\n", write_size);
#else
    printf("compressed unitigs size: %u\n", write_size);
#endif
    if(iteration > 0){
        Kokkos::parallel_for("move old entries", policy(n, Kokkos::AUTO), KOKKOS_LAMBDA(const member& thread){
            const edge_offset_t u = thread.league_rank() + 1;
            edge_offset_t write_offset = next_offsets(u - 1);
            //not likely to be very many here, about 2 to 7
            bool first = true;
            for(edge_offset_t i = glue_action.row_map(u); i < glue_action.row_map(u + 1); i++){
                ordinal_t f = glue_action.entries(i);
                edge_offset_t start = kmer_offsets(f);
                edge_offset_t end = kmer_offsets(f + 1);
                if(!first){
                    //subtract for overlap of k-mers/unitigs
                    start += (k - 1);
                }
                first = false;
                //this grows larger the deeper we are in the coarsening hierarchy
                Kokkos::parallel_for(Kokkos::TeamThreadRange(thread, start, end), [=](const edge_offset_t j){
                    edge_offset_t offset = write_offset + (j - start);
                    writes(offset) = kmers(j);
                });
                write_offset += (end - start);
            }
        });
    }
    else {
        Kokkos::parallel_for("move old entries", r_policy(0, n), KOKKOS_LAMBDA(const edge_offset_t& x){
            const edge_offset_t u = x + 1;
            edge_offset_t write_offset = next_offsets(u - 1);
            //not likely to be very many here, about 2 to 7
            bool first = true;
            for(edge_offset_t i = glue_action.row_map(u); i < glue_action.row_map(u + 1); i++){
                ordinal_t f = glue_action.entries(i);
                edge_offset_t start = kmer_offsets(f);
                edge_offset_t end = kmer_offsets(f + 1);
                if(!first){
                    //subtract for overlap of k-mers/unitigs
                    start += (k - 1);
                }
                first = false;
                //this grows larger the deeper we are in the coarsening hierarchy
                for(edge_offset_t j = start; j < end; j++) {
                    writes(write_offset++) = kmers(j);
                }
            }
        });
    }
    //Kokkos::parallel_for("move old entries", policy(n, 32), KOKKOS_LAMBDA(const member& thread){
    //    const edge_offset_t u = thread.league_rank() + 1;
    //    edge_offset_t write_offset = next_offsets(u - 1);
    //    //not likely to be very many here, about 2 to 7
    //    Kokkos::parallel_scan(Kokkos::TeamThreadRange(thread, glue_action.row_map(u), glue_action.row_map(u + 1)), [=] (const edge_offset_t i, edge_offset_t& update, const bool final){
    //        ordinal_t f = glue_action.entries(i);
    //        edge_offset_t start = kmer_offsets(f);
    //        edge_offset_t end = kmer_offsets(f + 1);
    //        if(i > glue_action.row_map(u)){
    //            //subtract for overlap of k-mers/unitigs
    //            start += (k - 1);
    //        }
    //        if(final){
    //            edge_offset_t offset = write_offset + update;
    //            //this grows larger the deeper we are in the coarsening hierarchy
    //            for(edge_offset_t j = start; j < end; j++) {
    //                writes(offset) = kmers(j);
    //                offset++;
    //            }
    //        }
    //        update += (end - start);
    //    });
    //});
    kmers = writes;
    kmer_offsets = next_offsets;
}

edge_view_t sizes_init(ordinal_t n, edge_offset_t k){
    edge_view_t sizes("unitig sizes", n + 1);
    Kokkos::parallel_for("init sizes", n + 1, KOKKOS_LAMBDA(const ordinal_t i){
        sizes(i) = k*i;
    });
    return sizes;
}

void compress_unitigs_maximally(char_view_t kmers, std::list<graph_type> glue_actions, edge_offset_t k, std::string fname){
    ordinal_t n = kmers.extent(0) / k;
    edge_view_t sizes = sizes_init(n, k);
    auto glue_iter = glue_actions.begin();
    int iteration = 0;
    while(glue_iter != glue_actions.end()){
        Kokkos::Timer t;
        write_unitigs(kmers, sizes, *glue_iter, fname);
        printf("Write time: %.3f\n", t.seconds());
        t.reset();
        compress_unitigs(kmers, sizes, *glue_iter, k, iteration++);
        printf("Compact time: %.3f\n", t.seconds());
        t.reset();
        glue_iter++;
    }
}

void compress_unitigs_maximally2(char_view_t kmers, std::list<graph_type> glue_actions, edge_offset_t k, std::string fname){
    ordinal_t n = kmers.extent(0) / k;
    auto glue_iter = glue_actions.begin();
    while(glue_iter != glue_actions.end()){
        write_unitigs2(kmers, k, *glue_iter, fname);
        glue_iter++;
    }
}

}

