
//don't technically need scalar_t but we use to template the matrix_t, which we also don't need
//maybe template on the graph_type instead?
template<typename ordinal_t, typename edge_offset_t, typename scalar_t, class Device>
class compact_graph {
public:

    // define internal types
    using exec_space = typename Device::execution_space;
    using mem_space = typename Device::memory_space;
    using matrix_t = KokkosSparse::CrsMatrix<scalar_t, ordinal_t, Device, void, edge_offset_t>;
    using vtx_view_t = Kokkos::View<ordinal_t*, Device>;
    using wgt_view_t = Kokkos::View<scalar_t*, Device>;
    using edge_view_t = Kokkos::View<edge_offset_t*, Device>;
    using edge_subview_t = Kokkos::View<edge_offset_t, Device>;
    using c_edge_subview_t = Kokkos::View<const edge_offset_t, Device>;
    using vtx_subview_t = Kokkos::View<ordinal_t, Device>;
    using graph_type = typename matrix_t::staticcrsgraph_type;
    using policy_t = Kokkos::RangePolicy<exec_space>;
    using team_policy_t = Kokkos::TeamPolicy<exec_space>;
    using member = typename team_policy_t::member_type;
    static constexpr ordinal_t ORD_MAX = std::numeric_limits<ordinal_t>::max();


//collect the fine vertices corresponding to each coarse vertex in the null aggregate
matrix_t collect_outputs(matrix_t glue_old, vtx_view_t nulls){
    edge_offset_t null_size = nulls.extent(0);
    edge_view_t write_sizes("write sizes", null_size + 1);
    Kokkos::parallel_scan("count writes", policy_t(0, null_size), KOKKOS_LAMBDA(const edge_offset_t i, edge_offset_t& update, const bool final){
        ordinal_t u = nulls(i);
        edge_offset_t size = glue_old.graph.row_map(u + 1) - glue_old.graph.row_map(u);
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
    vtx_view_t writes("writes", write_size);
    wgt_view_t orientation("orientation", write_size);
    Kokkos::parallel_for("move writes", team_policy_t(null_size, Kokkos::AUTO), KOKKOS_LAMBDA(const member& thread){
        const edge_offset_t i = thread.league_rank();
        ordinal_t u = nulls(i);
        edge_offset_t write_offset = write_sizes(i);
        Kokkos::parallel_for(Kokkos::TeamThreadRange(thread, glue_old.graph.row_map(u), glue_old.graph.row_map(u + 1)), [=] (const edge_offset_t j){
            edge_offset_t offset = write_offset + (j - glue_old.graph.row_map(u));
            writes(offset) = glue_old.graph.entries(j);
            orientation(offset) = glue_old.values(j);
        });
    });
    graph_type output_g(writes, write_sizes);
    //number of columns is not accurate, but is not used either
    matrix_t out("output matrix", null_size, orientation, output_g);
    return out;
}

//collect the fine vertices corresponding to each coarse vertex in each coarser aggregate
matrix_t collect_unitigs(matrix_t glue_old, matrix_t glue_action){
    edge_offset_t write_size = 0;
    //minus 1 because row_map is one bigger than number of rows
    ordinal_t n = glue_action.numRows();
    edge_view_t next_offsets("next offsets", n + 1);
    Kokkos::parallel_scan("count entries", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t u, edge_offset_t& update, const bool final){
        edge_offset_t size = 0;
        for(edge_offset_t i = glue_action.graph.row_map(u); i < glue_action.graph.row_map(u + 1); i++){
            ordinal_t f = glue_action.graph.entries(i);
            size += glue_old.graph.row_map(f + 1) - glue_old.graph.row_map(f);
        }
        if(final){
            next_offsets(u) = update;
            if(u + 1 == n){
                next_offsets(n) = update + size;
            }
        }
        update += size;
    });
    edge_subview_t write_size_sub = Kokkos::subview(next_offsets, n);
    Kokkos::deep_copy(write_size, write_size_sub);
    vtx_view_t writes("writes", write_size);
    wgt_view_t orientation("orientation", write_size);
    if(glue_old.graph.entries.extent(0) / glue_old.numRows() > 8){
        Kokkos::parallel_for("move old entries", team_policy_t(n, Kokkos::AUTO), KOKKOS_LAMBDA(const member& thread){
            const edge_offset_t u = thread.league_rank();
            edge_offset_t write_offset = next_offsets(u);
            //not likely to be very many here, about 2 to 7
            for(edge_offset_t i = glue_action.graph.row_map(u); i < glue_action.graph.row_map(u + 1); i++){
                ordinal_t f = glue_action.graph.entries(i);
                scalar_t o = glue_action.values(i);
                edge_offset_t start = glue_old.graph.row_map(f);
                edge_offset_t end = glue_old.graph.row_map(f + 1);
                //this grows larger the deeper we are in the coarsening hierarchy
                Kokkos::parallel_for(Kokkos::TeamThreadRange(thread, start, end), [=](const edge_offset_t j){
                    edge_offset_t offset = write_offset + (j - start);
                    if(o == 1){
                        offset = write_offset + (end - 1 - j);
                    }
                    orientation(offset) = glue_old.values(j);
                    writes(offset) = glue_old.graph.entries(j);
                    if(o == 1){
                        orientation(offset) ^= 1;
                    }
                });
                write_offset += (end - start);
            }
        });
    } else {
        Kokkos::parallel_for("move old entries", policy_t(0, n), KOKKOS_LAMBDA(const edge_offset_t u){
            edge_offset_t write_offset = next_offsets(u);
            //not likely to be very many here, about 2 to 7
            for(edge_offset_t i = glue_action.graph.row_map(u); i < glue_action.graph.row_map(u + 1); i++){
                ordinal_t f = glue_action.graph.entries(i);
                scalar_t o = glue_action.values(i);
                edge_offset_t start = glue_old.graph.row_map(f);
                edge_offset_t end = glue_old.graph.row_map(f + 1);
                //this grows larger the deeper we are in the coarsening hierarchy
                if(o == 0){
                    for(edge_offset_t j = start; j < end; j++) {
                        writes(write_offset) = glue_old.graph.entries(j);
                        orientation(write_offset) = glue_old.values(j);
                        write_offset++;
                    }
                } else {
                    for(edge_offset_t j = end - 1; j >= start; j--) {
                        writes(write_offset) = glue_old.graph.entries(j);
                        orientation(write_offset) = glue_old.values(j) ^ 1;
                        write_offset++;
                    }
                }
            }
        });
    }
    graph_type output_g(writes, next_offsets);
    //number of columns is not accurate, but is not used either
    matrix_t out("output matrix", n, orientation, output_g);
    return out;
}

};
