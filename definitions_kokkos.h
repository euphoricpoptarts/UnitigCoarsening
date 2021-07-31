#pragma once

#include <limits>
#include <Kokkos_Core.hpp>
#include <Kokkos_Atomic.hpp>
#include <Kokkos_UnorderedMap.hpp>
#include <Kokkos_Random.hpp>
#include <Kokkos_UniqueToken.hpp>
#include <Kokkos_Sort.hpp>
#include <Kokkos_Functional.hpp>
#include "KokkosSparse_CrsMatrix.hpp"
#include "KokkosSparse_spmv.hpp"
#include "KokkosSparse_spgemm.hpp"
#include "KokkosSparse_spadd.hpp"
#include "KokkosKernels_SparseUtils.hpp"
#include "KokkosKernels_HashmapAccumulator.hpp"
#include "KokkosKernels_Uniform_Initialized_MemoryPool.hpp"

namespace unitig_compact {

#ifdef __cplusplus
#define SGPAR_API 
#endif // __cplusplus

#define SGPAR_USE_ASSERT
#ifdef SGPAR_USE_ASSERT
#ifndef SGPAR_ASSERT
#include <assert.h>
#define SGPAR_ASSERT(expr) assert(expr)
#endif
#else
#define SGPAR_ASSERT(expr) 
#endif


    /**********************************************************
     * Internal
     **********************************************************
     */
#if defined(HUGE)
    typedef uint64_t ordinal_t;
    typedef uint64_t edge_offset_t;
#elif defined(LARGE)
    typedef uint32_t ordinal_t;
    typedef uint64_t edge_offset_t;
#else
    typedef uint32_t ordinal_t;
    typedef uint32_t edge_offset_t;
#endif
    typedef double sgp_real_t;
    typedef edge_offset_t value_t;
	static constexpr ordinal_t ORD_MAX = std::numeric_limits<ordinal_t>::max();
	static constexpr edge_offset_t EDGE_MAX = std::numeric_limits<edge_offset_t>::max();

    typedef Kokkos::Device<Kokkos::DefaultExecutionSpace, typename Kokkos::DefaultExecutionSpace::memory_space> Device;
    using matrix_t = typename KokkosSparse::CrsMatrix<int, ordinal_t, Device, void, edge_offset_t>;
    using host_matrix_t = typename KokkosSparse::CrsMatrix<int, ordinal_t, Kokkos::OpenMP, void, edge_offset_t>;
    using graph_type = typename matrix_t::staticcrsgraph_type;
    using host_graph_t = typename host_matrix_t::staticcrsgraph_type;
    using coarsener_t = coarse_builder<ordinal_t, edge_offset_t, int, Device>;
    using canon_graph = coarsener_t::canon_graph;

    using host_policy = Kokkos::RangePolicy<Kokkos::OpenMP>;

    using char_view_t = Kokkos::View<char*>;
    using char_mirror_t = typename char_view_t::HostMirror;
    using edge_view_t = Kokkos::View<edge_offset_t*>;
    using edge_subview_t = Kokkos::View<edge_offset_t, Device>;
    using c_edge_subview_t = Kokkos::View<const edge_offset_t, Device>;
    using edge_mirror_t = typename edge_view_t::HostMirror;
    using vtx_view_t = Kokkos::View<ordinal_t*>;
    using vtx_mirror_t = typename vtx_view_t::HostMirror;
    using wgt_view_t = Kokkos::View<value_t*>;
    using wgt_mirror_t = typename wgt_view_t::HostMirror;
    using policy = Kokkos::TeamPolicy<>;
    using r_policy = Kokkos::RangePolicy<>;
    using member = typename policy::member_type;
    using pool_t = Kokkos::Random_XorShift64_Pool<>;
    using gen_t = typename pool_t::generator_type;
    using hasher_t = Kokkos::pod_hash<ordinal_t>;
    using eigenview_t = Kokkos::View<sgp_real_t*>;
    using eigen_mirror_t = typename eigenview_t::HostMirror;
}
