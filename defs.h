#pragma once

#include <limits>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <Kokkos_Sort.hpp>
#include <Kokkos_StaticCrsGraph.hpp>

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
	static constexpr ordinal_t ORD_MAX = std::numeric_limits<ordinal_t>::max();
    typedef uint64_t edge_offset_t;
#elif defined(LARGE)
    typedef uint32_t ordinal_t;
	static constexpr ordinal_t ORD_MAX = std::numeric_limits<ordinal_t>::max();
    typedef uint64_t edge_offset_t;
#else
    typedef uint32_t ordinal_t;
	static constexpr ordinal_t ORD_MAX = std::numeric_limits<ordinal_t>::max();
    typedef uint32_t edge_offset_t;
#endif
    typedef double sgp_real_t;
    typedef edge_offset_t value_t;


    typedef Kokkos::Device<Kokkos::DefaultExecutionSpace, typename Kokkos::DefaultExecutionSpace::memory_space> Device;

    using host_policy = Kokkos::RangePolicy<Kokkos::OpenMP>;

    using coarsener_t = coarse_builder<ordinal_t, edge_offset_t, value_t, Device>;
    using graph_type = typename coarsener_t::graph_type;
    using graph_m = typename graph_type::HostMirror;
    using crosses = coarsener_t::crosses;
    using char_view_t = Kokkos::View<char*>;
    using char_mirror_t = typename char_view_t::HostMirror;
    using edge_view_t = Kokkos::View<edge_offset_t*>;
    using edge_subview_t = Kokkos::View<edge_offset_t, Device>;
    using c_edge_subview_t = Kokkos::View<const edge_offset_t, Device>;
    using edge_mirror_t = typename edge_view_t::HostMirror;
    using vtx_view_t = Kokkos::View<ordinal_t*>;
    using lmin_vt = Kokkos::View<const uint32_t*, Device, Kokkos::MemoryTraits<Kokkos::RandomAccess>>;
    using comp_vt = Kokkos::View<uint32_t*>;
    using hash_vt = Kokkos::View<const uint32_t*, Device, Kokkos::MemoryTraits<Kokkos::RandomAccess>>;
    using comp_mt = typename comp_vt::HostMirror;
    using vtx_subview_t = Kokkos::View<ordinal_t>;
    using vtx_mirror_t = typename vtx_view_t::HostMirror;
    using wgt_view_t = Kokkos::View<value_t*>;
    using wgt_mirror_t = typename wgt_view_t::HostMirror;
    using policy = Kokkos::TeamPolicy<>;
    using r_policy = Kokkos::RangePolicy<>;
    using r_policy_edge = Kokkos::RangePolicy<Kokkos::IndexType<edge_offset_t>, typename Device::execution_space>;
    using rand_view_t = typename Kokkos::View<uint64_t*, Device>;
    using member = typename policy::member_type;
    using pool_t = Kokkos::Random_XorShift64_Pool<>;
    using gen_t = typename pool_t::generator_type;
    using eigenview_t = Kokkos::View<sgp_real_t*>;
    using eigen_mirror_t = typename eigenview_t::HostMirror;
}
