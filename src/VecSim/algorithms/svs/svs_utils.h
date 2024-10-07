/* TODO: change the copyright here */

/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

/* TODO clean the includes */
#pragma once
#include "VecSim/query_results.h"

#include "svs/core/distance.h"
#include "svs/core/query_result.h"
#include "svs/core/logging.h"
#include "spdlog/sinks/callback_sink.h"

namespace details {
template <typename DistType>
float toVecSimDistance(float);

template <>
inline float toVecSimDistance<svs::distance::DistanceL2>(float v) {
    return v;
}

template <>
inline float toVecSimDistance<svs::distance::DistanceIP>(float v) {
    return 1.f - v;
}

template <>
inline float toVecSimDistance<svs::distance::DistanceCosineSimilarity>(float v) {
    return 1.f - v;
}

template <typename DistType, typename Idx>
VecSimQueryResult makeVecSimQueryResult(const svs::QueryResult<Idx> &result, size_t query,
                                        size_t neighbor) {
    return VecSimQueryResult{result.index(query, neighbor),
                             toVecSimDistance<DistType>(result.distance(query, neighbor))};
}

template <typename Ea, typename Eb, size_t Da, size_t Db>
float computeVecSimDistance(svs::distance::DistanceL2 dist, std::span<Ea, Da> a,
                            std::span<Eb, Db> b) {
    return toVecSimDistance<svs::distance::DistanceL2>(svs::distance::compute(dist, a, b));
}

template <typename Ea, typename Eb, size_t Da, size_t Db>
float computeVecSimDistance(svs::distance::DistanceIP dist, std::span<Ea, Da> a,
                            std::span<Eb, Db> b) {
    return toVecSimDistance<svs::distance::DistanceIP>(svs::distance::compute(dist, a, b));
}

template <typename Ea, typename Eb, size_t Da, size_t Db>
float computeVecSimDistance(svs::distance::DistanceCosineSimilarity /*dist*/, std::span<Ea, Da> a,
                            std::span<Eb, Db> b) {
    // VecSim uses IP for Cosine distance
    return computeVecSimDistance(svs::distance::DistanceIP{}, a, b);
}

template <typename T>
struct SVSAllocator {
private:
    std::shared_ptr<VecSimAllocator> allocator_;

public:
    // Type Aliases
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using propagate_on_container_move_assignment = std::true_type;

    // Constructor
    SVSAllocator(std::shared_ptr<VecSimAllocator> vs_allocator)
        : allocator_{std::move(vs_allocator)} {}

    // Construct from another value type allocator.

    // Allocation and Deallocation.
    [[nodiscard]] constexpr value_type *allocate(std::size_t n) {
        return static_cast<value_type *>(allocator_->allocate_aligned(n * sizeof(T), alignof(T)));
    }

    constexpr void deallocate(value_type *ptr, size_t count) noexcept {
        allocator_->deallocate(ptr, count * sizeof(T));
    }

    // Intercept zero-argument construction to do default initialization.
    // template <typename U>
    // void construct(U* p) noexcept(std::is_nothrow_default_constructible_v<U>) {
    //     ::new (static_cast<void*>(p)) U;
    // }
};

inline svs::index::vamana::VamanaSearchParameters
joinSearchParams(svs::index::vamana::VamanaSearchParameters &&sp,
                 const VecSimQueryParams *queryParams) {
    if (queryParams == nullptr) {
        return std::move(sp);
    }

    auto &rt_params = queryParams->svsRuntimeParams;
    if (rt_params.windowSize > 0) {
        sp.buffer_config({rt_params.windowSize});
    }
    switch (rt_params.visitedSet) {
    case VISITED_SET_ENABLE:
        sp.search_buffer_visited_set(true);
        break;
    case VISITED_SET_DISABLE:
        sp.search_buffer_visited_set(false);
        break;
    default:
        break;
    }
    return std::move(sp);
}

} // namespace details
