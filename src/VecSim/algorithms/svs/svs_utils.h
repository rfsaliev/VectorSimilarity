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

} // namespace details
