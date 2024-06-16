/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "svs.h"

template <typename DataType, typename DistType>
class SVSIndex_Single : public SVSIndex<DataType, DistType> {
public:
    SVSIndex_Single(const SVSParams *params, const AbstractIndexInitParams &abstractInitParams);
    ~SVSIndex_Single() = default;

    int addVector(const void *vector_data, labelType label, void *auxiliaryCtx = nullptr) override;
    int deleteVector(labelType label) override;
    double getDistanceFrom_Unsafe(labelType label, const void *vector_data) const override;
    inline size_t indexLabelCount() const override { return this->count; }
};

template <typename DataType, typename DistType>
SVSIndex_Single<DataType, DistType>::SVSIndex_Single(const SVSParams *params, const AbstractIndexInitParams &abstractInitParams) :
	SVSIndex<DataType, DistType>(params, abstractInitParams)
{
}

template <typename DataType, typename DistType>
double
SVSIndex_Single<DataType, DistType>::getDistanceFrom_Unsafe(labelType label,
                                                                   const void *vector_data) const {
	return 1;
}

template <typename DataType, typename DistType>
int SVSIndex_Single<DataType, DistType>::addVector(const void *vector_data, labelType label,
                                                          void *auxiliaryCtx) {
    return 1;
}

template <typename DataType, typename DistType>
int SVSIndex_Single<DataType, DistType>::deleteVector(labelType label) {
    return 1;
}
