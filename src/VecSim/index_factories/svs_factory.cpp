/* TODO: what kind of copyrght should we add here */

#include "VecSim/index_factories/svs_factory.h"
#include "VecSim/algorithms/svs/svs.h"

namespace SVSFactory {

// Can and should be deduplicated from the other factories
static AbstractIndexInitParams NewAbstractInitParams(const VecSimParams *params) {
    const SVSParams *svsParams = &params->algoParams.svsParams;
    AbstractIndexInitParams abstractInitParams = {.allocator =
                                                      VecSimAllocator::newVecsimAllocator(),
                                                  .dim = svsParams->dim,
                                                  .vecType = svsParams->type,
                                                  .metric = svsParams->metric,
                                                  .blockSize = svsParams->blockSize,
                                                  .multi = svsParams->multi,
                                                  .logCtx = params->logCtx};
    return abstractInitParams;
}

template <typename DataType>
VecSimIndex *NewIndex(const SVSParams *svsParams,
                      const AbstractIndexInitParams &abstractInitParams) {
    /* In fact, there is no need of NewAbstractInitParams for the Vamana/SVS algorithm.
     * However, the established pattern in VecSim is to inherit all index classes
     * from an abstract VecSimIndexAbstract class.
     * The abstract parent, in turn, needs the abstract parameters.
     * TODO: see if you can get rid of the abstrat params.
     */
    switch (svsParams->metric) {
    case VecSimMetric_L2:
        return new (abstractInitParams.allocator)
            SVSIndex<DataType, svs::distance::DistanceL2>(svsParams, abstractInitParams);
    case VecSimMetric_IP:
        return new (abstractInitParams.allocator)
            SVSIndex<DataType, svs::distance::DistanceIP>(svsParams, abstractInitParams);
    case VecSimMetric_Cosine:
        return new (abstractInitParams.allocator)
            SVSIndex<DataType, svs::distance::DistanceCosineSimilarity>(svsParams,
                                                                        abstractInitParams);
    default:
        // If we got here something is wrong.
        assert(false && "Unknown distance metric type");
        return NULL;
    }
}

VecSimIndex *NewIndex(const VecSimParams *params) {
    const SVSParams *svsParams = &params->algoParams.svsParams;
    auto abstractInitParams = NewAbstractInitParams(params);

    switch (svsParams->type) {
    case VecSimType_FLOAT32:
        return NewIndex<float>(svsParams, abstractInitParams);
    default:
        // If we got here something is wrong.
        assert(false && "Unsupported data type");
        return NULL;
    };
}

// VecSimIndex *NewIndex(const SVSParams *svsparams) {
//	VecSimParams params = {.algoParams{.svsParams = SVSParams{*svsparams}}};
//	return NewIndex(&params);
// }

size_t EstimateElementSize(const SVSParams *params) {
    return params->dim * VecSimType_sizeof(params->type) + sizeof(labelType) + sizeof(void *);
};

size_t EstimateInitialSize(const SVSParams *params) {
    size_t allocations_overhead = VecSimAllocator::getAllocationOverheadSize();
    size_t est = sizeof(VecSimAllocator) + allocations_overhead;

    // Assume FLOAT32, Single
    // using T = uint32_t;
    using T = svs::data::BlockedData<float, 10>;
    est += sizeof(SVSIndex<T, svs::distance::DistanceL2>);

    return est;
}

} // namespace SVSFactory
