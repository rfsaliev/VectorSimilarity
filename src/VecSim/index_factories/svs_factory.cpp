/* TODO: what kind of copyrght should we add here */

#include "VecSim/index_factories/svs_factory.h"
#include "VecSim/algorithms/svs/svs.h"
#include "VecSim/algorithms/svs/svs_single.h"

namespace SVSFactory{

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


// TODO: can be deduplicated from brute_force_factory.cpp
VecSimIndex *NewIndex(const SVSParams *svsparams, const AbstractIndexInitParams &abstractInitParams) {
	// assume FLOAT32 for now, single
	return new (abstractInitParams.allocator) SVSIndex_Single<float, float>(svsparams, abstractInitParams);
}

VecSimIndex *NewIndex(const VecSimParams *params) {
	const SVSParams *svsParams = &params->algoParams.svsParams;
	AbstractIndexInitParams abstractInitParams = NewAbstractInitParams(params);
	return NewIndex(svsParams, NewAbstractInitParams(params));
}
	
//VecSimIndex *NewIndex(const SVSParams *svsparams) {
//	VecSimParams params = {.algoParams{.svsParams = SVSParams{*svsparams}}};
//	return NewIndex(&params);
//}

size_t EstimateElementSize(const SVSParams *params) {
	return params->dim * VecSimType_sizeof(params->type) + sizeof(labelType) + sizeof(void *);
};

size_t EstimateInitialSize(const SVSParams *params) {
	size_t allocations_overhead = VecSimAllocator::getAllocationOverheadSize();
	size_t est = sizeof(VecSimAllocator) + allocations_overhead;

	// Assume FLOAT32, Single
	est += sizeof(SVSIndex<float, float>);

	return est;
}

} // namespace SVSFactory
