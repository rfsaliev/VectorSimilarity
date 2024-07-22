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

template <typename DataType, typename DistType>
VecSimIndex *NewIndex(const SVSParams *params, const AbstractIndexInitParams &abstractInitParams) {
	// assume FLOAT32 for now, single
	return new (abstractInitParams.allocator) SVSIndex_Single<DataType, DistType>(params, abstractInitParams);
}

VecSimIndex *NewIndex(const VecSimParams *params) {
	/* In fact, there is no need of NewAbstractInitParams for the Vamana/SVS algorithm.
	 * However, the established pattern in VecSim is to inherit all index classes
	 * from an abstract VecSimIndexAbstract class.
	 * The abstract parent, in turn, needs the abstract parameters.
	 * TODO: see if you can get rid of the abstrat params.
	 */ 
	const SVSParams *svsParams = &params->algoParams.svsParams;
	AbstractIndexInitParams abstractInitParams = NewAbstractInitParams(params);
	//using T = uint32_t;
	using T = svs::data::BlockedData<float, 10>;
	return NewIndex<T, svs::distance::DistanceL2>(svsParams, NewAbstractInitParams(params));
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
	//using T = uint32_t;
	using T = svs::data::BlockedData<float, 10>;
	est += sizeof(SVSIndex<T, svs::distance::DistanceL2>);

	return est;
}

} // namespace SVSFactory
