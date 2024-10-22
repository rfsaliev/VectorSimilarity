/* TODO: what kind of copyrght should we add here */

#include "VecSim/index_factories/svs_factory.h"
#include "VecSim/algorithms/svs/svs.h"

namespace SVSFactory {

template <typename DataType>
VecSimIndex *NewIndex(const SVSParams *svsParams,
                      const std::shared_ptr<VecSimAllocator>& allocator) {
    /* In fact, there is no need of NewAbstractInitParams for the Vamana/SVS algorithm.
     * However, the established pattern in VecSim is to inherit all index classes
     * from an abstract VecSimIndexAbstract class.
     * The abstract parent, in turn, needs the abstract parameters.
     * TODO: see if you can get rid of the abstrat params.
     */
    switch (svsParams->metric) {
    case VecSimMetric_L2:
        return new (allocator)
            SVSIndex<DataType, svs::distance::DistanceL2>(svsParams, allocator);
    case VecSimMetric_IP:
        return new (allocator)
            SVSIndex<DataType, svs::distance::DistanceIP>(svsParams, allocator);
    case VecSimMetric_Cosine:
        return new (allocator)
            // FIXME(rfsaliev) To be fixed in SVS:
            // is not defined in svs/include/svs/quantization/lvq/vectors.h :
            // template <> struct BiasedDistance<distance::DistanceCosineSimilarity>
            SVSIndex<DataType, svs::distance::DistanceIP>(svsParams, allocator);
    default:
        // If we got here something is wrong.
        assert(false && "Unknown distance metric type");
        return NULL;
    }
}

VecSimIndex *NewIndex(const VecSimParams *params) {
    const SVSParams *svsParams = &params->algoParams.svsParams;
    auto allocator = VecSimAllocator::newVecsimAllocator();

    switch (svsParams->type) {
    case VecSimType_FLOAT32:
        return NewIndex<float>(svsParams, allocator);
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
    // FIXME(rfsaliev): custom allocator for svs::index::MutableVamanaIndex::translator_
    // + sizeof(svs::IDTranslator::external_id_type)
    // + sizeof(svs::IDTranslator::internal_id_type)
    // FIXME(rfsaliev): fix SVS graph construction with custom allocator
    // + size_of_graph_node(labelType)

    return SVSIndexVectorSize(params->type, params->dim);
};

size_t EstimateInitialSize(const SVSParams *params) {
    size_t allocations_overhead = VecSimAllocator::getAllocationOverheadSize();
    size_t est = sizeof(VecSimAllocator) + allocations_overhead;

    // Assume FLOAT32, Single
    // using T = uint32_t;
    est += sizeof(SVSIndex<float, svs::distance::DistanceL2>);

    return est;
}

} // namespace SVSFactory
