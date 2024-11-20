/* TODO: what kind of copyrght should we add here */

#include "VecSim/index_factories/svs_factory.h"
#include "VecSim/algorithms/svs/svs.h"

namespace SVSFactory {

namespace {
bool FactoryLog(void *ctx, const char *lvl, const char *msg) {
    if (!VecSimIndexInterface::logCallback) {
        return false;
    }
    VecSimIndexInterface::logCallback(ctx, lvl, msg);
    return true;
}

template <typename DataType, size_t QuantBits>
struct is_lvq_compatible : public std::false_type {};

template <size_t QuantBits>
struct is_lvq_compatible<float, QuantBits> : public std::true_type {};

template <typename DataType>
struct is_lvq_compatible<DataType, 0> : public std::true_type {};

template <>
struct is_lvq_compatible<float, 0> : public std::true_type {};

template <typename MetricType, typename DataType, size_t QuantBits,
          std::enable_if_t<is_lvq_compatible<DataType, QuantBits>::value, bool> = true>
VecSimIndex *NewIndexImpl(const VecSimParams *params) {
    auto allocator = VecSimAllocator::newVecsimAllocator();
    return new (allocator) SVSIndex<DataType, MetricType, QuantBits>(params, allocator);
}

template <typename MetricType, typename DataType>
VecSimIndex *NewIndexImplLVQ(const VecSimParams *params) {
    switch (params->algoParams.svsParams.quantBits) {
    case 0:
        return NewIndexImpl<MetricType, DataType, 0>(params);
    case 8:
        return NewIndexImpl<MetricType, DataType, 8>(params);
    case 4:
        return NewIndexImpl<MetricType, DataType, 4>(params);
    default:
        // If we got here something is wrong.
        FactoryLog(params->logCtx, VecSimCommonStrings::LOG_WARNING_STRING,
                   "SVSIndex: Unsupported quantization mode");
        return NULL;
    }
}

template <typename MetricType>
VecSimIndex *NewIndexImpl(const VecSimParams *params) {
    assert(params && params->algo == VecSimAlgo_SVS);
    switch (params->algoParams.svsParams.type) {
    case VecSimType_FLOAT32:
        return NewIndexImplLVQ<MetricType, float>(params);
    case VecSimType_FLOAT16:
        // FIXME(rfsaliev) To be fixed in SVS:
        // Float16 + LVQ is not supported
        if (params->algoParams.svsParams.quantBits > 0) {
            FactoryLog(params->logCtx, VecSimCommonStrings::LOG_WARNING_STRING,
                       "SVSIndex: LVQ+FLOAT16 is not supported");
            return NULL;
        }
        return NewIndexImpl<MetricType, svs::Float16, 0>(params);
    default:
        // If we got here something is wrong.
        FactoryLog(params->logCtx, VecSimCommonStrings::LOG_WARNING_STRING,
                   "SVSIndex: Unsupported data type");
        return NULL;
    }
}

VecSimIndex *NewIndexImpl(const VecSimParams *params) {
    switch (params->algoParams.svsParams.metric) {
    case VecSimMetric_L2:
        return NewIndexImpl<svs::distance::DistanceL2>(params);
    case VecSimMetric_IP:
        return NewIndexImpl<svs::distance::DistanceIP>(params);
    case VecSimMetric_Cosine:
        // FIXME(rfsaliev) To be fixed in SVS:
        // is not defined in svs/include/svs/quantization/lvq/vectors.h :
        // template <> struct BiasedDistance<distance::DistanceCosineSimilarity>
        return NewIndexImpl<svs::distance::DistanceIP>(params);
    default:
        // If we got here something is wrong.
        FactoryLog(params->logCtx, VecSimCommonStrings::LOG_WARNING_STRING,
                   "SVSIndex: Unknown distance metric type");
        return NULL;
    }
}

template <typename DataType, size_t QuantBits>
constexpr size_t SVSIndexVectorSize(size_t dims, size_t alignment = 0) {
    return SVSStorageTraits<DataType, QuantBits>::element_size(dims, alignment);
}

template <typename DataType>
size_t SVSIndexVectorSize(size_t quant_bits, size_t dims, size_t alignment = 0) {
    switch (quant_bits) {
    case 0:
        return SVSIndexVectorSize<DataType, 0>(dims, alignment);
    case 8:
        return SVSIndexVectorSize<DataType, 8>(dims, alignment);
    case 4:
        return SVSIndexVectorSize<DataType, 4>(dims, alignment);
    default:
        // If we got here something is wrong.
        assert(false && "Unsupported quantization mode");
        return 0;
    }
}

size_t SVSIndexVectorSize(VecSimType data_type, size_t quant_bits, size_t dims,
                          size_t alignment = 0) {
    switch (data_type) {
    case VecSimType_FLOAT32:
        return SVSIndexVectorSize<float>(quant_bits, dims, alignment);
    default:
        // If we got here something is wrong.
        assert(false && "Unsupported data type");
        return 0;
    }
}
} // namespace

VecSimIndex *NewIndex(const VecSimParams *params) {
    FactoryLog(params->logCtx, VecSimCommonStrings::LOG_NOTICE_STRING, "Creating index: SVS");
    return NewIndexImpl(params);
}

size_t EstimateElementSize(const SVSParams *params) {
    // FIXME(rfsaliev): custom allocator for svs::index::MutableVamanaIndex::translator_
    // + sizeof(svs::IDTranslator::external_id_type)
    // + sizeof(svs::IDTranslator::internal_id_type)
    using graph_idx_type = uint32_t;
    const auto graph_node_size =
        SVSGraphBuilder<graph_idx_type>::element_size(params->graph_max_degree);
    const auto vector_size = SVSIndexVectorSize(params->type, params->quantBits, params->dim);

    return vector_size + graph_node_size;
}

size_t EstimateInitialSize(const SVSParams *params) {
    size_t allocations_overhead = VecSimAllocator::getAllocationOverheadSize();
    size_t est = sizeof(VecSimAllocator) + allocations_overhead;

    // Assume FLOAT32
    // Assume quantBits>0 cases have same sizes
    size_t index_size = (params->quantBits == 0)
                            ? sizeof(SVSIndex<float, svs::distance::DistanceL2, 0>)
                            : sizeof(SVSIndex<float, svs::distance::DistanceL2, 8>);

    est += index_size;

    return est;
}

} // namespace SVSFactory
