/* TODO: change the copyright here */

/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

/* TODO clean the includes */
#pragma once
#include "VecSim/vec_sim_interface.h"
#include "VecSim/spaces/spaces.h"
#include "VecSim/utils/vecsim_stl.h"
#include "VecSim/index_factories/brute_force_factory.h"
#include "VecSim/spaces/spaces.h"
#include "VecSim/query_result_definitions.h"
#include "VecSim/utils/vec_utils.h"

#include <cstring>
#include <cmath>
#include <memory>
#include <queue>
#include <cassert>
#include <limits>
#include <vector>

#include "svs/index/vamana/dynamic_index.h"

#include "VecSim/algorithms/svs/svs_utils.h"
#include "VecSim/algorithms/svs/svs_batch_iterator.h"

// QUANT_BITS == 0 means no LVQ
#define QUANT_BITS 8

template <typename DataType, size_t QuantBits, class Enable = void>
struct SVSStorageTraits {
    using allocator_type = details::SVSAllocator<DataType>;
    using blocked_type = svs::data::Blocked<allocator_type>;
    using index_storage_type = svs::data::BlockedData<DataType, svs::Dynamic, allocator_type>;

    template <svs::data::ImmutableMemoryDataset Dataset>
    static index_storage_type create_storage(const Dataset &data, size_t block_size,
                                             std::shared_ptr<VecSimAllocator> allocator) {
        const auto dim = data.dimensions();
        const auto size = data.size();
        auto svs_bs = details::SVSBlockSize(block_size, element_size(dim));
        allocator_type data_allocator{std::move(allocator)};
        blocked_type blocked_alloc{{svs_bs}, data_allocator};
        index_storage_type init_data{size, dim, blocked_alloc};
        for (const auto &i : data.eachindex()) {
            init_data.set_datum(i, data.get_datum(i));
        }
        return init_data;
    }

    static constexpr size_t element_size(size_t dims, size_t /*alignment*/ = 0) {
        return dims * sizeof(DataType);
    }
};

// Can be detected and defined via cmake config
#define LVQ_EXISTS 0

#if LVQ_EXISTS
#include "svs/extensions/vamana/lvq.h"
template <typename DataType, size_t QuantBits>
struct SVSStorageTraits<DataType, QuantBits, std::enable_if_t<(QuantBits > 0)>> {
    using allocator_type = details::SVSAllocator<std::byte>;
    using blocked_type = svs::data::Blocked<allocator_type>;
    using index_storage_type =
        svs::quantization::lvq::LVQDataset<QuantBits, 0, svs::Dynamic,
                                           svs::quantization::lvq::Sequential, blocked_type>;

    template <svs::data::ImmutableMemoryDataset Dataset>
    static index_storage_type create_storage(const Dataset &data, size_t block_size,
                                             std::shared_ptr<VecSimAllocator> allocator) {
        const auto dim = data.dimensions();
        auto svs_bs = details::SVSBlockSize(block_size, element_size(dim));

        allocator_type data_allocator{std::move(allocator)};
        blocked_type blocked_alloc{{svs_bs}, data_allocator};

        // FIXME(rfsaliev) svs::quantization::lvq::VectorBias to be fixed to support
        // ConstSimpleDataView here:
        /*
        --- a/include/svs/quantization/lvq/ops.h
        +++ b/include/svs/quantization/lvq/ops.h
        @@ -169,7 +169,7 @@ template <typename T> class ScaleShift {
        struct VectorBias : public DatasetPreOpBase {
            static std::string name() { return "preop-vector-bias"; }

        -    template <typename T> using element_type_t = typename T::element_type;
        +    template <typename T> using element_type_t = std::remove_cv_t<typename T::element_type>;
            using misc_type = std::vector<double>;

            ///
        */
        return index_storage_type::compress(data, blocked_alloc);
    }

    static constexpr size_t element_size(size_t dims, size_t alignment = 0) {
        using primary_type = typename index_storage_type::primary_type;
        using layout_type = typename primary_type::helper_type;
        using layout_dims_type = svs::lib::MaybeStatic<index_storage_type::extent>;
        const auto layout_dims = layout_dims_type{dims};
        return primary_type::compute_data_dimensions(layout_type{layout_dims}, alignment);
    }
};
#endif

template <size_t QuantBits = QUANT_BITS>
constexpr size_t SVSIndexVectorSize(VecSimType data_type, size_t dims, size_t alignment = 0) {
    switch (data_type) {
        case VecSimType_FLOAT32:
            return SVSStorageTraits<float, QuantBits>::element_size(dims, alignment);
        default:
            // If we got here something is wrong.
            assert(false && "Unsupported data type");
            return 0;
    }
}

class SVSIndexBase : public VecSimIndexInterface {
public:
    using VecSimIndexInterface::VecSimIndexInterface;
    virtual int addVectors(const void *vectors_data, const labelType *labels, size_t n) = 0;
};

// TODO(rfsaliev)
//  * wrap vamana_idx into a handler with init()/get() to avoid improper use risk
template <typename DataType, typename DistType, size_t QuantBits = QUANT_BITS>
class SVSIndex : public SVSIndexBase {
protected:
    using data_type = DataType;
    using dist_type = DistType;
    using Base = SVSIndexBase;

    using storage_traits_t = SVSStorageTraits<DataType, QuantBits>;
    using index_storage_type = typename storage_traits_t::index_storage_type;

    // FIXME(rfsaliev): Add SVS graph construction with custom allocator
    using graph_type = svs::graphs::SimpleBlockedGraph<uint32_t>;
    using impl_type =
        svs::index::vamana::MutableVamanaIndex<graph_type, index_storage_type, dist_type>;

    /* Notice: SimpleGraph template can only be instatiatied for std::unsigned_integral type */
    size_t changes_num = 0;
    SVSParams params_;
    std::unique_ptr<impl_type> vamana_idx;

    size_t num_threads() const { return params_.num_threads; }

    static constexpr SVSParams initParams(const SVSParams *hint) {
        // clang-format off
        return SVSParams {
            .type = hint->type,
            .dim = hint->dim,
            .metric = hint->metric,
            .multi = false,
            .initialCapacity = hint->initialCapacity,
            .blockSize = hint->blockSize ? hint->blockSize : DEFAULT_BLOCK_SIZE,

            .alpha = hint->alpha ? hint->alpha : (hint->metric == VecSimMetric_L2 ? 1.2f : 0.9f),
            .graph_max_degree = hint->graph_max_degree ? hint->graph_max_degree : 32,
            .window_size = hint->window_size ? hint->window_size : 64,
            .max_candidate_pool_size = hint->max_candidate_pool_size ? hint->max_candidate_pool_size : 80,
            .prune_to = hint->prune_to ? hint->prune_to : 32,
            .use_full_search_history = hint->use_full_search_history ? hint->use_full_search_history : true,
            .num_threads = hint->num_threads ? hint->num_threads : std::thread::hardware_concurrency()
        };
        // clang-format on
    }

    static svs::index::vamana::VamanaBuildParameters
    MakeVamanaBuildParameters(const SVSParams &params) {
        return {params.alpha,       params.graph_max_degree,
                params.window_size, params.max_candidate_pool_size,
                params.prune_to,    params.use_full_search_history};
    }

    std::unique_ptr<impl_type> makeImpl(const SVSParams &params, impl_type::data_type data,
                                        std::span<const labelType> ids) {
        auto idx = std::make_unique<impl_type>(MakeVamanaBuildParameters(params_), std::move(data),
                                               ids, DistType{}, num_threads());
        auto sp = idx->get_search_parameters();
        sp.buffer_config({idx->get_construction_window_size()});
        idx->set_search_parameters(sp);
        idx->reset_performance_parameters();
        return idx;
    }

    int addVectorsImpl(const DataType *vectors_data, const labelType *labels, size_t n) {
        std::span<const labelType> ids(labels, n);
        auto points = svs::data::ConstSimpleDataView<DataType>{vectors_data, n, params_.dim};

        // construct SVS index for first rows
        if (!vamana_idx) {
            auto bs = params_.blockSize > 0 ? params_.blockSize : DEFAULT_BLOCK_SIZE;
            auto init_data = storage_traits_t::create_storage(points, bs, this->getAllocator());
            this->vamana_idx = makeImpl(params_, std::move(init_data), ids);
            return n;
        }

        std::vector<labelType> entries_to_delete;
        entries_to_delete.reserve(n);
        for (const auto &label : ids) {
            if (get_vamana()->has_id(label)) {
                entries_to_delete.push_back(label);
            }
        }
        get_vamana()->delete_entries(entries_to_delete);

        get_vamana()->add_points(std::move(points), ids);
        return n - entries_to_delete.size();
    }

    impl_type *get_vamana() const {
        // assert(vamana_idx);
        return this->vamana_idx.get();
    }

    void mark_index_update() {
        if (get_vamana() == nullptr)
            return;

        if (indexSize() == 0) {
            this->vamana_idx.reset(nullptr);
            changes_num = 0;
            return;
        }

        // consolidate index if number of changes bigger than 50% of index size
        const float consolidation_threshold = .5f;
        // indexSize() can be 0, (++changes_num) is always > 0
        if (indexSize() / (++changes_num) < 1.f / consolidation_threshold) {
            get_vamana()->consolidate();
            changes_num = 0;
        }
    }

    static float toVecSimDistance(float v) { return details::toVecSimDistance<dist_type>(v); }

    template <typename Idx>
    static VecSimQueryResult makeVecSimQueryResult(const svs::QueryResult<Idx> &result,
                                                   size_t query, size_t neighbor) {
        return details::makeVecSimQueryResult<dist_type, Idx>(result, query, neighbor);
    }

public:
    SVSIndex(const SVSParams *params, std::shared_ptr<VecSimAllocator> allocator)
        : Base{allocator}, changes_num{0}, params_{initParams(params)}, vamana_idx{nullptr} {}

    ~SVSIndex() = default;

    size_t indexSize() const override { return get_vamana() ? get_vamana()->size() : 0; }

    size_t indexCapacity() const override { return indexSize() + 1; }

    size_t indexLabelCount() const override { return indexSize(); }

    VecSimIndexBasicInfo basicInfo() const override {
        VecSimIndexBasicInfo info{.algo = VecSimAlgo_SVS,
                                  .blockSize = params_.blockSize,
                                  .metric = params_.metric,
                                  .type = params_.type,
                                  .isMulti = false,
                                  .dim = params_.dim,
                                  .isTiered = false};
        return info;
    }

    VecSimIndexInfo info() const override {
        VecSimIndexInfo info;
        info.commonInfo = CommonInfo{.basicInfo = this->basicInfo(),
                                     .indexSize = this->indexSize(),
                                     .indexLabelCount = this->indexLabelCount(),
                                     .memory = this->getAllocationSize(),
                                     .lastMode = this->lastMode};
        return info;
    }

    VecSimInfoIterator *infoIterator() const override {
        VecSimIndexInfo info = this->info();
        // For readability. Update this number when needed.
        size_t numberOfInfoFields = 10;
        VecSimInfoIterator *infoIterator =
            new VecSimInfoIterator(numberOfInfoFields, this->allocator);

        infoIterator->addInfoField(VecSim_InfoField{
            .fieldName = VecSimCommonStrings::ALGORITHM_STRING,
            .fieldType = INFOFIELD_STRING,
            .fieldValue = {
                FieldValue{.stringValue = VecSimAlgo_ToString(info.commonInfo.basicInfo.algo)}}});
        this->addCommonInfoToIterator(infoIterator, info.commonInfo);
        infoIterator->addInfoField(VecSim_InfoField{
            .fieldName = VecSimCommonStrings::BLOCK_SIZE_STRING,
            .fieldType = INFOFIELD_UINT64,
            .fieldValue = {FieldValue{.uintegerValue = info.commonInfo.basicInfo.blockSize}}});
        return infoIterator;
    }

    int addVector(const void *vector_data, labelType label, void *auxiliaryCtx = nullptr) override {
        return addVectorsImpl(reinterpret_cast<const DataType *>(vector_data), &label, 1);
    }

    int addVectors(const void *vectors_data, const labelType *labels, size_t n) override {
        return addVectorsImpl(reinterpret_cast<const DataType *>(vectors_data), labels, n);
    }

    int deleteVector(labelType label) override {
        if (get_vamana() == nullptr || !get_vamana()->has_id(label)) {
            return 0;
        }
        get_vamana()->delete_entries(std::span<labelType, 1>{&label, 1});
        this->mark_index_update();
        return 1;
    }

    double getDistanceFrom_Unsafe(labelType label, const void *vector_data) const override {
        if (get_vamana() == nullptr || !get_vamana()->has_id(label)) {
            return std::numeric_limits<double>::quiet_NaN();
        };

        auto index_impl = get_vamana(); //->get_impl()->get_impl();
        auto my_datum = index_impl->get_datum(label);

        auto dist_f = svs::index::vamana::extensions::single_search_setup(
            index_impl->view_data(),
            index_impl->distance_function());

        auto query_datum = std::span{reinterpret_cast<const DataType *>(vector_data), params_.dim};

        svs::distance::maybe_fix_argument(dist_f, query_datum);

        auto dist = svs::distance::compute(dist_f, query_datum, my_datum);
        return toVecSimDistance(dist);
    }

    VecSimQueryReply *topKQuery(const void *queryBlob, size_t k,
                                VecSimQueryParams *queryParams) const override {
        auto rep = new VecSimQueryReply(this->allocator);
        this->lastMode = STANDARD_KNN;
        if (k == 0 || this->indexSize() == 0) {
            return rep;
        }

        auto queries = svs::data::ConstSimpleDataView<DataType>{
            reinterpret_cast<const DataType *>(queryBlob), 1, params_.dim};
        auto result = svs::QueryResult<size_t>{queries.size(), k};
        auto sp = details::joinSearchParams(get_vamana()->get_search_parameters(), queryParams);
        get_vamana()->search(result.view(), queries, sp);

        assert(result.n_queries() == 1);

        for (size_t i = 0; i < result.n_neighbors(); i++) {
            rep->results.push_back(makeVecSimQueryResult(result, 0, i));
        }
        return rep;
    }

    VecSimQueryReply *rangeQuery(const void *queryBlob, double radius,
                                 VecSimQueryParams *queryParams) const {
        auto rep = new VecSimQueryReply(this->allocator);
        this->lastMode = RANGE_QUERY;
        if (radius == 0 || this->indexSize() == 0) {
            return rep;
        }

        auto sp = details::joinSearchParams(get_vamana()->get_search_parameters(), queryParams);
        const size_t batch_size = queryParams && queryParams->batchSize
                                      ? queryParams->batchSize
                                      : sp.buffer_config_.get_search_window_size();
        // Base search parameters for the iterator schedule.
        auto schedule = svs::index::vamana::DefaultSchedule{sp, batch_size};
        std::span<const data_type> query{reinterpret_cast<const data_type *>(queryBlob),
                                         params_.dim};
        svs::index::vamana::BatchIterator<impl_type, data_type> svs_it{*get_vamana(), query,
                                                                       schedule};

        int batch_times = 3;
        bool done = false;
        while (svs_it.size() > 0 && batch_times > 0) {
            for (auto &neighbor : svs_it) {
                if (toVecSimDistance(neighbor.distance()) <= radius) {
                    rep->results.push_back(
                        VecSimQueryResult{neighbor.id(), toVecSimDistance(neighbor.distance())});
                    done = false;
                } else {
                    done = true;
                }
            }
            if (done)
                if (--batch_times == 0)
                    break;
            svs_it.next();
        }
        return rep;
    }

    VecSimBatchIterator *newBatchIterator(const void *queryBlob,
                                          VecSimQueryParams *queryParams) const override {
        auto *queryBlobCopy = this->allocator->allocate(sizeof(DataType) * params_.dim);
        memcpy(queryBlobCopy, queryBlob, params_.dim * sizeof(DataType));
        // Ownership of queryBlobCopy moves to VecSimBatchIterator that will free it at the end.
        if (get_vamana() == nullptr) {
            return new (this->getAllocator())
                NullSVS_BatchIterator(queryBlobCopy, queryParams, this->getAllocator());
        } else {
            return new (this->getAllocator()) SVS_BatchIterator<impl_type, data_type>(
                queryBlobCopy, get_vamana(), queryParams, this->getAllocator());
        }
    }

    bool preferAdHocSearch(size_t subsetSize, size_t k, bool initial_check) const override {
        bool res = true;
        this->lastMode =
            res ? (initial_check ? HYBRID_ADHOC_BF : HYBRID_BATCHES_TO_ADHOC_BF) : HYBRID_BATCHES;
        return res;
    }

    // From VecSimIndexAbstract
private:
    // TODO(rfsaliev) modify/remove below
    size_t alignment() const { return 0; }
    size_t dataSize() const { return params_.dim * sizeof(DataType); }
    mutable VecSearchMode lastMode =
        EMPTY_MODE; // The last search mode in RediSearch (used for debug/testing).
    spaces::normalizeVector_f<DataType> normalize_func =
        spaces::GetNormalizeFunc<DataType>(); // A pointer to a normalization function of specific
                                              // type.

    void setLastSearchMode(VecSearchMode mode) override { this->lastMode = mode; }

    // Adds all common info to the info iterator, besides the block size (currently 8 fields).
    void addCommonInfoToIterator(VecSimInfoIterator *infoIterator, const CommonInfo &info) const {
        infoIterator->addInfoField(VecSim_InfoField{
            .fieldName = VecSimCommonStrings::TYPE_STRING,
            .fieldType = INFOFIELD_STRING,
            .fieldValue = {FieldValue{.stringValue = VecSimType_ToString(info.basicInfo.type)}}});
        infoIterator->addInfoField(
            VecSim_InfoField{.fieldName = VecSimCommonStrings::DIMENSION_STRING,
                             .fieldType = INFOFIELD_UINT64,
                             .fieldValue = {FieldValue{.uintegerValue = info.basicInfo.dim}}});
        infoIterator->addInfoField(
            VecSim_InfoField{.fieldName = VecSimCommonStrings::METRIC_STRING,
                             .fieldType = INFOFIELD_STRING,
                             .fieldValue = {FieldValue{
                                 .stringValue = VecSimMetric_ToString(info.basicInfo.metric)}}});
        infoIterator->addInfoField(
            VecSim_InfoField{.fieldName = VecSimCommonStrings::IS_MULTI_STRING,
                             .fieldType = INFOFIELD_UINT64,
                             .fieldValue = {FieldValue{.uintegerValue = info.basicInfo.isMulti}}});
        infoIterator->addInfoField(
            VecSim_InfoField{.fieldName = VecSimCommonStrings::INDEX_SIZE_STRING,
                             .fieldType = INFOFIELD_UINT64,
                             .fieldValue = {FieldValue{.uintegerValue = info.indexSize}}});
        infoIterator->addInfoField(
            VecSim_InfoField{.fieldName = VecSimCommonStrings::INDEX_LABEL_COUNT_STRING,
                             .fieldType = INFOFIELD_UINT64,
                             .fieldValue = {FieldValue{.uintegerValue = info.indexLabelCount}}});
        infoIterator->addInfoField(
            VecSim_InfoField{.fieldName = VecSimCommonStrings::MEMORY_STRING,
                             .fieldType = INFOFIELD_UINT64,
                             .fieldValue = {FieldValue{.uintegerValue = info.memory}}});
        infoIterator->addInfoField(VecSim_InfoField{
            .fieldName = VecSimCommonStrings::SEARCH_MODE_STRING,
            .fieldType = INFOFIELD_STRING,
            .fieldValue = {FieldValue{.stringValue = VecSimSearchMode_ToString(info.lastMode)}}});
    }

    const void *processBlob(const void *original_blob, void *aligned_mem) const {
        void *processed_blob;
        // if the blob is not aligned, or we need to normalize, we copy it
        if ((this->alignment() && (uintptr_t)original_blob % this->alignment()) ||
            this->params_.metric == VecSimMetric_Cosine) {
            memcpy(aligned_mem, original_blob, this->dataSize());
            processed_blob = aligned_mem;
        } else {
            processed_blob = (void *)original_blob;
        }

        // if the metric is cosine, we need to normalize
        if (this->params_.metric == VecSimMetric_Cosine) {
            // normalize the copy in place
            normalize_func(processed_blob, this->params_.dim);
        }

        return processed_blob;
    }

    virtual int addVectorWrapper(const void *blob, labelType label, void *auxiliaryCtx) override {
        auto aligned_mem =
            this->getAllocator()->allocate_aligned_unique(this->dataSize(), this->alignment());
        const void *processed_blob = processBlob(blob, aligned_mem.get());

        return this->addVector(processed_blob, label, auxiliaryCtx);
    }

    virtual VecSimQueryReply *topKQueryWrapper(const void *queryBlob, size_t k,
                                               VecSimQueryParams *queryParams) const override {
        auto aligned_mem =
            this->getAllocator()->allocate_aligned_unique(this->dataSize(), this->alignment());
        const void *processed_blob = processBlob(queryBlob, aligned_mem.get());

        return this->topKQuery(processed_blob, k, queryParams);
    }

    virtual VecSimQueryReply *rangeQueryWrapper(const void *queryBlob, double radius,
                                                VecSimQueryParams *queryParams,
                                                VecSimQueryReply_Order order) const override {
        auto aligned_mem =
            this->getAllocator()->allocate_aligned_unique(this->dataSize(), this->alignment());
        const void *processed_blob = processBlob(queryBlob, aligned_mem.get());

        return this->rangeQuery(processed_blob, radius, queryParams, order);
    }

    VecSimQueryReply *rangeQuery(const void *queryBlob, double radius,
                                 VecSimQueryParams *queryParams,
                                 VecSimQueryReply_Order order) const override {
        auto results = rangeQuery(queryBlob, radius, queryParams);
        sort_results(results, order);
        return results;
    }

    virtual VecSimBatchIterator *
    newBatchIteratorWrapper(const void *queryBlob, VecSimQueryParams *queryParams) const override {
        auto aligned_mem =
            this->getAllocator()->allocate_aligned_unique(this->dataSize(), this->alignment());
        const void *processed_blob = processBlob(queryBlob, aligned_mem.get());

        return this->newBatchIterator(processed_blob, queryParams);
    }

    void runGC() override {}              // Do nothing, relevant for tiered index only.
    void acquireSharedLocks() override {} // Do nothing, relevant for tiered index only.
    void releaseSharedLocks() override {} // Do nothing, relevant for tiered index only.
};
