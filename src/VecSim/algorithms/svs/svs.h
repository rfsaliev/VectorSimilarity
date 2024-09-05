/* TODO: change the copyright here */

/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

/* TODO clean the includes */
#pragma once
#include "VecSim/utils/data_block.h"
#include "VecSim/vec_sim_index.h"
#include "VecSim/spaces/spaces.h"
#include "VecSim/utils/vecsim_stl.h"
#include "VecSim/utils/vecsim_results_container.h"
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

#include "svs/orchestrators/dynamic_vamana.h"

namespace details {
// Adjust SVS distance computation to VecSim
template <typename Ea, typename Eb, size_t Da, size_t Db>
float computeVecSimDistance(svs::distance::DistanceL2 dist, std::span<Ea, Da> a,
                            std::span<Eb, Db> b) {
    return svs::distance::compute(dist, a, b);
}

template <typename Ea, typename Eb, size_t Da, size_t Db>
float computeVecSimDistance(svs::distance::DistanceIP dist, std::span<Ea, Da> a,
                            std::span<Eb, Db> b) {
    return 1.0f - svs::distance::compute(dist, a, b);
}

template <typename Ea, typename Eb, size_t Da, size_t Db>
float computeVecSimDistance(svs::distance::DistanceCosineSimilarity /*dist*/, std::span<Ea, Da> a,
                            std::span<Eb, Db> b) {
    // VecSim uses IP for Cosine distance
    return computeVecSimDistance(svs::distance::DistanceIP{}, a, b);
}

} // namespace details

// TODO(rfsaliev)
//  * remove VecSimIndexAbstract from inheritance chain
//  * wrap vamana_idx into a handler with init()/get() to avoid improper use risk
template <typename DataType, typename DistType>
class SVSIndex : public VecSimIndexInterface {
protected:
    using index_storage_type = svs::data::BlockedData<DataType>;
    using dist_type = DistType;
    using graph_type = svs::graphs::SimpleBlockedGraph<uint32_t>;
    using impl_type =
        svs::index::vamana::MutableVamanaIndex<graph_type, index_storage_type, dist_type>;

    /* Notice: SimpleGraph template can only be instatiatied for std::unsigned_integral type */
    SVSParams params_;
    std::unique_ptr<impl_type> vamana_idx;

    // TODO(rfsaliev) move to params
    static size_t num_threads() { return 4; }

    static svs::index::vamana::VamanaBuildParameters
    MakeVamanaBuildParameters(const SVSParams &params) {
        return {params.alpha,       params.graph_max_degree,
                params.window_size, params.max_candidate_pool_size,
                params.prune_to,    params.use_full_search_history};
    }

    int addVectorImpl(const DataType *vector_data, labelType label) {
        std::vector<labelType> ids{label};

        // construct SVS index for first row
        if (!vamana_idx) {
            svs::data::BlockedData<DataType> init_data{1, params_.dim};
            auto dst = init_data.get_datum(0);
            std::copy(vector_data, vector_data + params_.dim, dst.begin());

            vamana_idx = std::make_unique<impl_type>(MakeVamanaBuildParameters(params_), init_data, ids,
                                                    DistType{}, num_threads());
            return 1;
        }

        int ret = 1;

        if (get_vamana()->has_id(label)) {
            get_vamana()->delete_entries(ids);
            ret = 0;
        }

        auto points = svs::data::ConstSimpleDataView<DataType>{vector_data, 1, params_.dim};
        get_vamana()->add_points(std::move(points), ids);
        return ret;
    }


    impl_type *get_vamana() const {
        assert(vamana_idx);
        return this->vamana_idx.get();
    }

public:
    SVSIndex(const SVSParams *params, std::shared_ptr<VecSimAllocator> allocator)
        : VecSimIndexInterface{std::move(allocator)}, params_{*params},
          vamana_idx{nullptr} {}

    ~SVSIndex() = default;

    size_t indexSize() const override { return vamana_idx ? get_vamana()->size() : 0; }

    size_t indexCapacity() const override { return indexSize() + 1; }

    size_t indexLabelCount() const override { return indexSize(); }


    VecSimIndexBasicInfo basicInfo() const override {
        VecSimIndexBasicInfo info {
            .algo = VecSimAlgo_SVS,
            .blockSize = 1,
            .metric = params_.metric,
            .type = params_.type,
            .isMulti = false,
            .dim = params_.dim,
            .isTiered = false
        };
        return info;
    }

    VecSimIndexInfo info() const override {
        VecSimIndexInfo info;
        info.commonInfo = CommonInfo{
            .basicInfo = this->basicInfo(),
            .indexSize = this->indexSize(),
            .indexLabelCount = this->indexLabelCount(),
            .memory = this->getAllocationSize(),
            .lastMode = this->lastMode
        };
        return info;
    }

    VecSimInfoIterator *infoIterator() const override {
        VecSimIndexInfo info = this->info();
        // For readability. Update this number when needed.
        size_t numberOfInfoFields = 10;
        VecSimInfoIterator *infoIterator = new VecSimInfoIterator(numberOfInfoFields, this->allocator);

        infoIterator->addInfoField(
            VecSim_InfoField{.fieldName = VecSimCommonStrings::ALGORITHM_STRING,
                            .fieldType = INFOFIELD_STRING,
                            .fieldValue = {FieldValue{
                                .stringValue = VecSimAlgo_ToString(info.commonInfo.basicInfo.algo)}}});
        this->addCommonInfoToIterator(infoIterator, info.commonInfo);
        infoIterator->addInfoField(VecSim_InfoField{
            .fieldName = VecSimCommonStrings::BLOCK_SIZE_STRING,
            .fieldType = INFOFIELD_UINT64,
            .fieldValue = {FieldValue{.uintegerValue = info.commonInfo.basicInfo.blockSize}}});
        return infoIterator;

    }

    int addVector(const void *vector_data, labelType label, void *auxiliaryCtx = nullptr) override {
        return addVectorImpl(reinterpret_cast<const DataType *>(vector_data), label);
    }

    int deleteVector(labelType label) override {
        if (!get_vamana()->has_id(label)) {
            return 0;
        }
        get_vamana()->delete_entries(std::span<labelType, 1>{&label, 1});
        return 1;
    }

    double getDistanceFrom_Unsafe(labelType label, const void *vector_data) const override {
        if (!get_vamana()->has_id(label)) {
            return std::numeric_limits<double>::quiet_NaN();
        };

        auto index_impl = get_vamana(); //->get_impl()->get_impl();
        auto my_datum = index_impl->get_datum(label);
        dist_type dist_f = index_impl->distance_function();

        return details::computeVecSimDistance(
            dist_f, std::span{reinterpret_cast<const DataType *>(vector_data), params_.dim},
            my_datum);
    }

    VecSimQueryReply *topKQuery(const void *queryBlob, size_t k,
                                VecSimQueryParams *queryParams) const override  {
        auto queries = svs::data::ConstSimpleDataView<DataType>{
            reinterpret_cast<const DataType *>(queryBlob), 1, params_.dim};
        auto result = svs::QueryResult<size_t>{queries.size(), k};
        auto sp = get_vamana()->get_search_parameters();
        get_vamana()->search(result.view(), queries, sp);

        assert(result.n_queries() == 1);

        auto rep = new VecSimQueryReply(this->allocator);
        for (size_t i = 0; i < result.n_neighbors(); i++) {
            rep->results.push_back(VecSimQueryResult{result.index(0, i), result.distance(0, i)});
        }
        return rep;
    }

    VecSimQueryReply *rangeQuery(const void *queryBlob, double radius,
                                 VecSimQueryParams *queryParams) const {
        return nullptr;
    }

    VecSimBatchIterator *newBatchIterator(const void *queryBlob,
                                          VecSimQueryParams *queryParams) const override {
        return nullptr;
    }

    bool preferAdHocSearch(size_t subsetSize, size_t k, bool initial_check) const override {
        return true;
    }

    void fitMemory() override {};


// From VecSimIndexAbstract
private:
    // TODO(rfsaliev) modify/remove below
    size_t alignment() const { return 0; }
    size_t dataSize() const { return params_.dim * sizeof(DataType); }
    mutable VecSearchMode lastMode = EMPTY_MODE; // The last search mode in RediSearch (used for debug/testing).
    spaces::normalizeVector_f<DataType>
        normalize_func = spaces::GetNormalizeFunc<DataType>(); // A pointer to a normalization function of specific type.

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

