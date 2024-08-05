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
template <svs::lib::TypeList QueryTypes, typename... Args>
std::unique_ptr<svs::DynamicVamana> make_dynamic_vamana_ptr(Args &&...args) {
    using Impl = decltype(svs::index::vamana::MutableVamanaIndex{std::forward<Args>(args)...});
    return std::make_unique<svs::DynamicVamana>(
        std::make_unique<svs::DynamicVamanaImpl<QueryTypes, Impl>>(std::forward<Args>(args)...));
}

} // namespace details

// TODO(rfsaliev)
//  * remove VecSimIndexAbstract from inheritance chain
//  * wrap vamana_idx into a handler with init()/get() to avoid improper use risk
template <typename DataType, typename DistType>
class SVSIndex : public VecSimIndexAbstract<DataType, DistType> {
protected:
    /* Notice: SimpleGraph template can only be instatiatied for std::unsigned_integral type */
    SVSParams params_;
    std::unique_ptr<svs::DynamicVamana> vamana_idx;
    // TODO(rfsaliev) move to params
    static size_t num_threads() { return 4; }

    static svs::index::vamana::VamanaBuildParameters
    MakeVamanaBuildParameters(const SVSParams &params) {
        return {params.alpha,       params.graph_max_degree,
                params.window_size, params.max_candidate_pool_size,
                params.prune_to,    params.use_full_search_history};
    }

    int addVectorImpl(const DataType *vector_data, labelType label);

    svs::DynamicVamana *get_vamana() const {
        assert(vamana_idx);
        return this->vamana_idx.get();
    }

public:
    SVSIndex(const SVSParams *params, const AbstractIndexInitParams &abstractInitParams)
        : VecSimIndexAbstract<DataType, DistType>(abstractInitParams), params_{*params},
          vamana_idx{nullptr} {}

    ~SVSIndex() = default;

    size_t indexSize() const override { return vamana_idx ? get_vamana()->size() : 0; }

    size_t indexCapacity() const override { return indexSize() + 1; }

    size_t indexLabelCount() const override { return indexSize(); }

    VecSimIndexInfo info() const override {
        VecSimIndexInfo info;
        info.commonInfo = this->getCommonInfo();
        info.commonInfo.basicInfo.algo = VecSimAlgo_SVS;

        return info;
    }

    VecSimInfoIterator *infoIterator() const override {
        // For readability. Update this number when needed.
        size_t numberOfInfoFields = 10;
        VecSimInfoIterator *infoIterator =
            new VecSimInfoIterator(numberOfInfoFields, this->allocator);
        return infoIterator;
    }

    VecSimIndexBasicInfo basicInfo() const override {
        VecSimIndexBasicInfo info = this->getBasicInfo();
        info.algo = VecSimAlgo_SVS;
        info.isTiered = false;
        return info;
    }

    int addVector(const void *vector_data, labelType label, void *auxiliaryCtx = nullptr) override {
        return addVectorImpl(reinterpret_cast<const DataType *>(vector_data), label);
    }

    int deleteVector(labelType label) override {
        if (!get_vamana()->has_id(label)) {
            return 0;
        }
        get_vamana()->delete_points({&label, 1});
        return 1;
    }

    double getDistanceFrom_Unsafe(labelType label, const void *vector_data) const override {
        return -1;
    }

    VecSimQueryReply *topKQuery(const void *queryBlob, size_t k,
                                VecSimQueryParams *queryParams) const override;
    VecSimQueryReply *rangeQuery(const void *queryBlob, double radius,
                                 VecSimQueryParams *queryParams) const override {
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
};

template <typename DataType, typename DistType>
int SVSIndex<DataType, DistType>::addVectorImpl(const DataType *vector_data, labelType label) {
    std::vector<labelType> ids{label};

    // construct SVS index for first row
    if (!vamana_idx) {
        svs::data::BlockedData<DataType> init_data{1, params_.dim};
        auto dst = init_data.get_datum(0);
        std::copy(vector_data, vector_data + params_.dim, dst.begin());

        vamana_idx =
            std::move(details::make_dynamic_vamana_ptr<svs::manager::as_typelist<DataType>>(
                MakeVamanaBuildParameters(params_), init_data, ids, svs::distance::DistanceL2(),
                num_threads()));
        return 1;
    }

    int ret = 1;

    if (get_vamana()->has_id(label)) {
        get_vamana()->delete_points(ids);
        ret = 0;
    }

    auto points = svs::data::ConstSimpleDataView<DataType>{vector_data, 1, params_.dim};
    get_vamana()->add_points(std::move(points), ids);
    return ret;
}

template <typename DataType, typename DistType>
VecSimQueryReply *SVSIndex<DataType, DistType>::topKQuery(const void *queryBlob, size_t k,
                                                          VecSimQueryParams *queryParams) const {
    auto queries = svs::data::ConstSimpleDataView<DataType>{
        reinterpret_cast<const DataType *>(queryBlob), 1, params_.dim};
    svs::QueryResult<size_t> result = get_vamana()->search(queries, k);

    assert(result.n_queries() == 1);

    auto rep = new VecSimQueryReply(this->allocator);
    for (size_t i = 0; i < result.n_neighbors(); i++) {
        rep->results.push_back(VecSimQueryResult{result.index(0, i), result.distance(0, i)});
    }
    return rep;
}
