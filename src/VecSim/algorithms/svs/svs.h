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

#include "svs/index/vamana/dynamic_index.h"
#include <vector>

template <typename DataType, typename DistType>
class SVSIndex : public VecSimIndexAbstract<DataType, DistType> {
protected:
    /* Notice: SimpleGraph template can only be instatiatied for std::unsigned_integral type */
    svs::index::vamana::MutableVamanaIndex<svs::graphs::SimpleBlockedGraph<uint32_t>, DataType, DistType> vamana_idx;
    std::vector<DataType> data;
    std::vector<DataType> labels;
    idType count;

public:
//    static auto& getVamanaIdx() {
//	auto vamana_params = svs::index::vamana::VamanaBuildParameters{
//		1,
//		1,
//		1,
//		1,
//		1,
//		true
//	};
//
//	auto data_mutable = svs::data::BlockedData<uint32_t, 10>(0, 0);
//	std::vector<int> initial_indices{};
//	
//	static auto vamana_idx = svs::index::vamana::MutableVamanaIndex(
//			vamana_params,
//			std::move(data_mutable),
//			initial_indices,
//			svs::distance::DistanceL2(),
//			4
//			);
//
//	return vamana_idx;
//    };

    SVSIndex(const SVSParams *params, const AbstractIndexInitParams &abstractInitParams);
    ~SVSIndex() = default;

    size_t indexSize() const override;
    size_t indexCapacity() const override;
    virtual VecSimQueryReply *topKQuery(const void *queryBlob, size_t k,
                                        VecSimQueryParams *queryParams) const override;
    virtual VecSimQueryReply *rangeQuery(const void *queryBlob, double radius,
                                         VecSimQueryParams *queryParams) const override;
    virtual VecSimIndexInfo info() const override;
    virtual VecSimInfoIterator *infoIterator() const override;
    VecSimIndexBasicInfo basicInfo() const override;
    virtual VecSimBatchIterator *newBatchIterator(const void *queryBlob,
                                                  VecSimQueryParams *queryParams) const override;
    bool preferAdHocSearch(size_t subsetSize, size_t k, bool initial_check) const override;

    virtual size_t indexLabelCount() const override;
};

template <typename DataType, typename DistType>
SVSIndex<DataType, DistType>::SVSIndex(
		const SVSParams *params,
		const AbstractIndexInitParams &abstractInitParams)
	: VecSimIndexAbstract<DataType, DistType>(abstractInitParams)
{
	/* TODO: remove the indirection layer for the struct */
	auto vamana_params = svs::index::vamana::VamanaBuildParameters{
		params->alpha,
		params->graph_max_degree,
		params->window_size,
		params->max_candidate_pool_size,
		params->prune_to,
		params->use_full_search_history
	};

	auto data_mutable = svs::data::BlockedData<float, 10>(2, 10);
	std::vector<float> initial_indices{};

	vamana_idx = svs::index::vamana::MutableVamanaIndex(
			vamana_params,
			std::move(data_mutable),
			initial_indices,
			svs::distance::DistanceL2(),
			4
			);
}

template <typename DataType, typename DistType>
size_t SVSIndex<DataType, DistType>::indexSize() const {
    return this->count;
}

template <typename DataType, typename DistType>
VecSimQueryReply *
SVSIndex<DataType, DistType>::rangeQuery(const void *queryBlob, double radius,
                                                VecSimQueryParams *queryParams) const {
    VecSimQueryReply *rep = new VecSimQueryReply(this->allocator);
    return rep;
}

template <typename DataType, typename DistType>
bool SVSIndex<DataType, DistType>::preferAdHocSearch(size_t subsetSize, size_t k,
                                                            bool initial_check) const {
    return true;
}

template <typename DataType, typename DistType>
VecSimIndexInfo SVSIndex<DataType, DistType>::info() const {
    VecSimIndexInfo info;
    info.commonInfo = this->getCommonInfo();
    info.commonInfo.basicInfo.algo = VecSimAlgo_SVS;

    return info;
};

template <typename DataType, typename DistType>
VecSimIndexBasicInfo SVSIndex<DataType, DistType>::basicInfo() const {

    VecSimIndexBasicInfo info = this->getBasicInfo();
    info.algo = VecSimAlgo_SVS;
    info.isTiered = false;
    return info;
}

template <typename DataType, typename DistType>
VecSimBatchIterator *
SVSIndex<DataType, DistType>::newBatchIterator(const void *queryBlob,
                                                      VecSimQueryParams *queryParams) const {
    //auto *queryBlobCopy = this->allocator->allocate(sizeof(DataType) * this->dim);
    //memcpy(queryBlobCopy, queryBlob, this->dim * sizeof(DataType));
    // Ownership of queryBlobCopy moves to BF_BatchIterator that will free it at the end.
    //return newBatchIterator_Instance(queryBlobCopy, queryParams);
    return nullptr;
}

template <typename DataType, typename DistType>
VecSimQueryReply *
SVSIndex<DataType, DistType>::topKQuery(const void *queryBlob, size_t k,
                                               VecSimQueryParams *queryParams) const {
    
   // auto queries = SimpleData(queryBlob);
   // // TODO: convert queryBlob to queries
   // // TODO: convert queryParams to results

   // index.search(queries.view(), k, results.view());
   //     
   auto rep = new VecSimQueryReply(this->allocator);
   return rep;
}

template <typename DataType, typename DistType>
size_t SVSIndex<DataType, DistType>::indexCapacity() const {
    return 1;
}

template <typename DataType, typename DistType>
VecSimInfoIterator *SVSIndex<DataType, DistType>::infoIterator() const {
    // For readability. Update this number when needed.
    size_t numberOfInfoFields = 10;
    VecSimInfoIterator *infoIterator = new VecSimInfoIterator(numberOfInfoFields, this->allocator);
    
    return infoIterator;
}
