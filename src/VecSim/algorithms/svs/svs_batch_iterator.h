/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "VecSim/batch_iterator.h"
#include "VecSim/utils/vec_utils.h"

#include <vector>
#include <limits>
#include <cassert>
#include <algorithm> //nth_element
#include <iostream>
#include <cmath>
#include <functional>
#include <memory>

#include "svs/index/vamana/iterator.h"

template <typename Index, typename DataType>
class SVS_BatchIterator : public VecSimBatchIterator {
private:
    size_t dim;

    using impl_type = svs::index::vamana::BatchIterator<Index, DataType>;
    std::unique_ptr<impl_type> impl_;
    decltype(impl_->begin()) curr_it;

    static std::unique_ptr<impl_type> make_impl(const Index *index, void *query_vector,
                                                VecSimQueryParams *queryParams) {
        const size_t batchSize = queryParams ? queryParams->batchSize : 10;

        // Base search parameters for the iterator schedule.
        // This uses a search window size/capacity of 4.
        auto base_parameters = svs::index::vamana::VamanaSearchParameters{}.buffer_config({4});
        auto schedule = svs::index::vamana::DefaultSchedule{base_parameters, batchSize};
        std::span<const DataType> query{reinterpret_cast<DataType *>(query_vector),
                                        index->dimensions()};
        return std::make_unique<svs::index::vamana::BatchIterator<Index, DataType>>(*index, query,
                                                                                    schedule);
    }

    VecSimQueryReply *getNextResultsImpl(size_t n_res) {
        auto rep = new VecSimQueryReply(this->allocator);
        // TODO(rfsaliev) verify iteration logic:
        for (size_t i = 0; i < n_res; i++) {
            if (curr_it == impl_->end()) {
                impl_->next();
                curr_it = impl_->begin();
                if (impl_->size() == 0) {
                    return rep;
                }
            }
            rep->results.push_back(VecSimQueryResult{curr_it->id(), curr_it->distance()});
            curr_it++;
        }
        return rep;
    }

public:
    SVS_BatchIterator(void *query_vector, Index *index, VecSimQueryParams *queryParams,
                      std::shared_ptr<VecSimAllocator> allocator)
        : VecSimBatchIterator{query_vector, queryParams ? queryParams->timeoutCtx : nullptr,
                              allocator},
          dim{index->dimensions()}, impl_{make_impl(index, query_vector, queryParams)} {
        curr_it = impl_->begin();
    }

    VecSimQueryReply *getNextResults(size_t n_res, VecSimQueryReply_Order order) override {
        auto rep = getNextResultsImpl(n_res);
        this->updateResultsCount(VecSimQueryReply_Len(rep));
        sort_results(rep, order);
        return rep;
    }

    bool isDepleted() override { return curr_it == impl_->end() && impl_->done(); }

    void reset() override {
        std::span<const DataType> query{reinterpret_cast<const DataType *>(this->getQueryBlob()),
                                        this->dim};
        impl_->update(query);
        curr_it = impl_->begin();
    }
};

// Empty index iterator
class NullSVS_BatchIterator : public VecSimBatchIterator {
private:
public:
    NullSVS_BatchIterator(void *query_vector, VecSimQueryParams *queryParams,
                          std::shared_ptr<VecSimAllocator> allocator)
        : VecSimBatchIterator{query_vector, queryParams ? queryParams->timeoutCtx : nullptr,
                              allocator} {}

    VecSimQueryReply *getNextResults(size_t n_res, VecSimQueryReply_Order order) override {
        return new VecSimQueryReply(this->allocator);
    }

    bool isDepleted() override { return true; }

    void reset() override {}
};
