/* TODO: change the copyright here */

/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

/* TODO clean the includes */
#pragma once
#include "VecSim/vec_sim_index.h"
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
#include "VecSim/algorithms/svs/svs_extensions.h"

struct SVSIndexBase {
    virtual ~SVSIndexBase() = default;
    virtual int addVectors(const void *vectors_data, const labelType *labels, size_t n) = 0;
};

// TODO(rfsaliev)
//  * wrap vamana_idx into a handler with init()/get() to avoid improper use risk
template <typename DataType, typename DistType, size_t QuantBits>
class SVSIndex : public VecSimIndexAbstract<DataType, float>, public SVSIndexBase {
protected:
    using data_type = DataType;
    using dist_type = DistType;
    using Base = VecSimIndexAbstract<DataType, DataType>;

    using storage_traits_t = SVSStorageTraits<DataType, QuantBits>;
    using index_storage_type = typename storage_traits_t::index_storage_type;

    using graph_builder_t = SVSGraphBuilder<uint32_t>;
    using graph_type = typename graph_builder_t::graph_type;

    using impl_type =
        svs::index::vamana::MutableVamanaIndex<graph_type, index_storage_type, dist_type>;

    /* Notice: SimpleGraph template can only be instatiatied for std::unsigned_integral type */
    size_t changes_num = 0;
    SVSParams params_;
    std::unique_ptr<impl_type> vamana_idx;

    size_t num_threads() const { return params_.num_threads; }

    static constexpr SVSParams initParams(const SVSParams &hint) {
        // TODO(rfsaliev) evaluate optimal default parameters
        // current assumption:
        // * graph_max_degree (64): =~ HNSW_M * 2; may be 63 for alignment?
        // * construction_window_size (250): =~ HNSW_EF_CONSTRUCTION
        // * max_candiate_pool_size (750): = windos_size_construction * 3
        // * prune_to (60): < graph_max_degree, optimal = graph_max_degree - 4
        // * num_threads: = CPU cores per socket
        // * search_window_size: 10 =~ HNSW_EF_RUNTIME
        // clang-format off
        #define GET_WITH_DEFAULT(v, d) ((v)?(v):(d))
        const auto construction_window_size = GET_WITH_DEFAULT(hint.construction_window_size, 250);
        const auto graph_degree = GET_WITH_DEFAULT(hint.graph_max_degree, 64);
        return SVSParams {
            .type = hint.type,
            .dim = hint.dim,
            .metric = hint.metric,
            .multi = false,
            .initialCapacity = hint.initialCapacity,
            .blockSize = hint.blockSize ? hint.blockSize : DEFAULT_BLOCK_SIZE,

            .alpha = GET_WITH_DEFAULT(hint.alpha, (hint.metric == VecSimMetric_L2 ? 1.2f : 0.9f)),
            .graph_max_degree = graph_degree,
            .construction_window_size = construction_window_size,
            .max_candidate_pool_size = GET_WITH_DEFAULT(hint.max_candidate_pool_size, construction_window_size * 3),
            .prune_to = GET_WITH_DEFAULT(hint.prune_to, graph_degree - 4),
            .use_search_history = hint.use_search_history != VecSimOption_DEFAULT ? hint.use_search_history : VecSimOption_ENABLE,
            .num_threads = GET_WITH_DEFAULT(hint.num_threads, std::thread::hardware_concurrency()),
            .search_window_size = GET_WITH_DEFAULT(hint.search_window_size, 10)
        };
        #undef GET_WITH_DEFAULT
        // clang-format on
    }

    static AbstractIndexInitParams InitBaseParams(const VecSimParams *params,
                                                  std::shared_ptr<VecSimAllocator> allocator) {
        assert(params && params->algo == VecSimAlgo_SVS);
        auto &svsParams = params->algoParams.svsParams;
        return {.allocator = std::move(allocator),
                .dim = svsParams.dim,
                .vecType = svsParams.type,
                .metric = svsParams.metric,
                .blockSize = svsParams.blockSize,
                .multi = false,
                .logCtx = params->logCtx};
    }

    static svs::index::vamana::VamanaBuildParameters
    MakeVamanaBuildParameters(const SVSParams &params) {
        return {params.alpha,
                params.graph_max_degree,
                params.construction_window_size,
                params.max_candidate_pool_size,
                params.prune_to,
                params.use_search_history != VecSimOption_DISABLE};
    }

    std::unique_ptr<impl_type> makeImpl(const SVSParams &params, impl_type::data_type data,
                                        std::span<const labelType> ids) {
        svs::threads::NativeThreadPool threadpool{num_threads()};
        // Compute the entry point.
        auto entry_point = svs::index::vamana::extensions::compute_entry_point(data, threadpool);

        // Perform graph construction.
        auto distance = DistType{};
        auto parameters = MakeVamanaBuildParameters(params_);

        auto bs = params_.blockSize > 0 ? params_.blockSize : DEFAULT_BLOCK_SIZE;
        auto graph = graph_builder_t::build_graph(parameters, data, distance, threadpool,
                                                  entry_point, bs, this->getAllocator());

        auto idx = std::make_unique<impl_type>(std::move(graph), std::move(data), entry_point,
                                               std::move(distance), ids, std::move(threadpool));

        // Set MutableIndex build parameters
        idx->set_construction_window_size(parameters.window_size);
        idx->set_max_candidates(parameters.max_candidate_pool_size);
        idx->set_prune_to(parameters.prune_to);
        idx->set_alpha(parameters.alpha);
        idx->set_full_search_history(parameters.use_full_search_history);

        // Configure default search parameters
        auto sp = idx->get_search_parameters();
        sp.buffer_config({params_.search_window_size});
        idx->set_search_parameters(sp);
        idx->reset_performance_parameters();
        return idx;
    }

    int addVectorsImpl(const DataType *vectors_data, const labelType *labels, size_t n) {
        std::span<const labelType> ids(labels, n);
        // FIXME(rfsaliev) const_cast below workarounds LVQ VectorBias definition issue
        // explained in svs_extensions.h
        auto remove_const_vectors_data = const_cast<DataType *>(vectors_data);
        auto points =
            svs::data::SimpleDataView<DataType>{remove_const_vectors_data, n, params_.dim};

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
    SVSIndex(const VecSimParams *params, std::shared_ptr<VecSimAllocator> allocator)
        : Base{InitBaseParams(params, std::move(allocator))}, changes_num{0},
          params_{initParams(params->algoParams.svsParams)}, vamana_idx{nullptr} {}

    ~SVSIndex() = default;

    size_t indexSize() const override { return get_vamana() ? get_vamana()->size() : 0; }

    size_t indexCapacity() const override { return indexSize() + 1; }

    size_t indexLabelCount() const override { return indexSize(); }

    VecSimIndexBasicInfo basicInfo() const override {
        VecSimIndexBasicInfo info = this->getBasicInfo();
        info.algo = VecSimAlgo_SVS;
        info.isTiered = false;
        return info;
    }

    VecSimIndexInfo info() const override {
        VecSimIndexInfo info;
        info.commonInfo = this->getCommonInfo();
        info.commonInfo.basicInfo.algo = VecSimAlgo_SVS;
        info.commonInfo.basicInfo.isTiered = false;
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
            index_impl->view_data(), index_impl->distance_function());

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
        if(get_vamana()->get_num_threads() != queries.size()) {
            get_vamana()->set_num_threads(queries.size());
        }
        get_vamana()->search(result.view(), queries, sp);

        assert(result.n_queries() == 1);

        for (size_t i = 0; i < result.n_neighbors(); i++) {
            rep->results.push_back(makeVecSimQueryResult(result, 0, i));
        }
        return rep;
    }

    VecSimQueryReply *rangeQuery(const void *queryBlob, double radius,
                                 VecSimQueryParams *queryParams) const override {
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

#ifdef BUILD_TESTS
    virtual void fitMemory() {};
#endif
};
