/* TODO: change the copyright here */

/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once
#include "VecSim/algorithms/svs/svs_utils.h"

// TODO(rfsaliev) to be detected and defined via cmake config
#define LVQ_EXISTS 1

#if LVQ_EXISTS
#include "svs/extensions/vamana/lvq.h"

namespace details {
template <size_t Primary, size_t Residual>
struct LVQSelector {
    using strategy = svs::quantization::lvq::Sequential;
};

template <>
struct LVQSelector<4, 8> {
    using strategy = svs::quantization::lvq::Turbo<16, 8>;
};

template <>
struct LVQSelector<8, 0> {
    using strategy = svs::quantization::lvq::Turbo<16, 4>;
};
} // namespace details

template <typename DataType, size_t QuantBits, size_t ResidualBits>
struct SVSStorageTraits<DataType, QuantBits, ResidualBits, std::enable_if_t<(QuantBits > 0)>> {
    using allocator_type = details::SVSAllocator<std::byte>;
    using blocked_type = svs::data::Blocked<allocator_type>;
    using strategy_type = typename details::LVQSelector<QuantBits, ResidualBits>::strategy;
    using index_storage_type =
        svs::quantization::lvq::LVQDataset<QuantBits, ResidualBits, svs::Dynamic, strategy_type,
                                           blocked_type>;

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
        +    template <typename T> using element_type_t = std::remove_cv_t<typename
        T::element_type>; using misc_type = std::vector<double>;

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
#endif // LVQ_EXISTS
