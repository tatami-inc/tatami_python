#ifndef TATAMI_PYTHON_SPARSE_EXTRACTOR_HPP
#define TATAMI_PYTHON_SPARSE_EXTRACTOR_HPP

#include "pybind11/pybind11.h"
#include "tatami/tatami.hpp"
#include "tatami_chunked/tatami_chunked.hpp"

#include "utils.hpp"
#include "sparse_matrix.hpp"
#include "parallelize.hpp"

#include <vector>
#include <stdexcept>
#include <optional>

namespace tatami_python {

// GENERAL COMMENTS:
//
// - No need to protect against overflows when creating pybind11::array_t<Index_t> from dimension extents.
//   We already know that the dimension extent can be safely converted to/from an int, based on checks in the UnknownMatrix constructor.

/********************
 *** Core classes ***
 ********************/

template<typename Index_, typename CachedValue_, typename CachedIndex_>
void initialize_tmp_buffers(
    const bool row,
    const Index_ max_rows_by_row,
    const Index_ max_rows_by_col,
    const bool needs_value, 
    std::vector<CachedValue_>& tmp_value,
    const bool needs_index, 
    std::vector<CachedIndex_>& tmp_index
) {
    const auto len = (row ? max_rows_by_row : max_rows_by_col);
    if (needs_value) {
        sanisizer::resize(tmp_value, len);
    }
    if (needs_index || row) { // we always need the indices for row-major extraction, to keep a running count for each row.
        sanisizer::resize(tmp_index, len);
    }
}

template<bool oracle_, typename Index_, typename CachedValue_, typename CachedIndex_>
class SoloSparseCore {
public:
    SoloSparseCore(
        const pybind11::object& matrix, 
        const pybind11::object& sparse_extractor,
        const bool row,
        tatami::MaybeOracle<oracle_, Index_> oracle,
        pybind11::array non_target_extract, 
        [[maybe_unused]] Index_ max_target_chunk_length, // provided here for compatibility with the other Sparse*Core classes.
        [[maybe_unused]] const std::vector<Index_>& ticks,
        [[maybe_unused]] const std::vector<Index_>& map,
        [[maybe_unused]] const tatami_chunked::SlabCacheStats<Index_>& stats,
        const bool needs_value,
        const bool needs_index
    ) : 
        my_matrix(matrix),
        my_sparse_extractor(sparse_extractor),
        my_row(row),
        my_factory(
            1,
            sanisizer::cast<CachedIndex_>(non_target_extract.size()),
            1,
            needs_value,
            needs_index
        ),
        my_solo(my_factory.create()),
        my_oracle(std::move(oracle))
    {
        initialize_tmp_buffers<Index_>(row, 1, non_target_extract.size(), needs_value, my_value_tmp, needs_index, my_index_tmp);
        my_extract_args.emplace(2);
        (*my_extract_args)[static_cast<int>(row)] = std::move(non_target_extract);
    }

    ~SoloSparseCore() {
#ifdef TATAMI_PYTHON_PARALLELIZE_UNKNOWN 
        TATAMI_PYTHON_SERIALIZE([&]() -> void {
            my_extract_args.reset();
        });
#endif
    }

private:
    const pybind11::object& my_matrix;
    const pybind11::object& my_sparse_extractor;
    std::optional<pybind11::tuple> my_extract_args;

    bool my_row;

    tatami_chunked::SparseSlabFactory<CachedValue_, CachedIndex_> my_factory;
    typedef typename decltype(my_factory)::Slab Slab;
    Slab my_solo;

    tatami::MaybeOracle<oracle_, Index_> my_oracle;
    typename std::conditional<oracle_, tatami::PredictionIndex, bool>::type my_counter = 0;

    std::vector<CachedValue_> my_value_tmp;
    std::vector<CachedIndex_> my_index_tmp;

public:
    std::pair<const Slab*, Index_> fetch_raw(Index_ i) {
        if constexpr(oracle_) {
            i = my_oracle->get(my_counter++);
        }
        my_solo.number[0] = 0;

#ifdef TATAMI_PYTHON_PARALLELIZE_UNKNOWN 
        TATAMI_PYTHON_SERIALIZE([&]() -> void {
#endif

        (*my_extract_args)[static_cast<int>(!my_row)] = create_indexing_array(i, 1);
        const auto obj = my_sparse_extractor(my_matrix, *my_extract_args);
        parse_sparse_matrix(
            obj,
            my_row,
            my_solo.values,
            my_value_tmp.data(),
            my_solo.indices,
            my_index_tmp.data(),
            my_solo.number
        );

#ifdef TATAMI_PYTHON_PARALLELIZE_UNKNOWN 
        });
#endif

        return std::make_pair(&my_solo, static_cast<Index_>(0));
    }
};

template<typename Index_, typename CachedValue_, typename CachedIndex_>
class MyopicSparseCore {
public:
    MyopicSparseCore(
        const pybind11::object& matrix, 
        const pybind11::object& sparse_extractor,
        const bool row,
        [[maybe_unused]] tatami::MaybeOracle<false, Index_> oracle, // provided here for compatibility with the other Sparse*Core classes.
        pybind11::array non_target_extract, 
        const Index_ max_target_chunk_length, 
        const std::vector<Index_>& ticks,
        const std::vector<Index_>& map,
        const tatami_chunked::SlabCacheStats<Index_>& stats,
        const bool needs_value,
        const bool needs_index
    ) : 
        my_matrix(matrix),
        my_sparse_extractor(sparse_extractor),
        my_row(row),
        my_chunk_ticks(ticks),
        my_chunk_map(map),
        my_factory(
            sanisizer::cast<CachedIndex_>(max_target_chunk_length),
            sanisizer::cast<CachedIndex_>(non_target_extract.size()),
            stats,
            needs_value,
            needs_index
        ),
        my_cache(stats.max_slabs_in_cache)
    {
        initialize_tmp_buffers<Index_>(row, max_target_chunk_length, non_target_extract.size(), needs_value, my_value_tmp, needs_index, my_index_tmp);
        my_extract_args.emplace(2);
        (*my_extract_args)[static_cast<int>(row)] = std::move(non_target_extract);
    }

    ~MyopicSparseCore() {
#ifdef TATAMI_PYTHON_PARALLELIZE_UNKNOWN 
        TATAMI_PYTHON_SERIALIZE([&]() -> void {
            my_extract_args.reset();
        });
#endif
    }

private:
    const pybind11::object& my_matrix;
    const pybind11::object& my_sparse_extractor;
    std::optional<pybind11::tuple> my_extract_args;

    bool my_row;

    const std::vector<Index_>& my_chunk_ticks;
    const std::vector<Index_>& my_chunk_map;

    tatami_chunked::SparseSlabFactory<CachedValue_, CachedIndex_> my_factory;
    typedef typename decltype(my_factory)::Slab Slab;
    tatami_chunked::LruSlabCache<Index_, Slab> my_cache;

    std::vector<CachedValue_> my_value_tmp;
    std::vector<CachedIndex_> my_index_tmp;

public:
    std::pair<const Slab*, Index_> fetch_raw(Index_ i) {
        const auto chosen = my_chunk_map[i];

        const auto& slab = my_cache.find(
            chosen,
            [&]() -> Slab {
                return my_factory.create();
            },
            [&](const Index_ id, Slab& cache) -> void {
                const auto chunk_start = my_chunk_ticks[id], chunk_end = my_chunk_ticks[id + 1];
                const Index_ chunk_len = chunk_end - chunk_start;
                std::fill_n(cache.number, chunk_len, 0);

#ifdef TATAMI_PYTHON_PARALLELIZE_UNKNOWN 
                TATAMI_PYTHON_SERIALIZE([&]() -> void {
#endif

                (*my_extract_args)[static_cast<int>(!my_row)] = create_indexing_array<Index_>(chunk_start, chunk_len);
                const auto obj = my_sparse_extractor(my_matrix, *my_extract_args);
                parse_sparse_matrix(
                    obj,
                    my_row,
                    cache.values,
                    my_value_tmp.data(),
                    cache.indices,
                    my_index_tmp.data(),
                    cache.number
                );

#ifdef TATAMI_PYTHON_PARALLELIZE_UNKNOWN 
                });
#endif
            }
        );

        const Index_ offset = i - my_chunk_ticks[chosen];
        return std::make_pair(&slab, offset);
    }
};

template<typename Index_, typename CachedValue_, typename CachedIndex_>
class OracularSparseCore {
public:
    OracularSparseCore(
        const pybind11::object& matrix, 
        const pybind11::object& sparse_extractor,
        const bool row,
        tatami::MaybeOracle<true, Index_> oracle,
        pybind11::array non_target_extract, 
        const Index_ max_target_chunk_length, 
        const std::vector<Index_>& ticks,
        const std::vector<Index_>& map,
        const tatami_chunked::SlabCacheStats<Index_>& stats,
        const bool needs_value,
        const bool needs_index
    ) : 
        my_matrix(matrix),
        my_sparse_extractor(sparse_extractor),
        my_row(row),
        my_chunk_ticks(ticks),
        my_chunk_map(map),
        my_factory(
            sanisizer::cast<CachedIndex_>(max_target_chunk_length),
            sanisizer::cast<CachedIndex_>(non_target_extract.size()),
            stats,
            needs_value,
            needs_index
        ),
        my_cache(std::move(oracle), stats.max_slabs_in_cache),
        my_needs_value(needs_value),
        my_needs_index(needs_index)
    {
        // map.size() is equal to the extent of the target dimension.
        // We don't know how many chunks we might bundle together in a single call, so better overestimate to be safe.
        initialize_tmp_buffers<Index_>(row, map.size(), non_target_extract.size(), needs_value, my_value_tmp, needs_index, my_index_tmp);
        my_extract_args.emplace(2);
        (*my_extract_args)[static_cast<int>(row)] = std::move(non_target_extract);
    }

    ~OracularSparseCore() {
#ifdef TATAMI_PYTHON_PARALLELIZE_UNKNOWN 
        TATAMI_PYTHON_SERIALIZE([&]() -> void {
            my_extract_args.reset();
        });
#endif
    }

private:
    const pybind11::object& my_matrix;
    const pybind11::object& my_sparse_extractor;
    std::optional<pybind11::tuple> my_extract_args;

    bool my_row;

    const std::vector<Index_>& my_chunk_ticks;
    const std::vector<Index_>& my_chunk_map;

    tatami_chunked::SparseSlabFactory<CachedValue_, CachedIndex_> my_factory;
    typedef typename decltype(my_factory)::Slab Slab;
    tatami_chunked::OracularSlabCache<Index_, Index_, Slab> my_cache;

    std::vector<CachedValue_*> my_chunk_value_ptrs;
    std::vector<CachedIndex_*> my_chunk_index_ptrs;
    std::vector<CachedIndex_> my_chunk_numbers;

    bool my_needs_value;
    bool my_needs_index;

    std::vector<CachedValue_> my_value_tmp;
    std::vector<CachedIndex_> my_index_tmp;

public:
    std::pair<const Slab*, Index_> fetch_raw(const Index_) {
        return my_cache.next(
            [&](const Index_ i) -> std::pair<Index_, Index_> {
                auto chosen = my_chunk_map[i];
                return std::make_pair(chosen, static_cast<Index_>(i - my_chunk_ticks[chosen]));
            },
            [&]() -> Slab {
                return my_factory.create();
            },
            [&](std::vector<std::pair<Index_, Slab*> >& to_populate) -> void {
                // Sorting them so that the indices are in order.
                auto cmp = [](const std::pair<Index_, Slab*>& left, const std::pair<Index_, Slab*> right) -> bool {
                    return left.first < right.first; 
                };
                if (!std::is_sorted(to_populate.begin(), to_populate.end(), cmp)) {
                    std::sort(to_populate.begin(), to_populate.end(), cmp);
                }

                if (my_needs_value) {
                    my_chunk_value_ptrs.clear();
                }
                if (my_needs_index) {
                    my_chunk_index_ptrs.clear();
                }

                Index_ total_len = 0;
                for (const auto& p : to_populate) {
                    Index_ chunk_len = my_chunk_ticks[p.first + 1] - my_chunk_ticks[p.first];
                    total_len += chunk_len;
                    if (my_needs_value) {
                        auto vIt = p.second->values.begin();
                        my_chunk_value_ptrs.insert(my_chunk_value_ptrs.end(), vIt, vIt + chunk_len);
                    }
                    if (my_needs_index) {
                        auto iIt = p.second->indices.begin();
                        my_chunk_index_ptrs.insert(my_chunk_index_ptrs.end(), iIt, iIt + chunk_len);
                    }
                }

                my_chunk_numbers.clear();
                tatami::resize_container_to_Index_size(my_chunk_numbers, total_len);

#ifdef TATAMI_PYTHON_PARALLELIZE_UNKNOWN 
                TATAMI_PYTHON_SERIALIZE([&]() -> void {
#endif

                pybind11::array_t<Index_> primary_extract(total_len); // known to be safe, from the constructor.
                auto pptr = static_cast<Index_*>(primary_extract.request().ptr);
                Index_ current = 0;
                for (const auto& p : to_populate) {
                    const Index_ chunk_start = my_chunk_ticks[p.first];
                    const Index_ chunk_len = my_chunk_ticks[p.first + 1] - chunk_start;
                    auto start = pptr + current;
                    std::iota(start, start + chunk_len, chunk_start);
                    current += chunk_len;
                }

                (*my_extract_args)[static_cast<int>(!my_row)] = std::move(primary_extract);
                auto obj = my_sparse_extractor(my_matrix, *my_extract_args);
                parse_sparse_matrix(
                    obj,
                    my_row,
                    my_chunk_value_ptrs,
                    my_value_tmp.data(),
                    my_chunk_index_ptrs,
                    my_index_tmp.data(),
                    my_chunk_numbers.data()
                );

                current = 0;
                for (const auto& p : to_populate) {
                    Index_ chunk_len = my_chunk_ticks[p.first + 1] - my_chunk_ticks[p.first];
                    std::copy_n(my_chunk_numbers.begin() + current, chunk_len, p.second->number);
                    current += chunk_len;
                }

#ifdef TATAMI_PYTHON_PARALLELIZE_UNKNOWN 
                });
#endif
            }
        );
    }
};

template<bool solo_, bool oracle_, typename Index_, typename CachedValue_, typename CachedIndex_>
using SparseCore = typename std::conditional<solo_,
    SoloSparseCore<oracle_, Index_, CachedValue_, CachedIndex_>,
    typename std::conditional<oracle_,
        OracularSparseCore<Index_, CachedValue_, CachedIndex_>,
        MyopicSparseCore<Index_, CachedValue_, CachedIndex_>
    >::type
>::type;

/******************************
 *** Pure sparse extractors ***
 ******************************/

template<bool solo_, bool oracle_, typename Value_, typename Index_, typename CachedValue_, typename CachedIndex_>
class SparseFull : public tatami::SparseExtractor<oracle_, Value_, Index_> {
public:
    SparseFull(
        const pybind11::object& matrix, 
        const pybind11::object& sparse_extractor,
        const bool row,
        tatami::MaybeOracle<oracle_, Index_> oracle,
        const Index_ non_target_dim,
        const Index_ max_target_chunk_length, 
        const std::vector<Index_>& ticks,
        const std::vector<Index_>& map,
        const tatami_chunked::SlabCacheStats<Index_>& stats,
        const bool needs_value,
        const bool needs_index
    ) : 
        my_core(
            matrix,
            sparse_extractor,
            row,
            std::move(oracle),
            create_indexing_array<Index_>(0, non_target_dim),
            max_target_chunk_length,
            ticks,
            map,
            stats,
            needs_value,
            needs_index
        ),
        my_non_target_dim(non_target_dim),
        my_needs_value(needs_value),
        my_needs_index(needs_index)
    {}

private:
    SparseCore<solo_, oracle_, Index_, CachedValue_, CachedIndex_> my_core;
    Index_ my_non_target_dim;
    bool my_needs_value, my_needs_index;

public:
    tatami::SparseRange<Value_, Index_> fetch(const Index_ i, Value_* const value_buffer, Index_* const index_buffer) {
        const auto res = my_core.fetch_raw(i);
        const auto& slab = *(res.first);
        const Index_ offset = res.second;

        tatami::SparseRange<Value_, Index_> output(slab.number[offset]);
        if (my_needs_value) {
            std::copy_n(slab.values[offset], my_non_target_dim, value_buffer);
            output.value = value_buffer;
        }

        if (my_needs_index) {
            std::copy_n(slab.indices[offset], my_non_target_dim, index_buffer);
            output.index = index_buffer;
        }

        return output;
    }
};

template<bool solo_, bool oracle_, typename Value_, typename Index_, typename CachedValue_, typename CachedIndex_>
class SparseBlock : public tatami::SparseExtractor<oracle_, Value_, Index_> {
public:
    SparseBlock(
        const pybind11::object& matrix, 
        const pybind11::object& sparse_extractor,
        const bool row,
        tatami::MaybeOracle<oracle_, Index_> oracle,
        const Index_ block_start,
        const Index_ block_length,
        const Index_ max_target_chunk_length, 
        const std::vector<Index_>& ticks,
        const std::vector<Index_>& map,
        const tatami_chunked::SlabCacheStats<Index_>& stats,
        const bool needs_value,
        const bool needs_index
    ) : 
        my_core(
            matrix,
            sparse_extractor,
            row,
            std::move(oracle),
            create_indexing_array<Index_>(block_start, block_length),
            max_target_chunk_length,
            ticks,
            map,
            stats,
            needs_value,
            needs_index
        ),
        my_block_start(block_start),
        my_needs_value(needs_value),
        my_needs_index(needs_index)
    {}

private:
    SparseCore<solo_, oracle_, Index_, CachedValue_, CachedIndex_> my_core;
    Index_ my_block_start; 
    bool my_needs_value, my_needs_index;

public:
    tatami::SparseRange<Value_, Index_> fetch(const Index_ i, Value_* const value_buffer, Index_* const index_buffer) {
        auto res = my_core.fetch_raw(i);
        const auto& slab = *(res.first);
        const Index_ offset = res.second;

        tatami::SparseRange<Value_, Index_> output(slab.number[offset]);
        if (my_needs_value) {
            std::copy_n(slab.values[offset], output.number, value_buffer); 
            output.value = value_buffer;
        }

        if (my_needs_index) {
            const auto iptr = slab.indices[offset];
            for (Index_ i = 0; i < output.number; ++i) {
                index_buffer[i] = static_cast<Index_>(iptr[i]) + my_block_start;
            }
            output.index = index_buffer;
        }

        return output;
    }
};

template<bool solo_, bool oracle_, typename Value_, typename Index_, typename CachedValue_, typename CachedIndex_>
class SparseIndexed : public tatami::SparseExtractor<oracle_, Value_, Index_> {
public:
    SparseIndexed(
        const pybind11::object& matrix, 
        const pybind11::object& sparse_extractor,
        const bool row,
        tatami::MaybeOracle<oracle_, Index_> oracle,
        tatami::VectorPtr<Index_> indices_ptr,
        const Index_ max_target_chunk_length, 
        const std::vector<Index_>& ticks,
        const std::vector<Index_>& map,
        const tatami_chunked::SlabCacheStats<Index_>& stats,
        const bool needs_value,
        const bool needs_index
    ) : 
        my_core(
            matrix,
            sparse_extractor,
            row,
            std::move(oracle),
            create_indexing_array(*indices_ptr),
            max_target_chunk_length,
            ticks,
            map,
            stats,
            needs_value,
            needs_index
        ),
        my_indices_ptr(std::move(indices_ptr)),
        my_needs_value(needs_value),
        my_needs_index(needs_index)
    {}

private:
    SparseCore<solo_, oracle_, Index_, CachedValue_, CachedIndex_> my_core;
    tatami::VectorPtr<Index_> my_indices_ptr;
    bool my_needs_value, my_needs_index;

public:
    tatami::SparseRange<Value_, Index_> fetch(const Index_ i, Value_* const value_buffer, Index_* const index_buffer) {
        const auto res = my_core.fetch_raw(i);
        const auto& slab = *(res.first);
        const Index_ offset = res.second;

        tatami::SparseRange<Value_, Index_> output(slab.number[offset]);
        if (my_needs_value) {
            std::copy_n(slab.values[offset], output.number, value_buffer); 
            output.value = value_buffer;
        }

        if (my_needs_index) {
            const auto iptr = slab.indices[offset];
            const auto& indices = *my_indices_ptr;
            for (Index_ i = 0; i < output.number; ++i) {
                index_buffer[i] = indices[iptr[i]];
            }
            output.index = index_buffer;
        }

        return output;
    }
};

/***********************************
 *** Densified sparse extractors ***
 ***********************************/

template<typename Slab_, typename Value_, typename Index_>
const Value_* densify(const Slab_& slab, const Index_ offset, const Index_ non_target_length, Value_* const buffer) {
    const auto vptr = slab.values[offset];
    const auto iptr = slab.indices[offset];
    std::fill_n(buffer, non_target_length, 0);
    const auto num = slab.number[offset];
    for (Index_ i = 0; i < num; ++i) {
        buffer[iptr[i]] = vptr[i];
    }
    return buffer;
}

template<bool solo_, bool oracle_, typename Value_, typename Index_, typename CachedValue_, typename CachedIndex_>
class DensifiedSparseFull : public tatami::DenseExtractor<oracle_, Value_, Index_> {
public:
    DensifiedSparseFull(
        const pybind11::object& matrix, 
        const pybind11::object& sparse_extractor,
        const bool row,
        tatami::MaybeOracle<oracle_, Index_> oracle,
        const Index_ non_target_dim,
        const Index_ max_target_chunk_length, 
        const std::vector<Index_>& ticks,
        const std::vector<Index_>& map,
        const tatami_chunked::SlabCacheStats<Index_>& stats
    ) :
        my_core(
            matrix,
            sparse_extractor,
            row,
            std::move(oracle),
            create_indexing_array<Index_>(0, non_target_dim),
            max_target_chunk_length,
            ticks,
            map,
            stats,
            true,
            true
        ),
        my_non_target_dim(non_target_dim)
    {}

private:
    SparseCore<solo_, oracle_, Index_, CachedValue_, CachedIndex_> my_core;
    Index_ my_non_target_dim;

public:
    const Value_* fetch(const Index_ i, Value_* const buffer) {
        const auto res = my_core.fetch_raw(i);
        return densify(*(res.first), res.second, my_non_target_dim, buffer);
    }
};

template<bool solo_, bool oracle_, typename Value_, typename Index_, typename CachedValue_, typename CachedIndex_>
class DensifiedSparseBlock : public tatami::DenseExtractor<oracle_, Value_, Index_> {
public:
    DensifiedSparseBlock(
        const pybind11::object& matrix, 
        const pybind11::object& sparse_extractor,
        const bool row,
        tatami::MaybeOracle<oracle_, Index_> oracle,
        const Index_ block_start,
        const Index_ block_length,
        const Index_ max_target_chunk_length, 
        const std::vector<Index_>& ticks,
        const std::vector<Index_>& map,
        const tatami_chunked::SlabCacheStats<Index_>& stats
    ) :
        my_core(
            matrix,
            sparse_extractor,
            row,
            std::move(oracle),
            create_indexing_array<Index_>(block_start, block_length),
            max_target_chunk_length,
            ticks,
            map,
            stats,
            true,
            true
        ),
        my_block_length(block_length)
    {}

private:
    SparseCore<solo_, oracle_, Index_, CachedValue_, CachedIndex_> my_core;
    Index_ my_block_length;

public:
    const Value_* fetch(const Index_ i, Value_* const buffer) {
        const auto res = my_core.fetch_raw(i);
        return densify(*(res.first), res.second, my_block_length, buffer);
    }
};

template<bool solo_, bool oracle_, typename Value_, typename Index_, typename CachedValue_, typename CachedIndex_>
class DensifiedSparseIndexed : public tatami::DenseExtractor<oracle_, Value_, Index_> {
public:
    DensifiedSparseIndexed(
        const pybind11::object& matrix, 
        const pybind11::object& sparse_extractor,
        const bool row,
        tatami::MaybeOracle<oracle_, Index_> oracle,
        tatami::VectorPtr<Index_> idx_ptr,
        const Index_ max_target_chunk_length, 
        const std::vector<Index_>& ticks,
        const std::vector<Index_>& map,
        const tatami_chunked::SlabCacheStats<Index_>& stats
    ) :
        my_core( 
            matrix,
            sparse_extractor,
            row,
            std::move(oracle),
            create_indexing_array(*idx_ptr),
            max_target_chunk_length,
            ticks,
            map,
            stats,
            true,
            true
        ),
        my_num_indices(idx_ptr->size())
    {}

private:
    SparseCore<solo_, oracle_, Index_, CachedValue_, CachedIndex_> my_core;
    Index_ my_num_indices;

public:
    const Value_* fetch(const Index_ i, Value_* const buffer) {
        const auto res = my_core.fetch_raw(i);
        return densify(*(res.first), res.second, my_num_indices, buffer);
    }
};

}

#endif
