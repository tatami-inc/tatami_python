#ifndef TATAMI_PYTHON_UNKNOWNMATRIX_HPP
#define TATAMI_PYTHON_UNKNOWNMATRIX_HPP

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "tatami/tatami.hpp"

#include "dense_extractor.hpp"
#include "sparse_extractor.hpp"
#include "parallelize.hpp"

#include <vector>
#include <memory>
#include <string>
#include <stdexcept>
#include <optional>
#include <cstddef>

namespace tatami_python {

/**
 * @brief Options for data extraction from an `UnknownMatrix`.
 */
struct UnknownMatrixOptions {
    /**
     * Size of the cache, in bytes.
     */
    std::size_t maximum_cache_size = sanisizer::cap<std::size_t>(100000000);

    /**
     * Whether to automatically enforce a minimum size for the cache, regardless of `maximum_cache_size`.
     * This minimum is chosen to ensure that all chunks overlapping one row (or a slice/subset thereof) can be retained in memory,
     * so that the same chunks are not repeatedly re-read from disk when iterating over consecutive rows/columns of the matrix.
     */
    bool require_minimum_cache = true;
};

/**
 * @brief Unknown matrix-like object in Python.
 *
 * @tparam Value_ Numeric type of data value for the interface.
 * @tparam Index_ Integer type for the row/column indices, for the interface.
 *
 * Pull data out of an unknown matrix-like object by calling methods from the [**DelayedArray**](https://bioconductor.org/packages/DelayedArray) package via **Rcpp**.
 * This effectively extends **tatami** to work with any abstract numeric matrix that might be consumed by an R function.
 * 
 * Instances of class should only be constructed and destroyed in a serial context, specifically on the same thread running R itself. 
 * Calls to its methods may be parallelized but some additional effort is required to serialize calls to the R API; see `executor()` for more details.
 */
template<typename Value_, typename Index_, typename CachedValue_ = Value_, typename CachedIndex_ = Index_>
class UnknownMatrix : public tatami::Matrix<Value_, Index_> {
public:
    /**
     * This constructor should only be called when the current thread is holding the GIL,
     * as the construction of **pybind11** objects may call the Python API.
     *
     * @param seed A matrix-like Python object.
     * @param opt Extraction options.
     */
    UnknownMatrix(pybind11::object seed, const UnknownMatrixOptions& opt) : 
        my_seed(std::move(seed)), 
        my_module(pybind11::module::import("delayedarray")),
        my_dense_extractor(my_module.attr("extract_dense_array")),
        my_sparse_extractor(my_module.attr("extract_sparse_array")),
        my_cache_size_in_bytes(opt.maximum_cache_size),
        my_require_minimum_cache(opt.require_minimum_cache)
    {
        // We assume the constructor only occurs on the main thread, so we
        // won't bother locking things up. I'm also not sure that the
        // operations in the initialization list are thread-safe.

        const auto shape = get_shape<Index_>(my_seed);
        my_nrow = shape.first;
        my_ncol = shape.second;

        // Checking that we can safely create a pybind11::array_t<Index_> without overflow.
        // We do it here once, so that we don't need to check in each call to create_indexing_array().
        tatami::can_cast_Index_to_container_size<pybind11::array_t<Index_> >(std::max(my_nrow, my_ncol));

        auto sparse = my_module.attr("is_sparse")(my_seed);
        my_sparse = sparse.cast<bool>();

        auto grid = my_module.attr("chunk_grid")(my_seed);
        auto bounds = grid.attr("boundaries").cast<pybind11::tuple>();
        if (bounds.size() != 2) {
            auto ctype = get_class_name(seed);
            throw std::runtime_error("'chunk_grid(<" + ctype + ">).boundaries' should be a tuple of length 2");
        }

        auto np = pybind11::module::import("numpy");
        auto arrayfun = np.attr("array");
        auto populate = [&](Index_ extent, const pybind11::object& raw_ticks, std::vector<Index_>& map, std::vector<Index_>& new_ticks, Index_& max_chunk_size) {
            // Force realization of Iterable into a numpy array on the Python side.
            // Assume that casting to Index_ is safe, given that we were able to cast the extent.
            auto tick_array = arrayfun(raw_ticks, pybind11::dtype::of<Index_>());
            auto ticks = tick_array.template cast<pybind11::array_t<Index_> >();
            const auto tptr = static_cast<Index_*>(ticks.request().ptr);
            const auto nticks = ticks.size();

            new_ticks.reserve(sanisizer::sum<decltype(new_ticks.size())>(nticks, 1));
            new_ticks.push_back(0);
            tatami::resize_container_to_Index_size(map, extent);
            Index_ counter = 0;
            max_chunk_size = 0;

            for (I<decltype(nticks)> i = 0; i < nticks; ++i) {
                const auto latest = tptr[i];
                const auto previous = new_ticks.back();
                if (latest <= previous) {
                    auto ctype = get_class_name(seed);
                    throw std::runtime_error("boundaries are not strictly increasing in the output of 'chunk_grid(<" + ctype + ">).boundaries'");
                }
                new_ticks.push_back(latest);

                std::fill(map.begin() + previous, map.begin() + latest, counter);
                ++counter;
                const auto to_fill = latest - previous;
                if (to_fill > max_chunk_size) {
                    max_chunk_size = to_fill;
                }
            }

            if (!sanisizer::is_equal(new_ticks.back(), extent)) {
                auto ctype = get_class_name(seed);
                throw std::runtime_error("invalid ticks returned in 'chunk_grid(<" + ctype + ">).boundaries'");
            }
        };

        populate(my_nrow, bounds[0], my_row_chunk_map, my_row_chunk_ticks, my_row_max_chunk_size);
        populate(my_ncol, bounds[1], my_col_chunk_map, my_col_chunk_ticks, my_col_max_chunk_size);

        // Choose the dimension that requires pulling out fewer chunks.
        auto chunks_per_row = my_col_chunk_ticks.size() - 1;
        auto chunks_per_col = my_row_chunk_ticks.size() - 1;
        my_prefer_rows = chunks_per_row <= chunks_per_col;
    }

private:
    Index_ my_nrow, my_ncol;
    bool my_sparse, my_prefer_rows;

    std::vector<Index_> my_row_chunk_map, my_col_chunk_map;
    std::vector<Index_> my_row_chunk_ticks, my_col_chunk_ticks;

    // To decide how many chunks to store in the cache, we pretend the largest
    // chunk is a good representative. This is a bit suboptimal for irregular
    // chunks but the LruSlabCache class doesn't have a good way of dealing
    // with this right now. The fundamental problem is that variable slabs will
    // either (i) all reach the maximum allocation eventually, if slabs are
    // reused, or (ii) require lots of allocations, if slabs are not reused, or
    // (iii) require manual defragmentation, if slabs are reused in a manner
    // that avoids inflation to the maximum allocation.
    Index_ my_row_max_chunk_size, my_col_max_chunk_size;

    pybind11::object my_seed;
    pybind11::module my_module;
    pybind11::object my_dense_extractor, my_sparse_extractor;

    std::size_t my_cache_size_in_bytes;
    bool my_require_minimum_cache;

public:
    Index_ nrow() const {
        return my_nrow;
    }

    Index_ ncol() const {
        return my_ncol;
    }

    bool is_sparse() const {
        return my_sparse;
    }

    double is_sparse_proportion() const {
        return static_cast<double>(my_sparse);
    }

    bool prefer_rows() const {
        return my_prefer_rows;
    }

    double prefer_rows_proportion() const {
        return static_cast<double>(my_prefer_rows);
    }

    bool uses_oracle(bool) const {
        return true;
    }

private:
    Index_ max_primary_chunk_length(bool row) const {
        return (row ? my_row_max_chunk_size : my_col_max_chunk_size);
    }

    Index_ primary_num_chunks(bool row, Index_ primary_chunk_length) const {
        auto primary_dim = (row ? my_nrow : my_ncol);
        if (primary_chunk_length == 0) {
            return primary_dim;
        } else {
            return primary_dim / primary_chunk_length;
        }
    }

    Index_ secondary_dim(bool row) const {
        return (row ? my_ncol : my_nrow);
    }

    const std::vector<Index_>& chunk_ticks(bool row) const {
        if (row) {
            return my_row_chunk_ticks;
        } else {
            return my_col_chunk_ticks;
        }
    }

    const std::vector<Index_>& chunk_map(bool row) const {
        if (row) {
            return my_row_chunk_map;
        } else {
            return my_col_chunk_map;
        }
    }

    /********************
     *** Myopic dense ***
     ********************/
private:
    template<
        bool oracle_, 
        template <bool, bool, typename, typename, typename> class FromDense_,
        template <bool, bool, typename, typename, typename, typename> class FromSparse_,
        typename ... Args_
    >
    std::unique_ptr<tatami::DenseExtractor<oracle_, Value_, Index_> > populate_dense_internal(
        bool row,
        Index_ non_target_length,
        tatami::MaybeOracle<oracle_, Index_> oracle,
        Args_&& ... args
    ) const {
        Index_ max_target_chunk_length = max_primary_chunk_length(row);
        tatami_chunked::SlabCacheStats<Index_> stats(
            /* target length = */ max_target_chunk_length,
            /* non_target_length = */ non_target_length,
            /* target_num_slabs = */ primary_num_chunks(row, max_target_chunk_length),
            /* cache_size_in_bytes = */ my_cache_size_in_bytes,
            /* element_size = */ sizeof(CachedValue_),
            /* require_minimum_cache = */ my_require_minimum_cache
        );

        const auto& map = chunk_map(row);
        const auto& ticks = chunk_ticks(row);
        const bool solo = (stats.max_slabs_in_cache == 0);

        std::unique_ptr<tatami::DenseExtractor<oracle_, Value_, Index_> > output;
#ifdef TATAMI_PYTHON_PARALLELIZE_UNKNOWN 
        TATAMI_PYTHON_SERIALIZE([&]() -> void {
#endif

        if (!my_sparse) {
            if (solo) {
                output.reset(
                    new FromDense_<true, oracle_, Value_, Index_, CachedValue_>(
                        my_seed,
                        my_dense_extractor,
                        row,
                        std::move(oracle),
                        std::forward<Args_>(args)...,
                        ticks,
                        map,
                        stats
                    )
                );

            } else {
                output.reset(
                    new FromDense_<false, oracle_, Value_, Index_, CachedValue_>( 
                        my_seed,
                        my_dense_extractor,
                        row,
                        std::move(oracle),
                        std::forward<Args_>(args)...,
                        ticks,
                        map,
                        stats
                    )
                );
            }

        } else {
            if (solo) {
                output.reset(
                    new FromSparse_<true, oracle_, Value_, Index_, CachedValue_, CachedIndex_>(
                        my_seed,
                        my_sparse_extractor,
                        row,
                        std::move(oracle),
                        std::forward<Args_>(args)...,
                        max_target_chunk_length,
                        ticks,
                        map,
                        stats
                    )
                );

            } else {
                output.reset(
                    new FromSparse_<false, oracle_, Value_, Index_, CachedValue_, CachedIndex_>( 
                        my_seed,
                        my_sparse_extractor,
                        row,
                        std::move(oracle),
                        std::forward<Args_>(args)...,
                        max_target_chunk_length,
                        ticks,
                        map,
                        stats
                    )
                );
            }
        }

#ifdef TATAMI_PYTHON_PARALLELIZE_UNKNOWN 
        });
#endif

        return output;
    }

    template<bool oracle_>
    std::unique_ptr<tatami::DenseExtractor<oracle_, Value_, Index_> > populate_dense(
        bool row,
        tatami::MaybeOracle<oracle_, Index_> ora,
        const tatami::Options&
    ) const {
        Index_ non_target_dim = secondary_dim(row);
        return populate_dense_internal<oracle_, DenseFull, DensifiedSparseFull>(
            row,
            non_target_dim,
            std::move(ora),
            non_target_dim
        );
    }

    template<bool oracle_>
    std::unique_ptr<tatami::DenseExtractor<oracle_, Value_, Index_> > populate_dense(
        bool row,
        tatami::MaybeOracle<oracle_, Index_> ora,
        Index_ block_start,
        Index_ block_length,
        const tatami::Options&
    ) const {
        return populate_dense_internal<oracle_, DenseBlock, DensifiedSparseBlock>(
            row,
            block_length,
            std::move(ora),
            block_start,
            block_length
        );
    }

    template<bool oracle_>
    std::unique_ptr<tatami::DenseExtractor<oracle_, Value_, Index_> > populate_dense(
        bool row,
        tatami::MaybeOracle<oracle_, Index_> ora,
        tatami::VectorPtr<Index_> indices_ptr,
        const tatami::Options&
    ) const {
        Index_ nidx = indices_ptr->size();
        return populate_dense_internal<oracle_, DenseIndexed, DensifiedSparseIndexed>(
            row,
            nidx,
            std::move(ora),
            std::move(indices_ptr)
        );
    }

public:
    std::unique_ptr<tatami::MyopicDenseExtractor<Value_, Index_> > dense(
        bool row,
        const tatami::Options& opt
    ) const {
        return populate_dense<false>(row, false, opt); 
    }

    std::unique_ptr<tatami::MyopicDenseExtractor<Value_, Index_> > dense(
        bool row,
        Index_ block_start,
        Index_ block_length,
        const tatami::Options& opt
    ) const {
        return populate_dense<false>(row, false, block_start, block_length, opt); 
    }

    std::unique_ptr<tatami::MyopicDenseExtractor<Value_, Index_> > dense(
        bool row,
        tatami::VectorPtr<Index_> indices_ptr,
        const tatami::Options& opt
    ) const {
        return populate_dense<false>(row, false, std::move(indices_ptr), opt); 
    }

    /**********************
     *** Oracular dense ***
     **********************/
public:
    std::unique_ptr<tatami::OracularDenseExtractor<Value_, Index_> > dense(
        bool row,
        std::shared_ptr<const tatami::Oracle<Index_> > ora,
        const tatami::Options& opt
    ) const {
        return populate_dense<true>(row, std::move(ora), opt); 
    }

    std::unique_ptr<tatami::OracularDenseExtractor<Value_, Index_> > dense(
        bool row,
        std::shared_ptr<const tatami::Oracle<Index_> > ora,
        Index_ block_start,
        Index_ block_length,
        const tatami::Options& opt
    ) const {
        return populate_dense<true>(row, std::move(ora), block_start, block_length, opt); 
    }

    std::unique_ptr<tatami::OracularDenseExtractor<Value_, Index_> > dense(
        bool row,
        std::shared_ptr<const tatami::Oracle<Index_> > ora,
        tatami::VectorPtr<Index_> indices_ptr,
        const tatami::Options& opt
    ) const {
        return populate_dense<true>(row, std::move(ora), std::move(indices_ptr), opt); 
    }

    /*********************
     *** Myopic sparse ***
     *********************/
public:
    template<
        bool oracle_, 
        template<bool, bool, typename, typename, typename, typename> class FromSparse_,
        typename ... Args_
    >
    std::unique_ptr<tatami::SparseExtractor<oracle_, Value_, Index_> > populate_sparse_internal(
        bool row,
        Index_ non_target_length, 
        tatami::MaybeOracle<oracle_, Index_> oracle, 
        const tatami::Options& opt, 
        Args_&& ... args
    ) const {
        Index_ max_target_chunk_length = max_primary_chunk_length(row);
        tatami_chunked::SlabCacheStats<Index_> stats(
            /* target_length = */ max_target_chunk_length,
            /* non_target_length = */ non_target_length, 
            /* target_num_slabs = */ primary_num_chunks(row, max_target_chunk_length),
            /* cache_size_in_bytes = */ my_cache_size_in_bytes, 
            /* element_size = */ (opt.sparse_extract_index ? sizeof(CachedIndex_) : 0) + (opt.sparse_extract_value ? sizeof(CachedValue_) : 0),
            /* require_minimum_cache = */ my_require_minimum_cache
        );

        const auto& map = chunk_map(row);
        const auto& ticks = chunk_ticks(row);
        const bool needs_value = opt.sparse_extract_value;
        const bool needs_index = opt.sparse_extract_index;
        const bool solo = stats.max_slabs_in_cache == 0;

        std::unique_ptr<tatami::SparseExtractor<oracle_, Value_, Index_> > output;
#ifdef TATAMI_PYTHON_PARALLELIZE_UNKNOWN 
        TATAMI_PYTHON_SERIALIZE([&]() -> void {
#endif

        if (solo) {
            output.reset(
                new FromSparse_<true, oracle_, Value_, Index_, CachedValue_, CachedIndex_>( 
                    my_seed,
                    my_sparse_extractor,
                    row,
                    std::move(oracle),
                    std::forward<Args_>(args)...,
                    max_target_chunk_length,
                    ticks,
                    map,
                    stats,
                    needs_value,
                    needs_index
                )
            );

        } else {
            output.reset(
                new FromSparse_<false, oracle_, Value_, Index_, CachedValue_, CachedIndex_>( 
                    my_seed,
                    my_sparse_extractor,
                    row,
                    std::move(oracle),
                    std::forward<Args_>(args)...,
                    max_target_chunk_length,
                    ticks,
                    map,
                    stats,
                    needs_value,
                    needs_index
                )
            );
        }

#ifdef TATAMI_PYTHON_PARALLELIZE_UNKNOWN 
        });
#endif

        return output;
    }

    template<bool oracle_>
    std::unique_ptr<tatami::SparseExtractor<oracle_, Value_, Index_> > populate_sparse(
        bool row,
        tatami::MaybeOracle<oracle_, Index_> ora,
        const tatami::Options& opt
    ) const {
        Index_ non_target_dim = secondary_dim(row);
        return populate_sparse_internal<oracle_, SparseFull>(
            row,
            non_target_dim,
            std::move(ora),
            opt,
            non_target_dim
        ); 
    }

    template<bool oracle_>
    std::unique_ptr<tatami::SparseExtractor<oracle_, Value_, Index_> > populate_sparse(
        bool row,
        tatami::MaybeOracle<oracle_, Index_> ora,
        Index_ block_start,
        Index_ block_length,
        const tatami::Options& opt
    ) const {
        return populate_sparse_internal<oracle_, SparseBlock>(
            row,
            block_length,
            std::move(ora),
            opt,
            block_start,
            block_length
        );
    }

    template<bool oracle_>
    std::unique_ptr<tatami::SparseExtractor<oracle_, Value_, Index_> > populate_sparse(
        bool row,
        tatami::MaybeOracle<oracle_, Index_> ora,
        tatami::VectorPtr<Index_> indices_ptr,
        const tatami::Options& opt
    ) const {
        Index_ nidx = indices_ptr->size();
        return populate_sparse_internal<oracle_, SparseIndexed>(
            row,
            nidx,
            std::move(ora),
            opt,
            std::move(indices_ptr)
        );
    }

public:
    std::unique_ptr<tatami::MyopicSparseExtractor<Value_, Index_> > sparse(
        bool row,
        const tatami::Options& opt
    ) const {
        if (!my_sparse) {
            return std::make_unique<tatami::FullSparsifiedWrapper<false, Value_, Index_> >(
                dense(row, opt),
                secondary_dim(row),
                opt
            );
        } else {
            return populate_sparse<false>(
                row,
                false,
                opt
            );
        }
    }

    std::unique_ptr<tatami::MyopicSparseExtractor<Value_, Index_> > sparse(
        bool row,
        Index_ block_start,
        Index_ block_length,
        const tatami::Options& opt
    ) const {
        if (!my_sparse) {
            return std::make_unique<tatami::BlockSparsifiedWrapper<false, Value_, Index_> >(
                dense(row, block_start, block_length, opt),
                block_start,
                block_length,
                opt
            );
        } else {
            return populate_sparse<false>(
                row,
                false,
                block_start,
                block_length,
                opt
            ); 
        }
    }

    std::unique_ptr<tatami::MyopicSparseExtractor<Value_, Index_> > sparse(
        bool row,
        tatami::VectorPtr<Index_> indices_ptr,
        const tatami::Options& opt
    ) const {
        if (!my_sparse) {
            auto index_copy = indices_ptr;
            return std::make_unique<tatami::IndexSparsifiedWrapper<false, Value_, Index_> >(
                dense(row, std::move(indices_ptr), opt),
                std::move(index_copy),
                opt
            );
        } else {
            return populate_sparse<false>(
                row,
                false,
                std::move(indices_ptr),
                opt
            ); 
        }
    }

    /**********************
     *** Oracular sparse ***
     **********************/
public:
    std::unique_ptr<tatami::OracularSparseExtractor<Value_, Index_> > sparse(
        bool row,
        std::shared_ptr<const tatami::Oracle<Index_> > ora,
        const tatami::Options& opt
    ) const {
        if (!my_sparse) {
            return std::make_unique<tatami::FullSparsifiedWrapper<true, Value_, Index_> >(
                dense(row, std::move(ora), opt),
                secondary_dim(row),
                opt
            );
        } else {
            return populate_sparse<true>(
                row,
                std::move(ora),
                opt
            ); 
        }
    }

    std::unique_ptr<tatami::OracularSparseExtractor<Value_, Index_> > sparse(
        bool row,
        std::shared_ptr<const tatami::Oracle<Index_> > ora,
        Index_ block_start,
        Index_ block_length,
        const tatami::Options& opt
    ) const {
        if (!my_sparse) {
            return std::make_unique<tatami::BlockSparsifiedWrapper<true, Value_, Index_> >(
                dense(row, std::move(ora), block_start, block_length, opt),
                block_start,
                block_length,
                opt
            );
        } else {
            return populate_sparse<true>(
                row,
                std::move(ora),
                block_start,
                block_length,
                opt
            ); 
        }
    }

    std::unique_ptr<tatami::OracularSparseExtractor<Value_, Index_> > sparse(
        bool row,
        std::shared_ptr<const tatami::Oracle<Index_> > ora,
        tatami::VectorPtr<Index_> indices_ptr,
        const tatami::Options& opt
    ) const {
        if (!my_sparse) {
            auto index_copy = indices_ptr;
            return std::make_unique<tatami::IndexSparsifiedWrapper<true, Value_, Index_> >(
                dense(row, std::move(ora),
                std::move(indices_ptr), opt),
                std::move(index_copy),
                opt
            );
        } else {
            return populate_sparse<true>(
                row,
                std::move(ora),
                std::move(indices_ptr),
                opt
            ); 
        }
    }
};

}

#endif
