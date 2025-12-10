#ifndef TATAMI_PYTHON_DENSE_MATRIX_HPP
#define TATAMI_PYTHON_DENSE_MATRIX_HPP

#include "pybind11/pybind11.h"
#include "tatami/tatami.hpp"
#include "sanisizer/sanisizer.hpp"

#include <algorithm>
#include <cstddef>

namespace tatami_python { 

template<typename InputValue_, typename Index_, typename CachedValue_>
void parse_dense_matrix_internal(
    const pybind11::array& data,
    Index_ data_start_row,
    Index_ data_start_col,
    bool row,
    CachedValue_* cache,
    Index_ cache_num_rows,
    Index_ cache_num_cols
) {
    const auto shape = data.request().shape;
    const Index_ data_num_rows = shape[0]; // casts are safe as everything should be less than the matrix extent.
    const Index_ data_num_cols = shape[1];
    auto ptr = static_cast<const InputValue_*>(data.request().ptr);

    if (row_major) {
        ptr += sanisizer::nd_offset<std::size_t>(data_start_col, data_num_cols, data_start_row);
        if (row) {
            for (Index_ r = 0; r < cache_num_rows; ++r) {
                std::copy_n(
                    input + sanisizer::product_unsafe<std::size_t>(r, data_num_cols),
                    cache_num_cols,
                    cache + sanisizer::product_unsafe<std::size_t>(r, cache_num_cols)
                );
            }
        } else {
            tatami::transpose(input, cache_num_rows, cache_num_cols, data_num_cols, cache, cache_num_rows);
        }

    } else {
        ptr += sanisizer::nd_offset<std::size_t>(data_start_row, data_num_rows, data_start_col);
        if (row) {
            // 'data' is a column-major matrix, but transpose() expects a row-major
            // input, so we just conceptually transpose it.
            tatami::transpose(input, cache_num_cols, cache_num_rows, data_num_rows, cache, cache_num_cols);
        } else {
            for (Index_ c = 0; c < cache_num_cols; ++c) {
                std::copy_n(
                    input + sanisizer::product_unsafe<std::size_t>(c, data_num_rows),
                    cache_num_rows,
                    cache + sanisizer::product_unsafe<std::size_t>(c, cache_num_rows)
                );
            }
        }
    }
}

template<typename Index_, typename CachedValue_>
void parse_dense_matrix(
    const pybind11::array& seed,
    Index_ data_start_row,
    Index_ data_start_col,
    bool by_row,
    CachedValue_* cache,
    Index_ cache_num_rows,
    Index_ cache_num_cols
) {
    auto flag = seed.flags();
    bool row_major = false;
    if (flag & pybind11::array::c_style) {
        row_major = true;
    } else if (flag & pybind11::array::f_style) {
        row_major = false;
    } else {
        throw std::runtime_error("numpy array contents should be contiguous");
    }

    auto dtype = buffer.dtype();
    if (dtype.is(pybind11::dtype::of<double>())) {
        parse_dense_matrix_internal<double>(seed, data_start_row, data_start_col, by_row, row_major, cache, cache_num_rows, cache_num_cols);

    } else if (dtype.is(pybind11::dtype::of<float>())) {
        parse_dense_matrix_internal<float>(seed, data_start_row, data_start_col, row, row_major, cache, cache_num_rows, cache_num_cols);

    } else if (dtype.is(pybind11::dtype::of<std::int64_t>())) {
        parse_dense_matrix_internal<std::int64_t>(seed, data_start_row, data_start_col, row, row_major, cache, cache_num_rows, cache_num_cols);

    } else if (dtype.is(pybind11::dtype::of<std::int32_t>())) {
        parse_dense_matrix_internal<std::int32_t>(seed, data_start_row, data_start_col, row, row_major, cache, cache_num_rows, cache_num_cols);

    } else if (dtype.is(pybind11::dtype::of<std::int16_t>())) {
        parse_dense_matrix_internal<std::int16_t>(seed, data_start_row, data_start_col, row, row_major, cache, cache_num_rows, cache_num_cols);

    } else if (dtype.is(pybind11::dtype::of<std::int8_t>())) {
        parse_dense_matrix_internal<std::int8_t>(seed, data_start_row, data_start_col, row, row_major, cache, cache_num_rows, cache_num_cols);

    } else if (dtype.is(pybind11::dtype::of<std::uint64_t>())) {
        parse_dense_matrix_internal<std::uint64_t>(seed, data_start_row, data_start_col, row, row_major, cache, cache_num_rows, cache_num_cols);

    } else if (dtype.is(pybind11::dtype::of<std::uint32_t>())) {
        parse_dense_matrix_internal<std::uint32_t>(seed, data_start_row, data_start_col, row, row_major, cache, cache_num_rows, cache_num_cols);

    } else if (dtype.is(pybind11::dtype::of<std::uint16_t>())) {
        parse_dense_matrix_internal<std::uint16_t>(seed, data_start_row, data_start_col, row, row_major, cache, cache_num_rows, cache_num_cols);

    } else if (dtype.is(pybind11::dtype::of<std::uint8_t>())) {
        parse_dense_matrix_internal<std::uint8_t>(seed, data_start_row, data_start_col, row, row_major, cache, cache_num_rows, cache_num_cols);

    } else {
        throw std::runtime_error("unrecognized array type '" + std::string(dtype.kind(), 1) + std::to_string(dtype.itemsize()) + "' from 'extract_dense_array()'");
    }
}

}

#endif
