#ifndef TATAMI_PYTHON_SPARSE_MATRIX_HPP
#define TATAMI_PYTHON_SPARSE_MATRIX_HPP

#include "tatami/tatami.hpp"
#include "pybind11/pybind11.h"

#include "utils.hpp"

#include <algorithm>
#include <cstdint>

/**
 * @file sparse_matrix.hpp
 * @brief Parse sparse matrices from block processing.
 */

namespace tatami_python { 

/**
 * @cond
 */
template<typename Type_>
void dump_to_buffer(const pybind11::array& input, Type_* const buffer) {
    auto dtype = input.dtype();
    if (dtype.is(pybind11::dtype::of<double>())) {
        std::copy_n(static_cast<const double*>(input.request().ptr), input.size(), buffer);

    } else if (dtype.is(pybind11::dtype::of<float>())) {
        std::copy_n(static_cast<const float*>(input.request().ptr), input.size(), buffer);

    } else if (dtype.is(pybind11::dtype::of<std::int64_t>())) {
        std::copy_n(static_cast<const std::int64_t*>(input.request().ptr), input.size(), buffer);

    } else if (dtype.is(pybind11::dtype::of<std::int32_t>())) {
        std::copy_n(static_cast<const std::int32_t*>(input.request().ptr), input.size(), buffer);

    } else if (dtype.is(pybind11::dtype::of<std::int16_t>())) {
        std::copy_n(static_cast<const std::int16_t*>(input.request().ptr), input.size(), buffer);

    } else if (dtype.is(pybind11::dtype::of<std::int8_t>())) {
        std::copy_n(static_cast<const std::int8_t*>(input.request().ptr), input.size(), buffer);

    } else if (dtype.is(pybind11::dtype::of<std::uint64_t>())) {
        std::copy_n(static_cast<const std::uint64_t*>(input.request().ptr), input.size(), buffer);

    } else if (dtype.is(pybind11::dtype::of<std::uint32_t>())) {
        std::copy_n(static_cast<const std::uint32_t*>(input.request().ptr), input.size(), buffer);

    } else if (dtype.is(pybind11::dtype::of<std::uint16_t>())) {
        std::copy_n(static_cast<const std::uint16_t*>(input.request().ptr), input.size(), buffer);

    } else if (dtype.is(pybind11::dtype::of<std::uint8_t>())) {
        std::copy_n(static_cast<const std::uint8_t*>(input.request().ptr), input.size(), buffer);

    } else {
        throw std::runtime_error("unrecognized array type '" + std::string(dtype.kind(), 1) + std::to_string(dtype.itemsize()) + "' from 'extract_sparse_array()'");
    }
}
/**
 * @endcond
 */

/**
 * Parse the contents of a 2-dimensional `SparseNdArray` from the **delayedarray** package.
 *
 * @tparam Value_ Numeric type of the matrix value.
 * @tparam Index_ Integer type of the matrix indices.
 * @tparam Function_ Function to be applied at each leaf node.
 *
 * @param matrix The `SparseNdarray` object.
 * @param vbuffer Pointer to an array of length greater than or equal to the number of rows of `matrix`,
 * in which to store the values of the structural non-zero elements.
 * This may also be NULL in which case no values are extracted.
 * On input, the contents of the array are ignored by this function. 
 * @param ibuffer Pointer to an array of length greater than or equal to the number of rows of `matrix`,
 * in which to store the indices of the structural non-zero elements.
 * This may also be NULL in which case no values are extracted.
 * On input, the contents of the array are ignored by this function. 
 * @param fun Function to apply to each leaf node, accepting two arguments:
 * - `c`, an `Index_` specifying the index of the leaf node, i.e., the column index. 
 * - `n`, an `Index_` specifying the number of structural non-zero elements for `c`.
 *   The first `n` entries of `vbuffer` and `ibuffer` will be filled with the values and indices of these non-zero elements, respectively, if they are not NULL.
 * .
 * The return value of this function is ignored.
 * Note that `fun` may not be called for all `c` - if leaf nodes do not contain any data, they will be skipped.
 */
template<typename Value_, typename Index_, class Function_>
void parse_Sparse2darray(const pybind11::object& matrix, Value_* const vbuffer, Index_* const ibuffer, Function_ fun) {
    pybind11::object raw_svt = matrix.attr("contents");
    if (pybind11::isinstance<pybind11::none>(raw_svt)) {
        return;
    }
    auto svt = raw_svt.template cast<pybind11::list>();

    const auto shape = get_shape<Index_>(matrix);
    const auto NR = shape.first;
    const auto NC = shape.second;

    for (I<decltype(NC)> c = 0; c < NC; ++c) {
        pybind11::object raw_inner(svt[c]);
        if (pybind11::isinstance<pybind11::none>(raw_inner)) {
            continue;
        }

        auto inner = raw_inner.template cast<pybind11::tuple>();
        if (inner.size() != 2) {
            auto ctype = get_class_name(matrix);
            throw std::runtime_error("each entry of '<" + ctype + ">.contents' should be a tuple of length 2 or None");
        }

        auto iinput = inner[0].template cast<pybind11::array>();
        if (ibuffer != NULL) {
            dump_to_buffer(iinput, ibuffer);
        }
        if (vbuffer != NULL) {
            auto vinput = inner[1].template cast<pybind11::array>();
            dump_to_buffer(vinput, vbuffer);
        }

        // cast is known to be safe as the length of these vectors cannot excced
        // the number of rows, the latter of which must fit in an Index_.
        fun(c, static_cast<Index_>(iinput.size()));
    }
}

/**
 * @cond
 */
template<typename CachedValue_, typename CachedIndex_, typename Index_>
void parse_sparse_matrix(
    const pybind11::object& matrix,
    bool row,
    std::vector<CachedValue_*>& value_ptrs, 
    CachedValue_* const vbuffer,
    std::vector<CachedIndex_*>& index_ptrs, 
    CachedIndex_* const ibuffer,
    Index_* const counts
) {
    const bool needs_value = !value_ptrs.empty();
    const bool needs_index = !index_ptrs.empty();

    parse_Sparse2darray(
        matrix,
        (needs_value ? vbuffer : NULL),
        (needs_index || row ? ibuffer : NULL),
        [&](const Index_ c, const Index_ nnz) -> void {
            // Note that non-empty value_ptrs and index_ptrs may be longer than the
            // number of rows/columns in the SVT matrix, due to the reuse of slabs.
            if (row) {
                if (needs_value) {
                    for (I<decltype(nnz)> i = 0; i < nnz; ++i) {
                        auto ix = ibuffer[i];
                        value_ptrs[ix][counts[ix]] = vbuffer[i];
                    }
                }
                if (needs_index) {
                    for (I<decltype(nnz)> i = 0; i < nnz; ++i) {
                        auto ix = ibuffer[i];
                        index_ptrs[ix][counts[ix]] = c;
                    }
                }
                for (I<decltype(nnz)> i = 0; i < nnz; ++i) {
                    ++(counts[ibuffer[i]]);
                }

            } else {
                if (needs_value) {
                    std::copy_n(vbuffer, nnz, value_ptrs[c]);
                }
                if (needs_index) {
                    std::copy_n(ibuffer, nnz, index_ptrs[c]);
                }
                counts[c] = nnz;
            }
        }
    );
}
/**
 * @endcond
 */

}

#endif
