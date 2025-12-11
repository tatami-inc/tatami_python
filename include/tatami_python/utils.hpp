#ifndef TATAMI_PYTHON_UTILS_HPP
#define TATAMI_PYTHON_UTILS_HPP

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"

#include <string>
#include <utility>
#include <stdexcept>
#include <memory>
#include <numeric>

#include "tatami/tatami.hpp"
#include "sanisizer/sanisizer.hpp"

namespace tatami_python { 

template<typename Input_>
using I = std::remove_reference_t<std::remove_cv_t<Input_> >;

inline std::string get_class_name(const pybind11::object& incoming) {
    if (!pybind11::hasattr(incoming, "__class__")) {
        return "unknown";
    }
    auto cls = incoming.attr("__class__");
    if (!pybind11::hasattr(cls, "__name__")) {
        return "unnamed";
    }
    return cls.attr("__name__").cast<std::string>();
}

template<typename Index_>
std::pair<Index_, Index_> get_shape(const pybind11::object& obj) {
    auto shape = obj.attr("shape")();
    auto tup = shape.cast<pybind11::tuple>();
    if (tup.size() != 2) {
        auto ctype = get_class_name(obj);
        throw std::runtime_error("'<" + ctype + ">.shape' should return an integer vector");
    }

    // We use pybind11's own size type that it uses for NumPy shapes.
    // TODO: check if pybind11::cast() throws an error if it is outside of the range of the target type.
    auto raw_nrow = tup[0].cast<pybind11::ssize_t>();
    auto raw_ncol = tup[1].cast<pybind11::ssize_t>();
    if (raw_nrow < 0 || raw_ncol < 0) {
        auto ctype = get_class_name(obj);
        throw std::runtime_error("'dim(<" + ctype + ">)' should contain two non-negative integers");
    }

    return std::make_pair(
        sanisizer::cast<Index_>(raw_nrow),
        sanisizer::cast<Index_>(raw_ncol)
    );
}

template<typename Index_>
pybind11::array_t<Index_> create_indexing_array(const Index_ start, const Index_ length) {
    // No need to check for overflow in length, we already checked in the UnknownMatrix constructor.
    pybind11::array_t<Index_> output(length);
    auto pptr = static_cast<Index_*>(output.request().ptr);
    std::iota(pptr, pptr + length, start);
    return output;
}

template<typename Index_>
pybind11::array_t<Index_> create_indexing_array(const std::vector<Index_>& indices) {
    // No need to check for overflow in length, we already checked in the UnknownMatrix constructor.
    // We also know that all indices fit as they should be less than the extents, which in turn can fit in Index_.
    pybind11::array_t<Index_> output(indices.size()); 
    auto pptr = static_cast<Index_*>(output.request().ptr);
    std::copy(indices.begin(), indices.end(), pptr);
    return output;
}

}

#endif
