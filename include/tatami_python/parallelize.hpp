#ifndef TATAMI_PYTHON_PARALLELIZE_HPP
#define TATAMI_PYTHON_PARALLELIZE_HPP

/**
 * @cond
 */
#ifdef TATAMI_PYTHON_PARALLELIZE_UNKNOWN
/**
 * @endcond
 */

#include "pybind11/pybind11.h"
#include "subpar/subpar.hpp"

#include <optional>

#ifndef TATAMI_PYTHON_SERIALIZE
#define TATAMI_PYTHON_SERIALIZE ::tatami_python::lock
#endif 

/**
 * @file parallelize.hpp
 * @brief Utilities for safe parallelization.
 */

namespace tatami_python {

/**
 * Replacement for `tatami::parallelize()` that applies a function to a set of tasks in parallel, usually for iterating over a dimension of a `Matrix`.
 * This releases the Python GIL so that it can be re-acquired by `UnknownMatrix` extractors in each individual thread.
 *
 * @tparam Function_ Function to be applied to a contiguous range of tasks.
 * This should accept three arguments:
 * - `thread`, the thread number executing this task range.
 *   This will be passed as an `int`.
 * - `task_start`, the start index of the task range.
 *   This will be passed as an `Index_`.
 * - `task_length`, the number of tasks in the task range.
 *   This will be passed as an `Index_`.
 * @tparam Index_ Integer type for the number of tasks.
 *
 * @param fun Function that executes a contiguous range of tasks.
 * @param tasks Number of tasks.
 * @param threads Number of threads.
 */
template<class Function_, class Index_>
void parallelize(const Function_ fun, const Index_ tasks, int threads) {
    std::optional<pybind11::gil_scoped_release> ungil;
    if (PyGILState_Check()) {
        ungil.emplace();
    }
    subpar::parallelize_range(threads, tasks, std::move(fun));
}

/**
 * This function is only available if `TATAMI_PYTHON_PARALLELIZE_UNKNOWN` is defined.
 * Applications can override this by defining a `TATAMI_PYTHON_SERIALIZE` function-like macro,
 * which should accept a function object and execute it in some serial context.
 *
 * @tparam Function_ Function that accepts no arguments.
 * @param fun Function to be evaluated after the GIL is acquired.
 * This typically involves calls to the Python interpreter or API.
 */
template<typename Function_>
void lock(Function_ fun) {
    std::optional<pybind11::gil_scoped_acquire> gil;
    if (!PyGILState_Check()) {
        gil.emplace();
    }
    fun();
}

}

/**
 * @cond
 */
#endif
/**
 * @endcond
 */

#endif
