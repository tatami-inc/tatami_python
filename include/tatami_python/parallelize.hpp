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
/**
 * Macro function that accepts a function object and executes it in a serial context.
 */
#define TATAMI_PYTHON_SERIALIZE ::tatami_python::lock
#endif 

/**
 * @file parallelize.hpp
 * @brief Utilities for safe parallelization.
 */

namespace tatami_python {

/**
 * Replacement for `tatami::parallelize()` that applies a function to a set of tasks in parallel, usually for iterating over a dimension of a `Matrix`.
 * This releases the Python GIL so that it can be re-acquired by `UnknownMatrix` extractors in each individual worker.
 *
 * @tparam Function_ Function to be applied to a contiguous range of tasks.
 * This should accept three arguments:
 * - `worker`, the worker ID executing this task range.
 *   This will be passed as an `int` in `[0, workers)`.
 * - `task_start`, the start index of the task range.
 *   This will be passed as an `Index_` in `[0, tasks)`.
 * - `task_length`, the number of tasks in the task range.
 *   This will be passed as an `Index_` in `(0, tasks)`, i.e., it is always positive.
 * @tparam Index_ Integer type for the number of tasks.
 *
 * @param fun Function that executes a contiguous range of tasks.
 * This will be called no more than once in each worker with a different non-overlapping range, where the union of all ranges will cover `[0, tasks)`. 
 * @param tasks Number of tasks.
 * This should be non-negative.
 * @param workers Number of workers.
 * This should be positive.
 *
 * @return The number of workers (`K`) that were actually used.
 * `K` is guaranteed to be no greater than `workers` (or 1, if `workers` is not positive).
 * `fun()` will have been called once for each of the worker IDs `[0, ..., K - 1]`.
 */
template<class Function_, class Index_>
int parallelize(const Function_ fun, const Index_ tasks, int workers) {
    std::optional<pybind11::gil_scoped_release> ungil;
    if (PyGILState_Check()) {
        ungil.emplace();
    }
    return subpar::parallelize_range(workers, tasks, std::move(fun));
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
