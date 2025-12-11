#ifndef TATAMI_PYTHON_PARALLELIZE_HPP
#define TATAMI_PYTHON_PARALLELIZE_HPP

/**
 * @cond
 */
#ifdef TATAMI_PYTHON_PARALLELIZE_UNKNOWN
/**
 * @endcond
 */

#include <mutex>

#ifndef TATAMI_PYTHON_SERIALIZE
#define TATAMI_PYTHON_SERIALIZE ::tatami_python::lock
#endif 

/**
 * @file parallelize.hpp
 * @brief Utilities for safe parallelization.
 */

namespace tatami_python {

/**
 * @cond
 */
inline std::mutex* mut_ptr = NULL;
/**
 * @endcond
 */

/**
 * Retrieve a global mutex object for all **tatami_python** applications.
 * This function is only available if `TATAMI_PYTHON_PARALLELIZE_UNKNOWN` is defined.
 *
 * @return Global mutex for locking all calls to the Python interpreter.
 * If `set_mutex()` was called with a non-`NULL` pointer, the provided instance will be used;
 * otherwise, a default instance will be instantiated.
 */
inline std::mutex& mutex() {
    if (mut_ptr) {
        return *mut_ptr;
    } else {
        // In theory, this should end up resolving to a single instance, even across dynamically linked libraries:
        // https://stackoverflow.com/questions/52851239/local-static-variable-linkage-in-a-template-class-static-member-function
        // In practice, this doesn't seem to be the case on a Mac, requiring us to use `set_mutex()`.
        static std::mutex mut;
        return mut;
    }
}

/**
 * Set a global mutex for all **tatami_python** applications.
 * This function is only available if `TATAMI_PYTHON_PARALLELIZE_UNKNOWN` is defined.
 * Calling this function is occasionally necessary if `mutex()` resolves to different instances of a `std::mutex` across different libraries.
 *
 * @param Pointer to a different global mutex, or `NULL` to unset this pointer.
 */
inline void set_mutex(std::mutex* ptr) {
    mut_ptr = ptr;
}

/**
 * This function is only available if `TATAMI_PYTHON_PARALLELIZE_UNKNOWN` is defined.
 * Applications can override this by defining a `TATAMI_PYTHON_SERIALIZE` function-like macro,
 * which should accept a function object and execute it in some serial context.
 *
 * @tparam Function_ Function that accepts no arguments.
 * @param fun Function to be evaluated after `mutex()` is locked.
 * This typically involves calls to the Python interpreter or API.
 */
template<typename Function_>
void lock(Function_ fun) {
    auto& mut = mutex();
    std::lock_guard lck(mut);
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
