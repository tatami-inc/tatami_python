# Parse Python objects via tatami 

![Unit tests](https://github.com/tatami-inc/tatami_python/actions/workflows/run-tests.yaml/badge.svg)
![Documentation](https://github.com/tatami-inc/tatami_python/actions/workflows/doxygenate.yaml/badge.svg)

## Overview

**tatami_python** is an header-only library for reading abstract Python matrices in [**tatami**](https://github.com/tatami-inc/tatami).
This allows **tatami**-based C++ functions to accept and operate on any matrix-like Python object containing numeric data.
Usage is as simple as:

```cpp
#include "tatami_pybind/tatami_pybind.hpp"

pybind11::object some_typical_pybind_function(const pybind11::object& x) {
    auto ptr = std::make_shared<tatami_python::UnknownMatrix<double, int> >(x);

    // Do stuff with the tatami::Matrix.
    ptr->nrow();
    auto row_extractor = ptr->dense_row();
    auto first_row = row_extractor->fetch(0);

    // Return something.
    return pybind11::none();
}
```

For more details, check out the [reference documentation](https://tatami-inc.github.io/tatami_python).

## Implementation

**tatami_r** assumes that the [**delayedarray**](https://github.com/BiocPy/delayedarray) package is installed.
The `UnknownMatrix` getters will use the `extract_dense_array()` and `extract_sparse_array()` Python functions to retrieve data from the abstract Python matrix.
Obviously, this involves calling into Python from C++, so high performance should not be expected here.
Rather, the purpose of **tatami_python** is to ensure that **tatami**-based functions keep working when a native implementation cannot be found for a Python matrix.

## Enabling parallelization

We enable thread-safe execution by defining the `TATAMI_PYTHON_PARALLELIZE_UNKNOWN` macro.
This instructs **tatami_python** to momentarily acquire the GIL whenever a thread needs to call a Python function.

```cpp
// Set up these macros before including any tatami libraries.
#define TATAMI_PYTHON_PARALLELIZE_UNKNOWN
#define TATAMI_CUSTOM_PARALLEL ::tatami_python::parallelize
```

Developers should release the GIL before any parallel section that might involve accessing an `UnknownMatrix`, otherwise all other threads would be blocked by the main thread.
This is facilitated by the `tatami_python::parallelize()` function, which is a drop-in replacement for `tatami::parallelize()` that releases the GIL before doing any parallel work.
By setting the `TATAMI_CUSTOM_PARALLEL` macro, we ensure that all calls to `tatami::parallelize()` will automatically use `tatami_python::parallelize()`:

```cpp
tatami::parallelize([&](int thread_id, int start, int len) -> void {
    // Do something with the UnknownMatrix.
    auto ext = ptr->dense_row();
    std::vector<double> buffer(ptr->ncol());
    for (int r = start, end = start + len; start < end; ++r) {
        auto out = ext->fetch(r, buffer.data());
        // Do something with each row.
    }
}, ptr->nrow(), num_threads);
```

Needless to say, the use of the GIL means that the Python calls are strictly serial, regardless of the number of threads requested in `tatami::parallelize()`.

## Deployment

**tatami_python** is intended to be compiled with other relevant C++ code inside an Python package using [**pybind11**](https://github.com/pybind/pybind11).
This is most easily done by linking to the [**mattress**](https://github.com/tatami-inc/mattress) and [**assorthead**](https://github.com/BiocPy/assorthead) packages,
follow the links for more instructions.

If **assorthead** or **mattress** cannot be used, the Python package developer will need to acquire the contents of the `include/` directory
(along with all dependencies in [`extern/CMakeLists.txt`](extern/CMakeLists.txt))
and make them available during package compilation.
