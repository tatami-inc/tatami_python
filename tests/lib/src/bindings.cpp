#include <vector>
#include <algorithm>
#include <iostream>
#include <cstdint>

#ifdef TEST_CUSTOM_PARALLEL
#define TATAMI_PYTHON_PARALLELIZE_UNKNOWN
#include "tatami_python/parallelize.hpp"
#define TATAMI_CUSTOM_PARALLEL tatami_python::parallelize
#endif

#include "tatami_python/tatami_python.hpp"

typedef tatami::Matrix<double, std::int32_t> TestMatrix;

void free_test(std::uintptr_t ptr0) {
    auto ptr = reinterpret_cast<TestMatrix*>(reinterpret_cast<void*>(ptr0));
    delete ptr;
    return;
}

std::uintptr_t parse_test(pybind11::object seed, double cache_size, bool require_min) {
    tatami_python::UnknownMatrixOptions opt;
    opt.maximum_cache_size = cache_size;
    opt.require_minimum_cache = require_min;
    auto optr = new tatami_python::UnknownMatrix<double, std::int32_t>(std::move(seed), opt);
    return reinterpret_cast<std::uintptr_t>(static_cast<void*>(static_cast<TestMatrix*>(optr)));
}

int nrow_test(std::uintptr_t ptr0) {
    return reinterpret_cast<TestMatrix*>(ptr0)->nrow();
}

int ncol_test(std::uintptr_t ptr0) {
    return reinterpret_cast<TestMatrix*>(ptr0)->ncol();
}

bool prefer_rows_test(std::uintptr_t ptr0) {
    return reinterpret_cast<TestMatrix*>(ptr0)->prefer_rows();
}

bool is_sparse_test(std::uintptr_t ptr0) {
    return reinterpret_cast<TestMatrix*>(ptr0)->is_sparse();
}

/******************
 *** Dense full ***
 ******************/

std::shared_ptr<tatami::FixedVectorOracle<std::int32_t> > create_oracle(const pybind11::array_t<std::int32_t>& idx) {
    auto iptr = static_cast<const std::int32_t*>(idx.request().ptr);
    return std::make_shared<tatami::FixedVectorOracle<std::int32_t> >(std::vector<std::int32_t>(iptr, iptr + idx.size()));
}

template<bool oracle_, class Extractor_>
pybind11::array_t<double> format_dense_output(Extractor_& ext, const std::int32_t i, const std::int32_t len) {
    pybind11::array_t<double> vec(len);
    auto optr = static_cast<double*>(vec.request().ptr); 
    auto iptr = [&]() {
        if constexpr(oracle_) {
            return ext.fetch(optr);
        } else {
            return ext.fetch(i, optr);
        }
    }();
    tatami::copy_n(iptr, len, optr);
    return vec;
}

template<bool sparse_, bool oracle_>
auto create_extractor(
    const TestMatrix& mat,
    const bool row,
    const pybind11::array_t<std::int32_t>& idx,
    const bool needs_value,
    const bool needs_index
) {
    tatami::Options opt;
    opt.sparse_extract_value = needs_value;
    opt.sparse_extract_index = needs_index;
    if constexpr(oracle_) {
        return tatami::new_extractor<sparse_, oracle_>(mat, row, create_oracle(idx), opt);
    } else {
        return tatami::new_extractor<sparse_, oracle_>(mat, row, false, opt);
    }
}

template<bool oracle_>
pybind11::list dense_full(
    const std::uintptr_t ptr0,
    const bool row,
    const pybind11::array_t<std::int32_t>& idx
) {
    const auto ptr = reinterpret_cast<TestMatrix*>(ptr0);
    auto ext = create_extractor<false, oracle_>(*ptr, row, idx, true, true);
    const auto secondary = (!row ? ptr->nrow() : ptr->ncol());
    const auto iptr = static_cast<const std::int32_t*>(idx.request().ptr);
    const std::size_t num = idx.size();

    pybind11::list output(num);
    for (std::size_t i = 0; i < num; ++i) {
        output[i] = format_dense_output<oracle_>(*ext, iptr[i], secondary);
    }
    return output;
}

pybind11::list myopic_dense_full(
    const std::uintptr_t ptr0,
    const bool row,
    const pybind11::array_t<std::int32_t>& idx
) {
    return dense_full<false>(ptr0, row, idx);
}

pybind11::list oracular_dense_full(
    const std::uintptr_t ptr0,
    const bool row,
    const pybind11::array_t<std::int32_t>& idx
) {
    return dense_full<true>(ptr0, row, idx);
}

/*******************
 *** Dense block ***
 *******************/

template<bool sparse_, bool oracle_>
auto create_extractor(
    const TestMatrix& mat,
    const bool row,
    const pybind11::array_t<std::int32_t>& idx,
    const std::int32_t first,
    const std::int32_t len,
    const bool needs_value,
    const bool needs_index
) {
    tatami::Options opt;
    opt.sparse_extract_value = needs_value;
    opt.sparse_extract_index = needs_index;
    if constexpr(oracle_) {
        return tatami::new_extractor<sparse_, oracle_>(mat, row, create_oracle(idx), first, len, opt);
    } else {
        return tatami::new_extractor<sparse_, oracle_>(mat, row, false, first, len, opt);
    }
}

template<bool oracle_>
pybind11::list dense_block(
    const std::uintptr_t ptr0,
    const bool row,
    const pybind11::array_t<std::int32_t>& idx,
    const std::int32_t first,
    const std::int32_t len
) {
    auto ptr = reinterpret_cast<TestMatrix*>(ptr0);
    const auto secondary = (!row ? ptr->nrow() : ptr->ncol());
    auto ext = create_extractor<false, oracle_>(*ptr, row, idx, first, len, true, true);
    const auto iptr = static_cast<const std::int32_t*>(idx.request().ptr);
    const std::size_t num = idx.size();

    pybind11::list output(num);
    for (std::size_t i = 0; i < num; ++i) {
        output[i] = format_dense_output<oracle_>(*ext, iptr[i], len);
    }
    return output;
}

pybind11::list myopic_dense_block(
    const std::uintptr_t ptr0,
    const bool row,
    const pybind11::array_t<std::int32_t>& idx,
    const std::int32_t first,
    const std::int32_t len
) {
    return dense_block<false>(ptr0, row, idx, first, len);
}

pybind11::list oracular_dense_block(
    const std::uintptr_t ptr0,
    const bool row,
    const pybind11::array_t<std::int32_t>& idx,
    const std::int32_t first,
    const std::int32_t len
) {
    return dense_block<true>(ptr0, row, idx, first, len);
}

/********************
 *** Dense subset ***
 ********************/

template<bool sparse_, bool oracle_>
auto create_extractor(
    const TestMatrix& mat,
    const bool row,
    const pybind11::array_t<std::int32_t>& idx,
    const pybind11::array_t<std::int32_t>& subset,
    const bool needs_value,
    const bool needs_index
) {
    tatami::Options opt;
    opt.sparse_extract_value = needs_value;
    opt.sparse_extract_index = needs_index;

    auto sptr = static_cast<const std::int32_t*>(subset.request().ptr);
    auto subs = std::make_shared<std::vector<std::int32_t> >(sptr, sptr + subset.size());

    if constexpr(oracle_) {
        return tatami::new_extractor<sparse_, oracle_>(mat, row, create_oracle(idx), std::move(subs), opt);
    } else {
        return tatami::new_extractor<sparse_, oracle_>(mat, row, false, std::move(subs), opt);
    }
}

template<bool oracle_>
pybind11::list dense_indexed(
    const std::uintptr_t ptr0,
    const bool row,
    const pybind11::array_t<std::int32_t>& idx,
    const pybind11::array_t<std::int32_t>& subset
) {
    auto ptr = reinterpret_cast<TestMatrix*>(ptr0);
    const auto secondary = (!row ? ptr->nrow() : ptr->ncol());
    auto ext = create_extractor<false, oracle_>(*ptr, row, idx, subset, true, true);
    const auto iptr = static_cast<const std::int32_t*>(idx.request().ptr);
    const std::size_t num = idx.size();

    pybind11::list output(num);
    for (std::size_t i = 0; i < num; ++i) {
        output[i] = format_dense_output<oracle_>(*ext, iptr[i], subset.size());
    }
    return output;
}

pybind11::list myopic_dense_indexed(
    const std::uintptr_t ptr0,
    const bool row,
    const pybind11::array_t<std::int32_t>& idx,
    const pybind11::array_t<std::int32_t>& subset
) {
    return dense_indexed<false>(ptr0, row, idx, subset);
}

pybind11::list oracular_dense_indexed(
    const std::uintptr_t ptr0,
    const bool row,
    const pybind11::array_t<std::int32_t>& idx,
    const pybind11::array_t<std::int32_t>& subset
) {
    return dense_indexed<true>(ptr0, row, idx, subset);
}

/*******************
 *** Sparse full ***
 *******************/

template<bool oracle_, class Extractor_>
pybind11::object format_sparse_output(
    Extractor_& ext,
    const std::int32_t i,
    double* const vbuffer,
    std::int32_t* const ibuffer,
    const bool needs_value,
    const bool needs_index
) {
    if (needs_index && needs_value) {
        auto x = [&](){
            if constexpr(oracle_) {
                return ext.fetch(vbuffer, ibuffer);
            } else {
                return ext.fetch(i, vbuffer, ibuffer);
            }
        }();

        pybind11::dict output;
        output["value"] = pybind11::array_t<double>(x.number, x.value);
        output["index"] = pybind11::array_t<std::int32_t>(x.number, x.index);
        return output;

    } else if (needs_index) {
        auto x = [&](){
            if constexpr(oracle_) {
                return ext.fetch(NULL, ibuffer);
            } else {
                return ext.fetch(i, NULL, ibuffer);
            }
        }();
        return pybind11::array_t<std::int32_t>(x.number, x.index);

    } else if (needs_value) {
        auto x = [&](){
            if constexpr(oracle_) {
                return ext.fetch(vbuffer, NULL);
            } else {
                return ext.fetch(i, vbuffer, NULL);
            }
        }();
        return pybind11::array_t<double>(x.number, x.value);

    } else {
        auto x = [&](){
            if constexpr(oracle_) {
                return ext.fetch(NULL, NULL);
            } else {
                return ext.fetch(i, NULL, NULL);
            }
        }();
        return pybind11::cast(x.number);
    }
}

template<bool oracle_>
pybind11::list sparse_full(
    const std::uintptr_t ptr0,
    const bool row,
    const pybind11::array_t<std::int32_t>& idx,
    const bool needs_value,
    const bool needs_index
) {
    auto ptr = reinterpret_cast<TestMatrix*>(ptr0);
    auto ext = create_extractor<true, oracle_>(*ptr, row, idx, needs_value, needs_index);
    const auto secondary = (!row ? ptr->nrow() : ptr->ncol());
    std::vector<double> vbuffer(secondary);
    std::vector<std::int32_t> ibuffer(secondary);

    const auto iptr = static_cast<const std::int32_t*>(idx.request().ptr);
    const std::size_t num = idx.size();
    pybind11::list output(num);
    for (std::size_t i = 0; i < num; ++i) {
        output[i] = format_sparse_output<oracle_>(*ext, iptr[i], vbuffer.data(), ibuffer.data(), needs_value, needs_index);
    }

    return output;
}

pybind11::list myopic_sparse_full(
    const std::uintptr_t ptr0,
    const bool row,
    const pybind11::array_t<std::int32_t>& idx,
    const bool needs_value,
    const bool needs_index
) {
    return sparse_full<false>(ptr0, row, idx, needs_value, needs_index);
}

pybind11::list oracular_sparse_full(
    const std::uintptr_t ptr0,
    const bool row,
    const pybind11::array_t<std::int32_t>& idx,
    const bool needs_value,
    const bool needs_index
) {
    return sparse_full<true>(ptr0, row, idx, needs_value, needs_index);
}

/********************
 *** Sparse block ***
 ********************/

template<bool oracle_>
pybind11::list sparse_block(
    const std::uintptr_t ptr0,
    const bool row,
    const pybind11::array_t<std::int32_t>& idx,
    const std::int32_t first,
    const std::int32_t len,
    const bool needs_value,
    const bool needs_index
) {
    auto ptr = reinterpret_cast<TestMatrix*>(reinterpret_cast<void*>(ptr0));
    auto ext = create_extractor<true, oracle_>(*ptr, row, idx, first, len, needs_value, needs_index);
    std::vector<double> vbuffer(len);
    std::vector<std::int32_t> ibuffer(len);

    const auto iptr = static_cast<const std::int32_t*>(idx.request().ptr);
    const std::size_t num = idx.size();
    pybind11::list output(num);
    for (std::size_t i = 0; i < num; ++i) {
        output[i] = format_sparse_output<oracle_>(*ext, iptr[i], vbuffer.data(), ibuffer.data(), needs_value, needs_index);
    }
    return output;
}

pybind11::list myopic_sparse_block(
    const std::uintptr_t ptr0,
    const bool row,
    const pybind11::array_t<std::int32_t>& idx,
    const std::int32_t first,
    const std::int32_t len,
    const bool needs_value,
    const bool needs_index
) {
    return sparse_block<false>(ptr0, row, idx, first, len, needs_value, needs_index);
}

pybind11::list oracular_sparse_block(
    const std::uintptr_t ptr0,
    const bool row,
    const pybind11::array_t<std::int32_t>& idx,
    const std::int32_t first,
    const std::int32_t len,
    const bool needs_value,
    const bool needs_index
) {
    return sparse_block<true>(ptr0, row, idx, first, len, needs_value, needs_index);
}

/**********************
 *** Sparse indexed ***
 **********************/

template<bool oracle_>
pybind11::list sparse_indexed(
    const std::uintptr_t ptr0,
    const bool row,
    const pybind11::array_t<std::int32_t>& idx,
    const pybind11::array_t<std::int32_t>& subset,
    const bool needs_value,
    const bool needs_index
) {
    auto ptr = reinterpret_cast<TestMatrix*>(reinterpret_cast<void*>(ptr0));
    auto ext = create_extractor<true, oracle_>(*ptr, row, idx, subset, needs_value, needs_index);
    std::vector<double> vbuffer(subset.size());
    std::vector<std::int32_t> ibuffer(subset.size());

    const auto iptr = static_cast<const std::int32_t*>(idx.request().ptr);
    const std::size_t num = idx.size();
    pybind11::list output(num);
    for (std::size_t i = 0; i < num; ++i) {
        output[i] = format_sparse_output<oracle_>(*ext, iptr[i], vbuffer.data(), ibuffer.data(), needs_value, needs_index);
    }
    return output;
}

pybind11::list myopic_sparse_indexed(
    const std::uintptr_t ptr0,
    const bool row,
    const pybind11::array_t<std::int32_t>& idx,
    const pybind11::array_t<std::int32_t>& subset,
    const bool needs_value,
    const bool needs_index
) {
    return sparse_indexed<false>(ptr0, row, idx, subset, needs_value, needs_index);
}

pybind11::list oracular_sparse_indexed(
    std::uintptr_t ptr0,
    const bool row,
    const pybind11::array_t<std::int32_t>& idx,
    const pybind11::array_t<std::int32_t>& subset,
    const bool needs_value,
    const bool needs_index
) {
    return sparse_indexed<true>(ptr0, row, idx, subset, needs_value, needs_index);
}

/****************
 *** Row sums ***
 ****************/

pybind11::array_t<double> collapse_vector(const std::vector<std::vector<double> >& temp) {
    std::size_t total = 0;
    for (const auto& t : temp) {
        total += t.size();
    }

    pybind11::array_t<double> output(total);
    auto start = static_cast<double*>(output.request().ptr);
    for (const auto& t : temp) {
        std::copy(t.begin(), t.end(), start);
        start += t.size();
    }

    return output;
}

template<bool oracle_>
pybind11::array_t<double> dense_sums(
    const std::uintptr_t ptr0,
    const bool row,
    const std::int32_t num_threads
) {
    auto ptr = reinterpret_cast<TestMatrix*>(reinterpret_cast<void*>(ptr0));
    const auto primary = (row ? ptr->nrow() : ptr->ncol());
    const auto secondary = (!row ? ptr->nrow() : ptr->ncol());

#ifdef TEST_CUSTOM_PARALLEL
#ifdef TATAMI_PYTHON_PARALLELIZE_UNKNOWN 
    pybind11::gil_scoped_acquire gillock();
#endif

    std::vector<std::vector<double> > output(num_threads);
    tatami::parallelize([&](    
        std::int32_t w,
        std::int32_t start,
        std::int32_t len
    ) -> void {
        auto ext = [&]() {
            if constexpr(oracle_) {
                return tatami::new_extractor<false, oracle_>(ptr, row, std::make_shared<tatami::ConsecutiveOracle<std::int32_t> >(start, len));
            } else {
                return tatami::new_extractor<false, oracle_>(ptr.get(), row, false);
            }
        }();

        std::vector<double> buffer(secondary);
        auto& mine = output[w];
        mine.resize(len);

        for (std::int32_t i = 0; i < len; ++i) {
            auto iptr = [&]() {
                if constexpr(oracle_) {
                    return ext->fetch(buffer.data());
                } else {
                    return ext->fetch(i + start, buffer.data());
                }
            }();
            mine[i] = std::accumulate(iptr, iptr + secondary, 0.0);
        }
    }, primary, num_threads);

    return collapse_vector(output);

#else
    auto ext = [&]() {
        if constexpr(oracle_) {
            return tatami::new_extractor<false, oracle_>(*ptr, row, std::make_shared<tatami::ConsecutiveOracle<std::int32_t> >(0, primary));
        } else {
            return tatami::new_extractor<false, oracle_>(*ptr, row, false);
        }
    }();

    std::vector<double> buffer(secondary);
    pybind11::array_t<double> output(primary);
    auto optr = static_cast<double*>(output.request().ptr);

    for (std::int32_t i = 0; i < primary; ++i) {
        auto iptr = [&]() {
            if constexpr(oracle_) {
                return ext->fetch(buffer.data());
            } else {
                return ext->fetch(i, buffer.data());
            }
        }();
        optr[i] = std::accumulate(iptr, iptr + secondary, 0.0);
    }

    return output;
#endif
}

pybind11::array_t<double> myopic_dense_sums(
    const std::uintptr_t ptr0,
    const bool row,
    const std::int32_t num_threads
) {
    return dense_sums<false>(ptr0, row, num_threads);
}

pybind11::array_t<double> oracular_dense_sums(
    const std::uintptr_t ptr0,
    const bool row,
    const std::int32_t num_threads
) {
    return dense_sums<true>(ptr0, row, num_threads);
}

template<bool oracle_>
pybind11::array_t<double> sparse_sums(
    const std::uintptr_t ptr0,
    const bool row,
    const std::int32_t num_threads
) {
    auto ptr = reinterpret_cast<TestMatrix*>(reinterpret_cast<void*>(ptr0));
    const auto primary = (row ? ptr->nrow() : ptr->ncol());
    const auto secondary = (!row ? ptr->nrow() : ptr->ncol());

#ifdef TEST_CUSTOM_PARALLEL
#ifdef TATAMI_PYTHON_PARALLELIZE_UNKNOWN 
    pybind11::gil_scoped_release gillock();
#endif

    std::vector<std::vector<double> > output(num_threads);
    tatami::parallelize([&](
        std::int32_t w,
        const std::int32_t start,
        const std::int32_t len
    ) -> void {
        auto ext = [&]() {
            if constexpr(oracle_) {
                return tatami::new_extractor<true, oracle_>(*ptr, row, std::make_shared<tatami::ConsecutiveOracle<std::int32_t> >(start, len));
            } else {
                return tatami::new_extractor<true, oracle_>(*ptr, row, false);
            }
        }();

        std::vector<double> vbuffer(secondary);
        std::vector<std::int32_t> ibuffer(secondary);
        auto& mine = output[w];
        mine.resize(len);

        for (std::int32_t i = 0; i < len; ++i) {
            auto range = [&]() {
                if constexpr(oracle_) {
                    return ext->fetch(vbuffer.data(), ibuffer.data());
                } else {
                    return ext->fetch(i + start, vbuffer.data(), ibuffer.data());
                }
            }();
            mine[i] = std::accumulate(range.value, range.value + range.number, 0.0);
        }
    }, primary, num_threads);

    return collapse_vector(output);

#else
    auto ext = [&]() {
        if constexpr(oracle_) {
            return tatami::new_extractor<true, oracle_>(*ptr, row, std::make_shared<tatami::ConsecutiveOracle<std::int32_t> >(0, primary));
        } else {
            return tatami::new_extractor<true, oracle_>(*ptr, row, false);
        }
    }();

    std::vector<double> vbuffer(secondary);
    std::vector<std::int32_t> ibuffer(secondary);
    pybind11::array_t<double> output(primary);
    auto optr = reinterpret_cast<double*>(output.request().ptr);

    for (std::int32_t i = 0; i < primary; ++i) {
        auto range = [&]() {
            if constexpr(oracle_) {
                return ext->fetch(vbuffer.data(), ibuffer.data());
            } else {
                return ext->fetch(i, vbuffer.data(), ibuffer.data());
            }
        }();
        optr[i] = std::accumulate(range.value, range.value + range.number, 0.0);
    }

    return output;
#endif
}

pybind11::array_t<double> myopic_sparse_sums(
    const std::uintptr_t ptr0,
    const bool row,
    const std::int32_t num_threads
) {
    return sparse_sums<false>(ptr0, row, num_threads);
}

pybind11::array_t<double> oracular_sparse_sums(
    const std::uintptr_t ptr0,
    const bool row,
    const std::int32_t num_threads
) {
    return sparse_sums<true>(ptr0, row, num_threads);
}

PYBIND11_MODULE(lib_tatami_python_test, m) {
    m.def("free_test", &free_test);
    m.def("nrow_test", &nrow_test);
    m.def("ncol_test", &ncol_test);
    m.def("prefer_rows_test", &prefer_rows_test);
    m.def("is_sparse_test", &is_sparse_test);

    m.def("myopic_dense_full", &myopic_dense_full);
    m.def("oracular_dense_full", &oracular_dense_full);
    m.def("myopic_dense_block", &myopic_dense_block);
    m.def("oracular_dense_block", &oracular_dense_block);
    m.def("myopic_dense_indexed", &myopic_dense_indexed);
    m.def("oracular_dense_indexed", &oracular_dense_indexed);

    m.def("myopic_sparse_full", &myopic_sparse_full);
    m.def("oracular_sparse_full", &oracular_sparse_full);
    m.def("myopic_sparse_block", &myopic_sparse_block);
    m.def("oracular_sparse_block", &oracular_sparse_block);
    m.def("myopic_sparse_indexed", &myopic_sparse_indexed);
    m.def("oracular_sparse_indexed", &oracular_sparse_indexed);

    m.def("myopic_dense_sums", &myopic_dense_sums);
    m.def("oracular_dense_sums", &oracular_dense_sums);
    m.def("myopic_sparse_sums", &myopic_sparse_sums);
    m.def("oracular_sparse_sums", &oracular_sparse_sums);
}
