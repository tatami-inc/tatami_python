import numpy
import delayedarray
import random
import tatami_python_test


def get_cache_size(mat, cache_fraction):
    return cache_fraction * mat.shape[0] * mat.shape[1] * mat.dtype.itemsize


def create_predictions(iterdim, step, mode):
    if iterdim == 0:
        seq = []
    else:
        seq = list(range(0, iterdim, step))
        if mode == "reverse":
            seq = seq[::-1]
        elif mode == "random":
            random.shuffle(seq)
    return numpy.array(seq, dtype=numpy.dtype("int32"))


def pretty_name(prefix, params):
    collected = []
    for k, v in params.items():
        collected.append(k + "=" + str(v))
    return prefix + "[" + ", ".join(collected) + "]"


def create_expected_dense(mat, row, iseq, keep):
    all_expected = []
    for i, j in enumerate(iseq):
        if row:
            expected = mat[j,:]
        else:
            expected = mat[:,j]
        if keep is not None:
            expected = expected[keep]
        all_expected.append(expected.astype("double"))
    return all_expected


def fill_sparse(observed, otherdim, keep):
    if keep is not None:
        keepmap = {}
        for i, j in enumerate(keep):
            keepmap[j] = i

    copy = []
    for i, both in enumerate(observed):
        idx = both["index"]
        if keep is not None:
            vec = numpy.zeros(len(keep), dtype=numpy.dtype("double"))
            for j, x in enumerate(both["value"]):
                vec[keepmap[idx[j]]] = x
        else:
            vec = numpy.zeros(otherdim, dtype=numpy.dtype("double"))
            for j, x in enumerate(both["value"]):
                vec[idx[j]] = x
        copy.append(vec)

    return copy


def _expand_grid(keys, values, position, current):
    if len(keys) == position:
        yield current
        return
    name = keys[position]
    for x in values[position]:
        current[name] = x
        for y in _expand_grid(keys, values, position + 1, current):
            yield y


def expand_grid(params):
    all_keys = list(params.keys())
    all_vals = list(params.values())
    for thing in _expand_grid(all_keys, all_vals, 0, {}):
        yield thing


def compare_list_of_vectors(x, y):
    assert len(x) == len(y)
    for i in range(len(x)):
        assert (x[i] == y[i]).all()


def full_test_suite(mat, cache_fractions):
    scenarios = expand_grid({
        "cache": cache_fractions,
        "row": [True, False],
        "oracle": [False, True],
        "mode": ["forward", "reverse", "random"],
        "step": [1, 5]
    })

    for scen in scenarios:
        print(pretty_name("dense full ", scen))
        cache = scen["cache"]
        row = scen["row"]
        oracle = scen["oracle"]
        mode = scen["mode"]
        step = scen["step"]

        iterdim = mat.shape[1 - int(row)]
        otherdim = mat.shape[int(row)]
        iseq = create_predictions(iterdim, step, mode)

        cache_size = get_cache_size(mat, cache)
        ptr = tatami_python_test.WrappedMatrix(mat, cache_size, cache_size > 0)

        all_expected = create_expected_dense(mat, row, iseq, None)
        extracted = ptr.extract_dense(row, iseq, None, oracle)
        compare_list_of_vectors(extracted, all_expected)

        print(pretty_name("sparse full ", scen))
        extracted_sparse = ptr.extract_sparse(row, iseq, None, oracle, needs_value=True, needs_index=True)
        compare_list_of_vectors(fill_sparse(extracted_sparse, otherdim, None), all_expected)

        extracted_index = ptr.extract_sparse(row, iseq, None, oracle, needs_value=False, needs_index=True)
        compare_list_of_vectors(extracted_index, [y["index"] for y in extracted_sparse])

        extracted_value = ptr.extract_sparse(row, iseq, None, oracle, needs_value=True, needs_index=False)
        compare_list_of_vectors(extracted_value, [y["value"] for y in extracted_sparse])

        extracted_n = ptr.extract_sparse(row, iseq, None, oracle, needs_value=False, needs_index=False)
        assert extracted_n == [len(y["value"]) for y in extracted_sparse]

        if ptr.is_sparse():
            prod = mat.nrow() * mat.ncol()
            if prod > 0:
                assert prod > sum(extracted_n)


#block_test_suite <- function(mat, cache.fraction) {
#    cache.size <- get_cache_size(mat, cache.fraction)
#    ptr <- raticate.tests::parse(mat, cache.size, cache.size > 0)
#
#    scenarios <- expand.grid(
#        cache = cache.fraction,
#        row = c(TRUE, FALSE),
#        oracle = c(FALSE, TRUE),
#        mode = c("forward", "reverse"), 
#        step = c(1, 5),
#        block = list(c(0, 0.3), c(0.2, 0.66), c(0.6, 0.37)),
#        stringsAsFactors=FALSE
#    )
#
#    for (i in seq_len(nrow(scenarios))) {
#        row <- scenarios[i,"row"]
#        oracle <- scenarios[i,"oracle"]
#        mode <- scenarios[i, "mode"]
#        step <- scenarios[i,"step"]
#        block <- scenarios[i,"block"][[1]]
#
#        iterdim <- if (row) nrow(mat) else ncol(mat) 
#        otherdim <- if (row) ncol(mat) else nrow(mat)
#        iseq <- create_predictions(iterdim, step, mode)
#
#        bstart <- floor(block[[1]] * otherdim) + 1L
#        blen <- floor(block[[2]] * otherdim)
#        keep <- (bstart - 1L) + seq_len(blen)
#        all.expected <- create_expected_dense(mat, row, iseq, keep)
#
#        test_that(pretty_name("dense block ", scenarios[i,]), {
#            if (oracle) {
#                extracted <- raticate.tests::oracular_dense_block(ptr, row, iseq, bstart, blen) 
#            } else {
#                extracted <- raticate.tests::myopic_dense_block(ptr, row, iseq, bstart, blen)
#            }
#            expect_identical(extracted, all.expected)
#        })
#
#        test_that(pretty_name("sparse block ", scenarios[i,]), {
#            if (oracle) {
#                FUN <- raticate.tests::oracular_sparse_block
#            } else {
#                FUN <- raticate.tests::myopic_sparse_block
#            }
#
#            extractor.b <- FUN(ptr, row, iseq, bstart, blen, TRUE, TRUE)
#            expect_identical(all.expected, fill_sparse(extractor.b, otherdim, keep))
#            extractor.i <- FUN(ptr, row, iseq, bstart, blen, FALSE, TRUE)
#            expect_identical(extractor.i, lapply(extractor.b, function(y) y$index))
#            extractor.v <- FUN(ptr, row, iseq, bstart, blen, TRUE, FALSE)
#            expect_identical(extractor.v, lapply(extractor.b, function(y) y$value))
#            extractor.n <- unlist_to_integer(FUN(ptr, row, iseq, bstart, blen, FALSE, FALSE))
#            expect_identical(extractor.n, lengths(extractor.v))
#
#            if (DelayedArray::is_sparse(mat)) {
#                prod <- nrow(mat) * ncol(mat)
#                if (prod > 0) {
#                    expect_true(sum(extractor.n) < prod)
#                }
#            }
#        })
#    }
#}
#
#index_test_suite <- function(mat, cache.fraction) {
#    cache.size <- get_cache_size(mat, cache.fraction)
#    ptr <- raticate.tests::parse(mat, cache.size, cache.size > 0)
#
#    scenarios <- expand.grid(
#        cache = cache.fraction,
#        row = c(TRUE, FALSE),
#        oracle = c(FALSE, TRUE),
#        mode = c("forward", "reverse"), 
#        step = c(1, 5),
#        index = list(c(0, 3), c(0.33, 4), c(0.5, 5)),
#        stringsAsFactors=FALSE
#    )
#
#    for (i in seq_len(nrow(scenarios))) {
#        row <- scenarios[i,"row"]
#        oracle <- scenarios[i,"oracle"]
#        mode <- scenarios[i, "mode"]
#        step <- scenarios[i,"step"]
#        index_params <- scenarios[i,"index"][[1]]
#
#        iterdim <- if (row) nrow(mat) else ncol(mat) 
#        otherdim <- if (row) ncol(mat) else nrow(mat)
#        iseq <- create_predictions(iterdim, step, mode)
#
#        istart <- floor(index_params[[1]] * otherdim) + 1L
#        if (otherdim == 0) {
#            keep <- integer(0)
#        } else {
#            keep <- seq(istart, otherdim, by=index_params[[2]])
#        }
#        all.expected <- create_expected_dense(mat, row, iseq, keep)
#
#        test_that(pretty_name("dense index ", scenarios[i,]), {
#            if (oracle) {
#                extracted <- raticate.tests::oracular_dense_indexed(ptr, row, iseq, keep) 
#            } else {
#                extracted <- raticate.tests::myopic_dense_indexed(ptr, row, iseq, keep)
#            }
#            expect_identical(all.expected, extracted)
#        })
#
#        test_that(pretty_name("sparse index ", scenarios[i,]), {
#            if (oracle) {
#                FUN <- raticate.tests::oracular_sparse_indexed
#            } else {
#                FUN <- raticate.tests::myopic_sparse_indexed
#            }
#
#            extractor.b <- FUN(ptr, row, iseq, keep, TRUE, TRUE)
#            expect_identical(all.expected, fill_sparse(extractor.b, otherdim, keep))
#            extractor.i <- FUN(ptr, row, iseq, keep, FALSE, TRUE)
#            expect_identical(extractor.i, lapply(extractor.b, function(y) y$index))
#            extractor.v <- FUN(ptr, row, iseq, keep, TRUE, FALSE)
#            expect_identical(extractor.v, lapply(extractor.b, function(y) y$value))
#            extractor.n <- unlist_to_integer(FUN(ptr, row, iseq, keep, FALSE, FALSE))
#            expect_identical(extractor.n, lengths(extractor.v))
#
#            if (DelayedArray::is_sparse(mat)) {
#                prod <- nrow(mat) * ncol(mat)
#                if (prod > 0) {
#                    expect_true(sum(extractor.n) < prod)
#                }
#            }
#        })
#    }
#}
#
#reuse_test_suite <- function(mat, cache.fraction) {
#    cache.size <- get_cache_size(mat, cache.fraction)
#    ptr <- raticate.tests::parse(mat, cache.size, cache.size > 0)
#
#    scenarios <- expand.grid(
#        cache = cache.fraction,
#        row = c(TRUE, FALSE),
#        oracle = c(FALSE, TRUE),
#        step = c(1, 5),
#        mode = c("forward", "alternating"),
#        stringsAsFactors=FALSE
#    )
#
#    for (i in seq_len(nrow(scenarios))) {
#        cache <- scenarios[i,"cache"]
#        row <- scenarios[i,"row"]
#        oracle <- scenarios[i,"oracle"]
#        step <- scenarios[i,"step"]
#        mode <- scenarios[i,"mode"]
#
#        iterdim <- if (row) nrow(mat) else ncol(mat) 
#        otherdim <- if (row) ncol(mat) else nrow(mat)
#
#        # Creating a vector of predictions where we constantly double back to
#        # re-use previous elements.
#        iseq <- (function() {
#            predictions <- list()
#            i <- 0L
#            while (i < iterdim) {
#                current <- i + seq_len(step * 2)
#                current <- current[current <= iterdim]
#                if (mode == "alternating") {
#                    if (length(predictions) %% 2 == 1L) {
#                        current <- rev(current)
#                    }
#                }
#                predictions <- append(predictions, list(current))
#                i <- i + step
#            }
#            unlist_to_integer(predictions)
#        })()
#        all.expected <- create_expected_dense(mat, row, iseq, NULL)
#
#        test_that(pretty_name("dense full re-used ", scenarios[i,]), {
#            if (oracle) {
#                extracted <- raticate.tests::oracular_dense_full(ptr, row, iseq)
#            } else {
#                extracted <- raticate.tests::myopic_dense_full(ptr, row, iseq)
#            }
#            expect_identical(all.expected, extracted)
#        })
#
#        test_that(pretty_name("sparse full re-used ", scenarios[i,]), {
#            if (oracle) {
#                FUN <- raticate.tests::oracular_sparse_full
#            } else {
#                FUN <- raticate.tests::myopic_sparse_full
#            }
#            extractor.b <- FUN(ptr, row, iseq, TRUE, TRUE)
#            expect_identical(all.expected, fill_sparse(extractor.b, otherdim, NULL))
#        })
#    }
#}
#
#parallel_test_suite <- function(mat, cache.fraction) {
#    cache.size <- get_cache_size(mat, cache.fraction)
#    ptr <- raticate.tests::parse(mat, cache.size, cache.size > 0)
#
#    refr <- Matrix::rowSums(mat)
#    refc <- Matrix::colSums(mat)
#
#    test_that("parallel dense rowsums", {
#        expect_equal(refr, raticate.tests::myopic_dense_sums(ptr, TRUE, 1))
#        expect_equal(refr, raticate.tests::oracular_dense_sums(ptr, TRUE, 1))
#        expect_equal(refr, raticate.tests::myopic_dense_sums(ptr, TRUE, 3))
#        expect_equal(refr, raticate.tests::oracular_dense_sums(ptr, TRUE, 3))
#    })
#
#    test_that("parallel dense colsums", {
#        expect_equal(refc, raticate.tests::myopic_dense_sums(ptr, FALSE, 1))
#        expect_equal(refc, raticate.tests::oracular_dense_sums(ptr, FALSE, 1))
#        expect_equal(refc, raticate.tests::myopic_dense_sums(ptr, FALSE, 3))
#        expect_equal(refc, raticate.tests::oracular_dense_sums(ptr, FALSE, 3))
#    })
#
#    test_that("parallel sparse rowsums", {
#        expect_equal(refr, raticate.tests::myopic_sparse_sums(ptr, TRUE, 1))
#        expect_equal(refr, raticate.tests::oracular_sparse_sums(ptr, TRUE, 1))
#        expect_equal(refr, raticate.tests::myopic_sparse_sums(ptr, TRUE, 3))
#        expect_equal(refr, raticate.tests::oracular_sparse_sums(ptr, TRUE, 3))
#    })
#
#    test_that("parallel sparse colsums", {
#        expect_equal(refc, raticate.tests::myopic_sparse_sums(ptr, FALSE, 1))
#        expect_equal(refc, raticate.tests::oracular_sparse_sums(ptr, FALSE, 1))
#        expect_equal(refc, raticate.tests::myopic_sparse_sums(ptr, FALSE, 3))
#        expect_equal(refc, raticate.tests::oracular_sparse_sums(ptr, FALSE, 3))
#    })
#}

def big_test_suite(mat, cache_fractions):
    full_test_suite(mat, cache_fractions)
    #block_test_suite(mat, cache.fraction)
    #index_test_suite(mat, cache.fraction)
    #reuse_test_suite(mat, cache.fraction)
    #parallel_test_suite(mat, cache.fraction)
