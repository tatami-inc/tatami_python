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


def quick_test_suite(mat):
    # Quick tests, mostly to verify that the functions work with other types.
    ptr = tatami_python_test.WrappedMatrix(mat, 0.2, True)

    iseq = create_predictions(mat.shape[0], 1, "forward")
    all_expected = create_expected_dense(mat, True, iseq, None)
    extracted = ptr.extract_dense(True, iseq, None, False)
    compare_list_of_vectors(extracted, all_expected)

    iseq = create_predictions(mat.shape[1], 1, "forward")
    all_expected = create_expected_dense(mat, False, iseq, None)
    extracted = ptr.extract_dense(False, iseq, None, True)
    compare_list_of_vectors(extracted, all_expected)


def full_test_suite(mat):
    scenarios = expand_grid({
        "cache": [0, 0.01, 0.1, 0.5],
        "row": [True, False],
        "oracle": [False, True],
        "mode": ["forward", "reverse", "random"],
        "step": [1, 5]
    })

    for scen in scenarios:
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

        print(pretty_name("dense full ", scen))
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
            prod = len(iseq) * otherdim
            if prod > 0:
                assert prod > sum(extracted_n)


def block_test_suite(mat):
    scenarios = expand_grid({
        "cache": [0, 0.01, 0.1, 0.5],
        "row": [True, False],
        "oracle": [False, True],
        "mode": ["forward", "reverse", "random"], 
        "step": [1, 5],
        "block": [(0, 0.3), (0.2, 0.66), (0.6, 0.37)]
    })

    for scen in scenarios:
        cache = scen["cache"]
        row = scen["row"]
        oracle = scen["oracle"]
        mode = scen[ "mode"]
        step = scen["step"]
        block = scen["block"]

        iterdim = mat.shape[1 - int(row)]
        otherdim = mat.shape[int(row)]
        iseq = create_predictions(iterdim, step, mode)
        cache_size = get_cache_size(mat, cache)
        ptr = tatami_python_test.WrappedMatrix(mat, cache_size, cache_size > 0)

        bstart = int(block[0] * otherdim)
        blen = int(block[1] * otherdim)
        block = (bstart, blen)
        block_keep = range(bstart, blen + bstart)

        print(pretty_name("dense block ", scen))
        all_expected = create_expected_dense(mat, row, iseq, block_keep)
        extracted = ptr.extract_dense(row, iseq, block, oracle)
        compare_list_of_vectors(extracted, all_expected)

        print(pretty_name("sparse block ", scen))
        extracted_sparse = ptr.extract_sparse(row, iseq, block, oracle, needs_value=True, needs_index=True)
        compare_list_of_vectors(fill_sparse(extracted_sparse, otherdim, block_keep), all_expected)

        extracted_index = ptr.extract_sparse(row, iseq, block, oracle, needs_value=False, needs_index=True)
        compare_list_of_vectors(extracted_index, [y["index"] for y in extracted_sparse])

        extracted_value = ptr.extract_sparse(row, iseq, block, oracle, needs_value=True, needs_index=False)
        compare_list_of_vectors(extracted_value, [y["value"] for y in extracted_sparse])

        extracted_n = ptr.extract_sparse(row, iseq, block, oracle, needs_value=False, needs_index=False)
        assert extracted_n == [len(y["value"]) for y in extracted_sparse]

        if ptr.is_sparse():
            prod = blen * len(iseq)
            if prod > 0:
                assert prod > sum(extracted_n)


def index_test_suite(mat):
    scenarios = expand_grid({
        "cache": [0, 0.01, 0.1, 0.5],
        "row": [True, False],
        "oracle": [False, True],
        "mode": ["forward", "reverse", "random"], 
        "step": [1, 5],
        "index": [(0, 3), (0.33, 4), (0.5, 5)],
    })

    for scen in scenarios:
        cache = scen["cache"]
        row = scen["row"]
        oracle = scen["oracle"]
        mode = scen[ "mode"]
        step = scen["step"]
        index_params = scen["index"]

        iterdim = mat.shape[1 - int(row)]
        otherdim = mat.shape[int(row)]
        iseq = create_predictions(iterdim, step, mode)
        cache_size = get_cache_size(mat, cache)
        ptr = tatami_python_test.WrappedMatrix(mat, cache_size, cache_size > 0)

        istart = int(index_params[0] * otherdim)
        indices = numpy.array(range(istart, otherdim, index_params[1]), dtype=numpy.dtype("int32"))

        print(pretty_name("dense indexed ", scen))
        all_expected = create_expected_dense(mat, row, iseq, indices)
        extracted = ptr.extract_dense(row, iseq, indices, oracle)
        compare_list_of_vectors(extracted, all_expected)

        print(pretty_name("sparse indexed ", scen))
        extracted_sparse = ptr.extract_sparse(row, iseq, indices, oracle, needs_value=True, needs_index=True)
        compare_list_of_vectors(fill_sparse(extracted_sparse, otherdim, indices), all_expected)

        extracted_index = ptr.extract_sparse(row, iseq, indices, oracle, needs_value=False, needs_index=True)
        compare_list_of_vectors(extracted_index, [y["index"] for y in extracted_sparse])

        extracted_value = ptr.extract_sparse(row, iseq, indices, oracle, needs_value=True, needs_index=False)
        compare_list_of_vectors(extracted_value, [y["value"] for y in extracted_sparse])

        extracted_n = ptr.extract_sparse(row, iseq, indices, oracle, needs_value=False, needs_index=False)
        assert extracted_n == [len(y["value"]) for y in extracted_sparse]

        if ptr.is_sparse():
            prod = len(indices) * len(iseq)
            if prod > 0:
                assert prod > sum(extracted_n)


def reuse_test_suite(mat):
    scenarios = expand_grid({
        "cache": [0, 0.01, 0.1, 0.5],
        "row": [True, False],
        "oracle": [False, True],
        "mode": ["forward", "alternating"], 
        "step": [1, 5]
    })

    for scen in scenarios:
        cache = scen["cache"]
        row = scen["row"]
        oracle = scen["oracle"]
        mode = scen["mode"]
        step = scen["step"]

        iterdim = mat.shape[1 - int(row)]
        otherdim = mat.shape[int(row)]

        # Creating a vector of predictions where we constantly double back to
        # re-use previous elements.
        iseq = []
        i = 0
        alternate = False
        while i < iterdim:
            current = range(i, min(iterdim, i + step * 2))
            if mode == "alternating":
                if alternate:
                    current = reversed(current)
                alternate = not alternate
            iseq += current
            i += step

        cache_size = get_cache_size(mat, cache)
        ptr = tatami_python_test.WrappedMatrix(mat, cache_size, cache_size > 0)

        print(pretty_name("dense full ", scen))
        all_expected = create_expected_dense(mat, row, iseq, None)
        extracted = ptr.extract_dense(row, iseq, None, oracle)
        compare_list_of_vectors(extracted, all_expected)

        print(pretty_name("sparse full ", scen))
        extracted_sparse = ptr.extract_sparse(row, iseq, None, oracle, needs_value=True, needs_index=True)
        compare_list_of_vectors(fill_sparse(extracted_sparse, otherdim, None), all_expected)


def parallel_test_suite(mat):
    shape = (range(mat.shape[0]), range(mat.shape[1]))
    extracted = delayedarray.extract_dense_array(mat, shape)
    refr = extracted.sum(axis=1)
    refc = extracted.sum(axis=0)

    for cache in [0, 0.01, 0.1, 0.5]: 
        cache_size = get_cache_size(mat, cache)
        ptr = tatami_python_test.WrappedMatrix(mat, cache_size, cache_size > 0)

        print("dense rowsums [cache=" + str(cache) + "]")
        assert numpy.allclose(refr, ptr.dense_sum(True, True, 1))
        assert numpy.allclose(refr, ptr.dense_sum(True, False, 1))
        assert numpy.allclose(refr, ptr.dense_sum(True, True, 3))
        assert numpy.allclose(refr, ptr.dense_sum(True, False, 3))

        print("dense colsums [cache=" + str(cache) + "]")
        assert numpy.allclose(refc, ptr.dense_sum(False, True, 1))
        assert numpy.allclose(refc, ptr.dense_sum(False, False, 1))
        assert numpy.allclose(refc, ptr.dense_sum(False, True, 3))
        assert numpy.allclose(refc, ptr.dense_sum(False, False, 3))

        print("sparse rowsums [cache=" + str(cache) + "]")
        assert numpy.allclose(refr, ptr.sparse_sum(True, True, 1))
        assert numpy.allclose(refr, ptr.sparse_sum(True, False, 1))
        assert numpy.allclose(refr, ptr.sparse_sum(True, True, 3))
        assert numpy.allclose(refr, ptr.sparse_sum(True, False, 3))

        print("sparse colsums [cache=" + str(cache) + "]")
        assert numpy.allclose(refc, ptr.sparse_sum(False, True, 1))
        assert numpy.allclose(refc, ptr.sparse_sum(False, False, 1))
        assert numpy.allclose(refc, ptr.sparse_sum(False, True, 3))
        assert numpy.allclose(refc, ptr.sparse_sum(False, False, 3))


def big_test_suite(mat):
    full_test_suite(mat)
    block_test_suite(mat)
    index_test_suite(mat)
    reuse_test_suite(mat)
    parallel_test_suite(mat)
