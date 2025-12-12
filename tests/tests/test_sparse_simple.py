import numpy
import delayedarray
import tatami_python_test
import compare
import simulate


def test_Sparse2darray_basic():
    NR = 34
    NC = 82
    mat = simulate.simulate_sparse(NR, NC)

    wrapped = tatami_python_test.WrappedMatrix(mat)
    assert wrapped.nrow() == NR
    assert wrapped.ncol() == NC
    assert wrapped.is_sparse()
    assert not wrapped.prefer_rows()

    compare.big_test_suite(mat)


def test_Sparse2darray_types():
    NR = 74
    NC = 90

    mat = simulate.simulate_sparse(NR, NC, value_dtype = numpy.dtype("int8"))
    compare.quick_test_suite(mat)
    mat = simulate.simulate_sparse(NR, NC, value_dtype = numpy.dtype("uint8"))
    compare.quick_test_suite(mat)
    mat = simulate.simulate_sparse(NR, NC, value_dtype = numpy.dtype("int16"))
    compare.quick_test_suite(mat)
    mat = simulate.simulate_sparse(NR, NC, value_dtype = numpy.dtype("uint16"))
    compare.quick_test_suite(mat)
    mat = simulate.simulate_sparse(NR, NC, value_dtype = numpy.dtype("int32"))
    compare.quick_test_suite(mat)
    mat = simulate.simulate_sparse(NR, NC, value_dtype = numpy.dtype("uint32"))
    compare.quick_test_suite(mat)
    mat = simulate.simulate_sparse(NR, NC, value_dtype = numpy.dtype("int64"))
    compare.quick_test_suite(mat)
    mat = simulate.simulate_sparse(NR, NC, value_dtype = numpy.dtype("uint64"))
    compare.quick_test_suite(mat)
    mat = simulate.simulate_sparse(NR, NC, value_dtype = numpy.dtype("double"))
    compare.quick_test_suite(mat)
    mat = simulate.simulate_sparse(NR, NC, value_dtype = numpy.dtype("float"))
    compare.quick_test_suite(mat)

    mat = simulate.simulate_sparse(NR, NC, index_dtype = numpy.dtype("int8"))
    compare.quick_test_suite(mat)
    mat = simulate.simulate_sparse(NR, NC, index_dtype = numpy.dtype("uint8"))
    compare.quick_test_suite(mat)
    mat = simulate.simulate_sparse(NR, NC, index_dtype = numpy.dtype("int16"))
    compare.quick_test_suite(mat)
    mat = simulate.simulate_sparse(NR, NC, index_dtype = numpy.dtype("uint16"))
    compare.quick_test_suite(mat)
    mat = simulate.simulate_sparse(NR, NC, index_dtype = numpy.dtype("int32"))
    compare.quick_test_suite(mat)
    mat = simulate.simulate_sparse(NR, NC, index_dtype = numpy.dtype("uint32"))
    compare.quick_test_suite(mat)
    mat = simulate.simulate_sparse(NR, NC, index_dtype = numpy.dtype("int64"))
    compare.quick_test_suite(mat)
    mat = simulate.simulate_sparse(NR, NC, index_dtype = numpy.dtype("uint64"))
    compare.quick_test_suite(mat)


def test_Sparse2darray_partial_empty():
    NR = 104
    NC = 66
    mat = simulate.simulate_sparse(NR, NC, density = 0.3, empty = 0.4)

    wrapped = tatami_python_test.WrappedMatrix(mat)
    assert wrapped.nrow() == NR
    assert wrapped.ncol() == NC
    assert wrapped.is_sparse()

    compare.big_test_suite(mat)


def test_Sparse2darray_empty():
    mat = delayedarray.SparseNdarray((10, 10), None, dtype=numpy.dtype("double"), index_dtype=numpy.dtype("int32"))
    compare.quick_test_suite(mat)

    mat = delayedarray.SparseNdarray((10, 0), None, dtype=numpy.dtype("double"), index_dtype=numpy.dtype("int32"))
    compare.quick_test_suite(mat)

    mat = delayedarray.SparseNdarray((0, 10), None, dtype=numpy.dtype("double"), index_dtype=numpy.dtype("int32"))
    compare.quick_test_suite(mat)
