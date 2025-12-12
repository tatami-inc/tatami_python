import numpy
import delayedarray
import tatami_python_test
import compare
import simulate


def test_dense_chunked_regular_rows(subtests):
    NR = 54
    NC = 92
    mat = simulate.RegularChunkedArray(numpy.random.rand(NR, NC), (10, 10))

    wrapped = tatami_python_test.WrappedMatrix(mat)
    assert wrapped.nrow() == NR
    assert wrapped.ncol() == NC
    assert not wrapped.is_sparse()
    assert not wrapped.prefer_rows() # more chunks required to load a row than to load a column.

    compare.big_test_suite(subtests, mat)


def test_dense_chunked_regular_columns(subtests):
    NR = 154
    NC = 32
    mat = simulate.RegularChunkedArray(numpy.random.rand(NR, NC), (10, 10))

    wrapped = tatami_python_test.WrappedMatrix(mat)
    assert wrapped.nrow() == NR
    assert wrapped.ncol() == NC
    assert not wrapped.is_sparse()
    assert wrapped.prefer_rows() # more chunks required to load a column than to load a row.

    compare.big_test_suite(subtests, mat)


def test_dense_chunked_irregular(subtests):
    NR = 77
    NC = 88
    row_ticks = simulate.create_irregular_ticks(NR, 0.2)
    col_ticks = simulate.create_irregular_ticks(NC, 0.1)
    mat = simulate.IrregularChunkedArray(numpy.random.rand(NR, NC), (row_ticks, col_ticks))

    wrapped = tatami_python_test.WrappedMatrix(mat)
    assert wrapped.nrow() == NR
    assert wrapped.ncol() == NC
    assert not wrapped.is_sparse()

    compare.big_test_suite(subtests, mat)
