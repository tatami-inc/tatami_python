import numpy
import delayedarray
import tatami_python_test
import compare
import simulate


def test_sparse_chunked_regular_rows():
    NR = 64
    NC = 102
    mat = simulate.RegularChunkedArray(simulate.simulate_sparse(NR, NC), (10, 10))

    wrapped = tatami_python_test.WrappedMatrix(mat)
    assert wrapped.nrow() == NR
    assert wrapped.ncol() == NC
    assert wrapped.is_sparse()
    assert not wrapped.prefer_rows() # more chunks required to load a row than to load a column.

    compare.big_test_suite(mat)


def test_sparse_chunked_regular_columns():
    NR = 124
    NC = 52
    mat = simulate.RegularChunkedArray(simulate.simulate_sparse(NR, NC), (10, 10))

    wrapped = tatami_python_test.WrappedMatrix(mat)
    assert wrapped.nrow() == NR
    assert wrapped.ncol() == NC
    assert wrapped.is_sparse()
    assert wrapped.prefer_rows() # more chunks required to load a column than to load a row.

    compare.big_test_suite(mat)


def test_sparse_chunked_irregular():
    NR = 97
    NC = 78
    row_ticks = simulate.create_irregular_ticks(NR, 0.1)
    col_ticks = simulate.create_irregular_ticks(NC, 0.15)
    mat = simulate.IrregularChunkedArray(simulate.simulate_sparse(NR, NC), (row_ticks, col_ticks))

    wrapped = tatami_python_test.WrappedMatrix(mat)
    assert wrapped.nrow() == NR
    assert wrapped.ncol() == NC
    assert wrapped.is_sparse()

    compare.big_test_suite(mat)
