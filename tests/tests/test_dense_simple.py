import numpy
import delayedarray
import tatami_python_test
import helpers


def test_dense_numpy_row_major():
    NR = 34
    NC = 82
    mat = numpy.random.rand(NR, NC)
    assert mat.flags.c_contiguous

    wrapped = tatami_python_test.WrappedMatrix(mat)
    assert wrapped.nrow() == NR
    assert wrapped.ncol() == NC
    assert not wrapped.is_sparse()
    assert wrapped.prefer_rows()

    helpers.big_test_suite(mat, [0, 0.01, 0.1, 0.5])


def test_dense_numpy_col_major():
    NR = 54
    NC = 62
    mat = numpy.asfortranarray(numpy.random.rand(NR, NC))
    assert mat.flags.f_contiguous

    wrapped = tatami_python_test.WrappedMatrix(mat)
    assert wrapped.nrow() == NR
    assert wrapped.ncol() == NC
    assert not wrapped.is_sparse()
    assert not wrapped.prefer_rows()

    helpers.big_test_suite(mat, [0, 0.01, 0.1, 0.5])
