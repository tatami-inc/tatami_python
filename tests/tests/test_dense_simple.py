import numpy
import delayedarray
import tatami_python_test
import compare


def test_numpy_row_major():
    NR = 34
    NC = 82
    mat = numpy.random.rand(NR, NC)
    assert mat.flags.c_contiguous

    # Check that it's still C layout after extraction,
    # to make sure that the corresponding C++ code is actually tested.
    extracted = delayedarray.extract_dense_array(mat, ([0, 1], [0, 1]))
    assert extracted.flags.c_contiguous

    wrapped = tatami_python_test.WrappedMatrix(mat)
    assert wrapped.nrow() == NR
    assert wrapped.ncol() == NC
    assert not wrapped.is_sparse()
    assert wrapped.prefer_rows()

    compare.big_test_suite(mat, [0, 0.01, 0.1, 0.5])


# Mock class for checking column major extraction.
class farray:
    def __init__(self, thing):
        self._thing = numpy.asfortranarray(thing)

    @property
    def shape(self):
        return self._thing.shape

    @property
    def dtype(self):
        return self._thing.dtype

    def __getitem__(self, *args, **kwargs):
        return self._thing.__getitem__(*args, **kwargs)


@delayedarray.extract_dense_array.register
def extract_array_from_farray(x: farray, indices):
    return numpy.asfortranarray(delayedarray.extract_dense_array(x._thing, indices))


@delayedarray.chunk_grid.register
def chunk_grid_from_farray(x: farray):
    return delayedarray.SimpleGrid(([x.shape[0]], delayedarray.RegularTicks(1, x.shape[1])), cost_factor=1)


def test_numpy_col_major():
    NR = 54
    NC = 62
    mat = farray(numpy.random.rand(NR, NC))

    # Check that it's still Fortran layout after extraction,
    # to make sure that the corresponding C++ code is actually tested.
    extracted = delayedarray.extract_dense_array(mat, ([0, 1], [0, 1]))
    assert extracted.flags.f_contiguous

    wrapped = tatami_python_test.WrappedMatrix(mat)
    assert wrapped.nrow() == NR
    assert wrapped.ncol() == NC
    assert not wrapped.is_sparse()
    assert not wrapped.prefer_rows()

    compare.big_test_suite(mat, [0, 0.01, 0.1, 0.5])


def test_numpy_types():
    NR = 100
    NC = 80
    mat = numpy.random.rand(NR, NC) * 100

    compare.quick_test_suite(mat.astype(numpy.dtype("int8")))
    compare.quick_test_suite(mat.astype(numpy.dtype("uint8")))
    compare.quick_test_suite(mat.astype(numpy.dtype("int16")))
    compare.quick_test_suite(mat.astype(numpy.dtype("uint16")))
    compare.quick_test_suite(mat.astype(numpy.dtype("int32")))
    compare.quick_test_suite(mat.astype(numpy.dtype("uint32")))
    compare.quick_test_suite(mat.astype(numpy.dtype("float")))
    compare.quick_test_suite(mat.astype(numpy.dtype("double")))


def test_numpy_empty():
    NR = 0 
    NC = 10
    mat = numpy.random.rand(NR, NC)
    compare.quick_test_suite(mat)

    NR = 10
    NC = 0
    mat = numpy.random.rand(NR, NC)
    compare.quick_test_suite(mat)
