import numpy
import delayedarray
from . import lib_tatami_python_test as lib

__author__ = "ltla"
__copyright__ = "ltla"
__license__ = "MIT"


class WrappedMatrix:
    def __init__(self, obj):
        self._ptr = lib.parse_test(obj)


    def __del__(self):
        lib.free_test(self._ptr)


    def nrow(self): 
        return lib.nrow_test(self._ptr);


    def ncol(self):
        return lib.ncol_test(self._ptr);


    def prefer_rows(self):
        return lib.prefer_rows_test(self._ptr);


    def is_sparse(self):
        return lib.is_sparse_test(self._ptr);


    def extract_dense(self, row, indices, subset, oracle = False):
        if subset is None:
            if oracle:
                return lib.oracular_dense_full(self._ptr, row, indices)
            else:
                return lib.myopic_dense_full(self._ptr, row, indices)
        elif isinstance(subset, tuple):
            if oracle:
                return lib.oracular_dense_block(self._ptr, row, indices, subset[0], subset[1])
            else:
                return lib.myopic_dense_block(self._ptr, row, indices, subset[0], subset[1])
        else:
            if oracle:
                return lib.oracular_dense_indexed(self._ptr, row, indices, numpy.array(subset[0], numpy.dtype("int32")))
            else:
                return lib.myopic_dense_indexed(self._ptr, row, indices, numpy.array(subset[1], numpy.dtype("int32")))


    def extract_sparse(self, row, indices, subset, oracle = False, needs_value = True, needs_index = True):
        if subset is None:
            if oracle:
                return lib.oracular_sparse_full(self._ptr, row, indices, needs_value, needs_index)
            else:
                return lib.myopic_sparse_full(self._ptr, row, indices, needs_value, needs_index)
        elif isinstance(subset, tuple):
            if oracle:
                return lib.oracular_sparse_block(self._ptr, row, indices, subset[0], subset[1], needs_value, needs_index)
            else:
                return lib.myopic_sparse_block(self._ptr, row, indices, subset[0], subset[1], needs_value, needs_index)
        else:
            if oracle:
                return lib.oracular_sparse_indexed(self._ptr, row, indices, numpy.array(subset[0], numpy.dtype("int32")), needs_value, needs_index)
            else:
                return lib.myopic_sparse_indexed(self._ptr, row, indices, numpy.array(subset[1], numpy.dtype("int32")), needs_value, needs_index)
