import random
import numpy
import delayedarray


def simulate_sparse(nrow, ncol, density = 0.1, empty = None, value_dtype = numpy.dtype("double"), index_dtype = numpy.dtype("int32")):
    svt = []
    for c in range(ncol):
        if empty is not None and random.random() < empty:
            svt.append(None)
            continue

        all_i = []
        all_v = []
        for r in range(nrow):
            if random.random() >= density:
                continue
            all_i.append(r)
            all_v.append(random.random() * 100)

        all_i = numpy.array(all_i, dtype=index_dtype)
        all_v = numpy.array(all_v, dtype=index_dtype)
        svt.append((all_i, all_v))

    return delayedarray.SparseNdarray((nrow, ncol), contents=svt, dtype=value_dtype, index_dtype=index_dtype, is_masked=False, check=False) 


class RegularChunkedArray:
    def __init__(self, thing, spacing):
        self._thing = thing
        self._spacing = spacing

    @property
    def shape(self):
        return self._thing.shape

    @property
    def dtype(self):
        return self._thing.dtype

    def __getitem__(self, *args, **kwargs):
        return self._thing.__getitem__(*args, **kwargs)


@delayedarray.is_sparse.register
def is_sparsec_RegularChunkedArray(x: RegularChunkedArray):
    return delayedarray.is_sparse(x._thing)


@delayedarray.extract_dense_array.register
def extract_array_from_RegularChunkedArray(x: RegularChunkedArray, indices):
    return delayedarray.extract_dense_array(x._thing, indices)


@delayedarray.extract_sparse_array.register
def extract_array_from_RegularChunkedArray(x: RegularChunkedArray, indices):
    return delayedarray.extract_sparse_array(x._thing, indices)


@delayedarray.chunk_grid.register
def chunk_grid_from_RegularChunkedArray(x: RegularChunkedArray):
    row_ticks = delayedarray.RegularTicks(x._spacing[0], x.shape[0])
    col_ticks = delayedarray.RegularTicks(x._spacing[1], x.shape[1])
    return delayedarray.SimpleGrid((row_ticks, col_ticks), cost_factor=1)


def create_irregular_ticks(n, p):
    output = []
    for i in range(1, n):
        if random.random() < p:
            output.append(i)
    output.append(n)
    return output


class IrregularChunkedArray:
    def __init__(self, thing, ticks):
        self._thing = thing
        self._ticks = ticks 

    @property
    def shape(self):
        return self._thing.shape

    @property
    def dtype(self):
        return self._thing.dtype

    def __getitem__(self, *args, **kwargs):
        return self._thing.__getitem__(*args, **kwargs)


@delayedarray.is_sparse.register
def is_sparsec_IrregularChunkedArray(x: IrregularChunkedArray):
    return delayedarray.is_sparse(x._thing)


@delayedarray.extract_dense_array.register
def extract_array_from_IrregularChunkedArray(x: IrregularChunkedArray, indices):
    return delayedarray.extract_dense_array(x._thing, indices)


@delayedarray.extract_sparse_array.register
def extract_array_from_IrregularChunkedArray(x: IrregularChunkedArray, indices):
    return delayedarray.extract_sparse_array(x._thing, indices)


@delayedarray.chunk_grid.register
def chunk_grid_from_IrregularChunkedArray(x: IrregularChunkedArray):
    return delayedarray.SimpleGrid(x._ticks, cost_factor=1)
