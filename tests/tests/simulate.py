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
