from __future__ import division
import numpy as np
cimport numpy as np
cimport cpsmethods


cdef class PrototypeSelector:
    cdef cpsmethods.Dataset ds
    cdef cpsmethods.Dataset ds_reduced

    def __cinit__(self, np.ndarray[np.float_t, ndim=2] X,
            np.ndarray[np.int_t] y):
        self.ds = self._out_of_numpy_arrays(X, y)

    def __dealloc__(self):
        cpsmethods.free_dataset(self.ds)
        cpsmethods.free_dataset(self.ds_reduced)

    def fcnn_reduce(self, int n_neighbors):
        cdef cpsmethods.Dataset ds_reduced
        ds_reduced = cpsmethods.fcnn_reduce(self.ds, n_neighbors)
        return self._to_numpy_arrays(ds_reduced)

    def cnn_reduce(self, int n_neighbors):
        cdef cpsmethods.Dataset ds_reduced
        ds_reduced = cpsmethods.cnn_reduce(self.ds, n_neighbors)
        return self._to_numpy_arrays(ds_reduced)

    def ccis_reduce(self, int n_neighbors):
        cdef cpsmethods.Dataset ds_reduced
        ds_reduced = cpsmethods.ccis_reduce(self.ds)
        return self._to_numpy_arrays(ds_reduced)

    cdef cpsmethods.Dataset _out_of_numpy_arrays(self,
            np.ndarray[np.float_t, ndim=2] X, np.ndarray[np.int_t] y):
        cdef int n_features = X.shape[1]
        cdef int n_instances = X.shape[0]
        cdef cpsmethods.Dataset temp_ds = cpsmethods.alloc_dataset(n_features,
                n_instances)
        cdef int i, j
        for i in range(n_instances):
            for j in range(n_features):
                temp_ds.X[i * n_features + j] = X[i, j]
            temp_ds.y[i] = y[i]

        return temp_ds
        
    cdef _to_numpy_arrays(self, cpsmethods.Dataset ds):
        cdef int n_features = ds.n_features
        cdef int n_instances = ds.n_instances
        cdef np.ndarray[np.float_t, ndim=2] X = np.empty(
                (n_instances, n_features), dtype=np.float)
        cdef np.ndarray[np.int_t] y = np.empty(n_instances, dtype=np.int)
        cdef int i, j
        for i in range(n_instances):
            for j in range(n_features):
                X[i, j] = ds.X[i * n_features + j]
            y[i] = ds.y[i]

        return X, y

