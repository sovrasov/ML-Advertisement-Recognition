from __future__ import division
import numpy as np
cimport numpy as np
cimport cfcnn


cdef class PrototypeSelector:
    cdef cfcnn.Dataset ds
    cdef cfcnn.Dataset ds_reduced

    def __cinit__(self, np.ndarray[np.float_t, ndim=2] X,
            np.ndarray[np.int_t] y):
        self.ds = self._out_of_numpy_arrays(X, y)

    def __dealloc__(self):
        cfcnn.free_dataset(self.ds)
        cfcnn.free_dataset(self.ds_reduced)

    def fcnn_reduce(self, int n_neighbors):
        cdef cfcnn.Dataset ds_reduced
        ds_reduced = cfcnn.fcnn_reduce(self.ds, n_neighbors)
        self.reduction_ratio = ds_reduced.n_instances / self.ds.n_instances
        return self._to_numpy_arrays(ds_reduced)

    cdef cfcnn.Dataset _out_of_numpy_arrays(self,
            np.ndarray[np.float_t, ndim=2] X, np.ndarray[np.int_t] y):
        cdef int n_features = X.shape[1]
        cdef int n_instances = X.shape[0]
        cdef cfcnn.Dataset temp_ds = cfcnn.alloc_dataset(n_features,
                n_instances)
        cdef int i, j
        for i in range(n_instances):
            for j in range(n_features):
                temp_ds.X[i * n_features + j] = X[i, j]
            temp_ds.y[i] = y[i]

        return temp_ds
        
    cdef _to_numpy_arrays(self, cfcnn.Dataset ds):
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

