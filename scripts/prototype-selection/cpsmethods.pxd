cdef extern from "psmethods.h":
    ctypedef double flpoint

    ctypedef struct Dataset:
        int n_features
        int n_instances
        flpoint* X
        int* y
        
    Dataset fcnn_reduce(Dataset ds, int n_neighbors);
    Dataset cnn_reduce(Dataset ds, int n_neighbors);

    Dataset alloc_dataset(int n_features, int n_instances);
    void free_dataset(Dataset ds);

