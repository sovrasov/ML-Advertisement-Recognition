typedef double flpoint;

struct _Dataset
{
    int n_features;
    int n_instances;
    flpoint* X;
    int* y;
};

typedef struct _Dataset Dataset;

Dataset fcnn_reduce(Dataset ds, int n_neighbors);

Dataset alloc_dataset(int n_features, int n_instances);
void free_dataset(Dataset ds);

