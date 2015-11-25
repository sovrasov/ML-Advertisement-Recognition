typedef double flpoint;

struct _Dataset
{
    int n_features;
    int n_instances;
    flpoint* X;
    int* y;
};

typedef struct _Dataset Dataset;

typedef char bool;
#define TRUE 1
#define FALSE 0

Dataset fcnn_reduce(Dataset ds, int n_neighbors);
Dataset ccis_reduce(Dataset ds, int n_neighbors);
Dataset cnn_reduce(Dataset ds, int n_neighbors);

Dataset alloc_dataset(int n_features, int n_instances);
void free_dataset(Dataset ds);

