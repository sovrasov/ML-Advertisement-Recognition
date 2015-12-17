#include "stdlib.h"

#include "psutils.h"

int compare_ints(const void* a, const void* b)
{
    int ai = *(int*)a;
    int bi = *(int*)b;

    if (ai < bi) return -1;
    if (ai > bi) return 1;
    return 0;
}

flpoint squared_dist(int N, flpoint* a, flpoint* b)
{
    int i;
    flpoint result = 0;
    for (i = 0; i < N; i++)
    {
        flpoint d = a[i] - b[i];
        result += d * d;
    }

    return result;
}

void fill_int_array(int* array, int N, int fill)
{
    int i;
    for (i = 0; i < N; i++)
        array[i] = fill;
}

void fill_flpoint_array(flpoint* array, int N, flpoint fill)
{
    int i;
    for (i = 0; i < N; i++)
        array[i] = fill;
}

void count_classes(const Dataset ds, int* n_classes, int** class_labels)
{
    int i, j;
    int* temp_class_labels = malloc(sizeof(int) * ds.n_instances);
    *n_classes = 0;
    for (i = 0; i < ds.n_instances; i++)
    {
        bool class_found = FALSE;
        for (j = 0; j < *n_classes; j++)
            if (ds.y[i] == temp_class_labels[j])
            {
                class_found = TRUE;
                break;
            }
        if (!class_found) temp_class_labels[(*n_classes)++] = ds.y[i];
    }

    *class_labels = malloc(sizeof(int) * *n_classes);
    for (i = 0; i < *n_classes; i++)
        (*class_labels)[i] = temp_class_labels[i];
    free(temp_class_labels);
}

int find_instances_class(const Dataset ds, int instance, int n_classes,
        const int* class_labels)
{
    int i;
    int instances_class_label = ds.y[instance];
    for (i = 0; i < n_classes; i++)
    {
        if (instances_class_label == class_labels[i])
            return i;
    }
    
    return -1;
}

Dataset alloc_dataset(int n_features, int n_instances)
{
    flpoint* X = malloc(sizeof(flpoint) * n_features * n_instances);
    int* y = malloc(sizeof(int) * n_instances);
    Dataset ds;
   	ds.n_features = n_features;
	ds.n_instances = n_instances;
    ds.X = X;
	ds.y = y;
    return ds;
}

void free_dataset(Dataset ds)
{
    if (ds.X != NULL)
        free(ds.X);
    if (ds.y != NULL)
        free(ds.y);
}

