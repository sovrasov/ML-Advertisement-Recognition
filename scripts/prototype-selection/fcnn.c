#include "stdlib.h"
#include "string.h"
#include "limits.h"
#include "stdio.h"

#include "fcnn.h"

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
    flpoint result = 0.;
    for (i = 0; i < N; i++)
    {
        int d = a[i] - b[i];
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

void count_classes(const Dataset ds, int* n_classes, int** class_labels)
{
    int i, j;
    int* temp_class_labels = malloc(sizeof(int) * ds.n_instances);
    *n_classes = 0;
    for (i = 0; i < ds.n_instances; i++)
    {
        char class_found = 0;
        for (j = 0; j < *n_classes; j++)
            if (ds.y[i] == temp_class_labels[j])
            {
                class_found = 1;
                break;
            }
        if (!class_found) temp_class_labels[(*n_classes)++] = ds.y[i];
    }

    *class_labels = malloc(sizeof(int) * *n_classes);
    for (i = 0; i < *n_classes; i++)
        (*class_labels)[i] = temp_class_labels[i];
    free(temp_class_labels);
}
        
void find_classes_centroids_in_data(const Dataset ds, int n_classes,
        int* class_labels, int* indices)
{
    int i, j;
    flpoint* centroids = malloc(sizeof(flpoint) * n_classes * ds.n_features);
    int* class_instance_count = malloc(sizeof(int) * n_classes);
    flpoint* min_squared_dists = NULL;
    int* closest_to_centroids = NULL;

    memset(centroids, 0, sizeof(flpoint) * n_classes * ds.n_features);
    memset(class_instance_count, 0, sizeof(int) * n_classes);

    // add each instance to the sum of instances of the corresponding
    // class
    for (i = 0; i < ds.n_instances; i++)
    {
        int current_class = -1;
        for (j = 0; j < n_classes; j++)
            if (ds.y[i] == class_labels[j])
            {
                current_class = j;
                break;
            }
        
        for (j = 0; j < ds.n_features; j++)
            centroids[current_class * ds.n_features + j] +=
                ds.X[i * ds.n_features + j];
        class_instance_count[current_class] += 1;
    }

    // divide all sums by the number of instances in the respective class
    for (i = 0; i < n_classes; i++)
    {
        flpoint norm = 1. / class_instance_count[i];
        for (j = 0; j < ds.n_features; j++)
            centroids[i * ds.n_features + j] *= norm;
    }

    // find instances in the dataset closest to centroids computed above
    min_squared_dists = malloc(sizeof(flpoint) * n_classes);
    closest_to_centroids = malloc(sizeof(int) * n_classes);
    fill_int_array(closest_to_centroids, n_classes, -1);
    for (i = 0; i < n_classes; i++)
        min_squared_dists[i] = -1.;
    for (i = 0; i < ds.n_instances; i++)
    {
        int current_class = -1;
        flpoint current_squared_dist;
        for (j = 0; j < n_classes; j++)
            if (ds.y[i] == class_labels[j])
            {
                current_class = j;
                break;
            }

        current_squared_dist = squared_dist(ds.n_features,
                centroids + current_class * ds.n_features,
                ds.X + i * ds.n_features);
        if (min_squared_dists[current_class] < 0 ||
                current_squared_dist < min_squared_dists[current_class])
        {
            min_squared_dists[current_class] = current_squared_dist;
            closest_to_centroids[current_class] = i;
        }
    }

    for (i = 0; i < n_classes; i++)
        indices[i] = closest_to_centroids[i];

    free(centroids);
    free(class_instance_count);
    free(min_squared_dists);
    free(closest_to_centroids);
}

Dataset fcnn_reduce(Dataset ds, int n_neighbors)
{
    int i, j, k, l;
    int n_classes;
    int* class_labels = NULL;
    int* S = malloc(sizeof(int) * ds.n_instances);
    int* delta_S = malloc(sizeof(int) * ds.n_instances);
    int* non_S = malloc(sizeof(int) * ds.n_instances);
    int S_size = 0;
    int delta_S_size = 0;
    int non_S_size = 0;
    int S_index;
    int* nearest = malloc(sizeof(int) * ds.n_instances * n_neighbors);
    int* rep = NULL;
    int* votes = NULL;
    int neighbor_majority_class;
    int neighbor_majority_class_count;
    Dataset ds_reduced;

    count_classes(ds, &n_classes, &class_labels);

    fill_int_array(nearest, ds.n_instances * n_neighbors, -1);
    fill_int_array(S, ds.n_instances, INT_MAX);

    delta_S_size = n_classes;
    find_classes_centroids_in_data(ds, n_classes, class_labels, delta_S);

    rep = malloc(sizeof(int) * ds.n_instances);
    votes = malloc(sizeof(int) * n_classes);
    // main loop
    while (delta_S_size > 0)
    {
        // merge delta_S into S
        for (i = 0; i < delta_S_size; i++)
        {
            S[S_size + i] = delta_S[i];
        }
        S_size += delta_S_size;
        qsort(S, S_size, sizeof(int), compare_ints);

        fill_int_array(rep, ds.n_instances, -1);

        // find instances which are not in S
        S_index = 0;
        for (i = 0; i < ds.n_instances; i++)
            if (i < S[S_index])
                non_S[non_S_size++] = i;
            else
                S_index++;

        for (i = 0; i < non_S_size; i++)
        {
            // find n_neighbors nearest neighbors for X[non_S[i]]
            // in delta_S
            for (j = 0; j < delta_S_size; j++)
            {
                for (k = 0; k < n_neighbors; k++)
                {
                    if (nearest[non_S[i] * n_neighbors + k] < 0)
                    {
                        nearest[non_S[i] * n_neighbors + k] =
                            delta_S[j];
                        break;
                    }
                    if (squared_dist(ds.n_features,
                                ds.X + ds.n_features *
                                nearest[non_S[i] * n_neighbors + k],
                                ds.X + ds.n_features * non_S[i]) >
                            squared_dist(ds.n_features,
                                ds.X + ds.n_features * non_S[i],
                                ds.X + ds.n_features * delta_S[j]))
                    {
                        for (l = n_neighbors - 1; l >= k + 1; l--)
                            nearest[non_S[i] * n_neighbors + l] =
                                nearest[non_S[i] * n_neighbors + l - 1];
                        nearest[non_S[i] * n_neighbors + k] = delta_S[j];
                    }
                }
            }

            fill_int_array(votes, n_classes, 0);
            // collect votes for their classes from these neighbors
            for (j = 0; j < n_neighbors; j++)
            {
                int current_neighbor = nearest[non_S[i] * n_neighbors + j];
                if (current_neighbor >= 0)
                {
                    int current_class = -1;
                    for (k = 0; k < n_classes; k++)
                        if (class_labels[k] == ds.y[current_neighbor])
                        {
                            current_class = k;
                            break;
                        }
                    votes[current_class]++;
                }
            }

            // find majority class of these neighbors
            neighbor_majority_class = 0;
            neighbor_majority_class_count = votes[0];
            for (j = 1; j < n_classes; j++)
                if (votes[j] > neighbor_majority_class_count)
                {
                    neighbor_majority_class_count = votes[j];
                    neighbor_majority_class = j;
                }

            // if majority class is incorrect (i.e. non_S[i] would
            // be misclassified by kNN-classifier trained on delta_S)
            // update representative instance for each neighbor
            if (ds.y[non_S[i]] != neighbor_majority_class)
            {
                for (j = 0; j < n_neighbors; j++)
                {
                    int current_neighbor =
                        nearest[non_S[i] * n_neighbors + j];
                    if (current_neighbor >= 0 &&
                            (rep[current_neighbor] < 0 ||
                             squared_dist(ds.n_features,
                                 ds.X + ds.n_features * current_neighbor,
                                 ds.X + ds.n_features * non_S[i]) <
                             squared_dist(ds.n_features,
                                 ds.X + ds.n_features * current_neighbor,
                                 ds.X + ds.n_features * rep[current_neighbor])
                            )
                        )
                        rep[current_neighbor] = non_S[i];
                }
            }
        }

        // refill delta_S again
        delta_S_size = 0;
        for (i = 0; i < S_size; i++)
        {
            char instance_in_delta_S = 0;
            for (j = 0; j < delta_S_size; j++)
                if (rep[S[i]] == delta_S[j])
                {
                    instance_in_delta_S = 1;
                    break;
                }
            if (rep[S[i]] >= 0 && !instance_in_delta_S)
                delta_S[delta_S_size++] = rep[S[i]];
        }
    }

    // form a new dataset with only selected instances
    ds_reduced = alloc_dataset(ds.n_features, S_size);
    for (i = 0; i < S_size; i++)
    {
        memcpy(ds_reduced.X + ds.n_features * i,
                ds.X + ds.n_features * S[i], sizeof(flpoint) * ds.n_features);
        ds_reduced.y[i] = ds.y[S[i]];
    }

    free(class_labels);
    free(S);
    free(delta_S);
    free(non_S);
    free(nearest);
    free(rep);
    free(votes);

    return ds_reduced;
}

Dataset alloc_dataset(int n_features, int n_instances)
{
    flpoint* X = malloc(sizeof(flpoint) * n_features * n_instances);
    int* y = malloc(sizeof(int) * n_instances);
    Dataset ds = { .n_features = n_features, .n_instances = n_instances,
        .X = X, .y = y };
    return ds;
}

void free_dataset(Dataset ds)
{
    if (ds.X != NULL)
        free(ds.X);
    if (ds.y != NULL)
        free(ds.y);
}
