#include "string.h"
#include "stdlib.h"

#include "psutils.h"

void find_classes_centroids_in_data(const Dataset ds, int n_classes,
        int* class_labels, int* indices)
{
    int i, j;
    flpoint* centroids = calloc(n_classes * ds.n_features, sizeof(flpoint));
    int* class_instance_count = calloc(n_classes, sizeof(int));
    flpoint* min_squared_dists = NULL;
    int* closest_to_centroids = NULL;

    // add each instance to the sum of instances of the corresponding
    // class
    for (i = 0; i < ds.n_instances; i++)
    {
        int current_class = find_instances_class(ds, i, n_classes, class_labels);

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
        min_squared_dists[i] = -1;

    for (i = 0; i < ds.n_instances; i++)
    {
        int current_class = find_instances_class(ds, i, n_classes, class_labels);
        flpoint current_squared_dist;

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
        non_S_size = 0;
        for (i = 0; i < ds.n_instances; i++)
            if (S_index == S_size || i < S[S_index])
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
                    int* nearest_for_i = nearest + non_S[i] * n_neighbors;
                    if (nearest_for_i[k] < 0)
                    {
                        nearest_for_i[k] =
                            delta_S[j];
                        break;
                    }
                    if (squared_dist(ds.n_features,
                                ds.X + ds.n_features *
                                nearest_for_i[k],
                                ds.X + ds.n_features * non_S[i]) >
                            squared_dist(ds.n_features,
                                ds.X + ds.n_features * non_S[i],
                                ds.X + ds.n_features * delta_S[j]))
                    {
                        // move all farther neighbors to the right
                        for (l = n_neighbors - 1; l >= k + 1; l--)
                            nearest_for_i[l] = nearest_for_i[l - 1];
                        nearest_for_i[k] = delta_S[j];
                        break;
                    }
                }
            }

            memset(votes, 0, sizeof(int) * n_classes);
            // collect votes for their classes from these neighbors
            for (j = 0; j < n_neighbors; j++)
            {
                int current_neighbor = nearest[non_S[i] * n_neighbors + j];
                if (current_neighbor >= 0)
                {
                    int current_class = find_instances_class(ds,
                            current_neighbor, n_classes, class_labels);
                    votes[current_class]++;
                }
                else
                    break;
            }

            // find majority class of these neighbors
            neighbor_majority_class = class_labels[0];
            neighbor_majority_class_count = votes[0];
            for (j = 1; j < n_classes; j++)
                if (votes[j] > neighbor_majority_class_count)
                {
                    neighbor_majority_class_count = votes[j];
                    neighbor_majority_class = class_labels[j];
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
                    if (current_neighbor >= 0)
                    {
                        if (rep[current_neighbor] < 0 ||
                             squared_dist(ds.n_features,
                                 ds.X + ds.n_features * current_neighbor,
                                 ds.X + ds.n_features * non_S[i]) <
                             squared_dist(ds.n_features,
                                 ds.X + ds.n_features * current_neighbor,
                                 ds.X + ds.n_features * rep[current_neighbor])
                            )
                            rep[current_neighbor] = non_S[i];
                    }
                    else break;
                }
            }
        }

        // refill delta_S again
        delta_S_size = 0;
        for (i = 0; i < S_size; i++)
        {
            bool instance_in_delta_S = FALSE;
            for (j = 0; j < delta_S_size; j++)
                if (rep[S[i]] == delta_S[j])
                {
                    instance_in_delta_S = TRUE;
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

