#include "string.h"
#include "stdlib.h"
#include "time.h"

#include "psutils.h"

void swap_ints(int* a, int* b)
{
    *a = *a ^ *b ^ (*b = *a);
}

void shuffle_ints(int N, int* array)
{
    int i;
    for (i = N - 1; i >= 0; i--)
    {
        int j = rand() % (i + 1);
        swap_ints(array + i, array + j);
    }
}

Dataset cnn_reduce(Dataset ds, int n_neighbors)
{
    int i, j, k, l;
    int n_classes;
    int* class_labels = NULL;
    int* S = malloc(sizeof(int) * ds.n_instances);
    int* S_copy = malloc(sizeof(int) * ds.n_instances);
    int* non_S = malloc(sizeof(int) * ds.n_instances);
    int* last_train_S_size = calloc(ds.n_instances, sizeof(int));
    int S_size = 0;
    int non_S_size = 0;
    int S_index;
    int* nearest = malloc(sizeof(int) * ds.n_instances * n_neighbors);
    int* votes = NULL;
    int neighbor_majority_class;
    int neighbor_majority_class_count;
    bool whole_non_S_classified_correctly = FALSE;
    Dataset ds_reduced;

    fill_int_array(nearest, ds.n_instances * n_neighbors, -1);

    count_classes(ds, &n_classes, &class_labels);
    votes = malloc(sizeof(int) * n_classes);
    
    // Add one random instance from each class to S
    srand(time(NULL));
    for (i = 0; i < n_classes; i++)
        while (1)
        {
            int j = rand() % ds.n_instances;
            if (ds.y[j] == class_labels[i])
            {
                S[S_size++] = j;
                break;
            }
        }

    while (!whole_non_S_classified_correctly)
    {
        whole_non_S_classified_correctly = TRUE;
        // copy S to auxiliary array and sort it
        memcpy(S_copy, S, sizeof(int) * S_size);
        qsort(S_copy, S_size, sizeof(int), compare_ints);

        // Find all instances not in S
        S_index = 0;
        non_S_size = 0;
        for (i = 0; i < ds.n_instances; i++)
            if (S_index == S_size || i < S_copy[S_index])
                non_S[non_S_size++] = i;
            else
                S_index++;

        shuffle_ints(non_S_size, non_S);

        for (i = 0; i < non_S_size; i++)
        {
            // update nearest neighbors for non_S[i]
            for (j = last_train_S_size[non_S[i]]; j < S_size; j++)
            {
                for (k = 0; k < n_neighbors; k++)
                {
                    int* nearest_for_i = nearest + non_S[i] * n_neighbors;
                    if (nearest_for_i[k] < 0)
                    {
                        nearest_for_i[k] = j;
                        break;
                    }
                    if (squared_dist(ds.n_features,
                                ds.X + ds.n_features * nearest_for_i[k],
                                ds.X + ds.n_features * non_S[i]) >
                            squared_dist(ds.n_features,
                                ds.X + ds.n_features * non_S[i],
                                ds.X + ds.n_features * j))
                    {
                        for (l = n_neighbors - 1; l >= k + 1; l--)
                            nearest_for_i[l] = nearest_for_i[l - 1];
                        nearest_for_i[k] = j;
                        break;
                    }
                }
            }

            // count votes for non_S[i]
            memset(votes, 0, n_classes * sizeof(int));
            for (j = 0; j < n_neighbors; j++)
            {
                int current_neighbor = nearest[non_S[i] * n_neighbors + j];
                if (current_neighbor >= 0)
                {
                    int current_class = find_instances_class(ds,
                            current_neighbor, n_classes, class_labels);
                    votes[current_class]++;
                }
                else break;
            }

            // find out the majority class of non_S[i]
            neighbor_majority_class = class_labels[0];
            neighbor_majority_class_count = votes[0];
            for (j = 1; j < n_classes; j++)
                if (votes[j] > neighbor_majority_class_count)
                {
                    neighbor_majority_class_count = votes[j];
                    neighbor_majority_class = class_labels[j];
                }

            // based on the majority class either add non_S[i] to S
            // or remember the S_size used to classify non_S[i]
            if (ds.y[non_S[i]] != neighbor_majority_class)
            {
                S[S_size++] = non_S[i];
                whole_non_S_classified_correctly = FALSE;
            }
            else
                last_train_S_size[non_S[i]] = S_size;
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
    free(S_copy);
    free(non_S);
    free(nearest);
    free(last_train_S_size);
    free(votes);

    return ds_reduced;
}

