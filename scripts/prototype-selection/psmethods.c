#include "stdlib.h"
#include "string.h"
#include "stdio.h"
#include "time.h"
#include "float.h"
#include "math.h"

#include "psmethods.h"

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
            return class_labels[i];
    }
    
    return -1;
}

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

int loo_score(Dataset ds, int n_classes, int* class_labels, int* S_elements,
        int S_size)
{
    int i, j;
    int result = 0;
    for (i = 0; i < S_size; i++)
    {
		int i_index = (S_elements == NULL) ? i : S_elements[i];
        int nearest = -1;
        int nearest_class = -1;
        flpoint min_squared_dist = DBL_MAX;

        for (j = 0; j < i; j++)
        {
			int j_index = (S_elements == NULL) ? j : S_elements[j];
            flpoint current_squared_dist = squared_dist(ds.n_features,
                    ds.X + i_index * ds.n_features,
                    ds.X + j_index * ds.n_features);
            if (current_squared_dist < min_squared_dist)
            {
                min_squared_dist = current_squared_dist;
                nearest = j_index;
            }
        }

        for (j = i + 1; j < S_size; j++)
        {
			int j_index = (S_elements == NULL) ? j : S_elements[j];
            flpoint current_squared_dist = squared_dist(ds.n_features,
                    ds.X + i_index * ds.n_features,
                    ds.X + j_index * ds.n_features);
            if (current_squared_dist < min_squared_dist)
            {
                min_squared_dist = current_squared_dist;
                nearest = j_index;
            }
        }

        nearest_class = find_instances_class(ds, nearest, n_classes,
                class_labels);

        if (ds.y[i] != class_labels[nearest_class])
            result += 1;
    }

    return result;
}

void compute_degrees(Dataset ds, int S_size, const int* S, int n_classes,
		const int* class_labels, int* within, int* between)
{
	int i, j;
    flpoint* min_squared_dists = malloc(sizeof(flpoint) * n_classes);
	int* graph = malloc(sizeof(flpoint) * S_size * n_classes);

    for (i = 0; i < S_size; i++)
    {
        fill_flpoint_array(min_squared_dists, n_classes, DBL_MAX);
        for (j = 0; j < S_size; j++)
        {
            int current_class;
            flpoint current_squared_dist;

            if (j == i) continue;

            current_class =
                find_instances_class(ds, S[j], n_classes, class_labels);
            current_squared_dist = squared_dist(ds.n_features,
                    ds.X + S[i] * ds.n_features, ds.X + S[j] * ds.n_features);
            if (graph[S[i] * n_classes + current_class] < 0 ||
                    current_squared_dist < min_squared_dists[current_class])
            {
                graph[S[i] * n_classes + current_class] = S[j];
                min_squared_dists[current_class] = current_squared_dist;
            }
        }
        
        // at the same time computing within and between in-degrees
        // and total in-degrees for Gwc and Gbc
        for (j = 0; j < n_classes; j++)
        {
            int current_nn = graph[S[i] * n_classes + j];
            if (current_nn <= 0) continue;
            if (class_labels[j] == ds.y[S[i]])
                within[current_nn]++;
            else
                between[current_nn]++;
        }
    }

	free(graph);
	free(min_squared_dists);
}

flpoint ccnn_score(flpoint pw, flpoint pb)
{
	return pw * log(2.0 * pw / (pw + pb)) -
		pb * log(2.0 * pb / (pw + pb));
}

int compare_ccnn_instances(const void* a, const void* b)
{
	CCNNInstance a_instance = *(CCNNInstance*) a;
	CCNNInstance b_instance = *(CCNNInstance*) b;

	if (a_instance.score < b_instance.score)
		return -1;
	else if (a_instance.score > b_instance.score)
		return 1;
	
	return 0;
}


Dataset ccis_reduce(Dataset ds)
{
	int i, j;
	int n_classes;
	int* class_labels;
	flpoint* min_squared_dists;
	int* ccnn_graph;
	int *within_in_degrees, *between_in_degrees;
	int total_within_in_degree, total_between_in_degree;
	CCNNInstance* scored_instances;
	int k0, upper_bound;
	int *Sprev_within, *Sprev_between, *S1_within, *S1_between;
	int *S, *Sf, *S1;
	int S_size, Sf_size, S1_size, St_size;
	int epsilon_A, epsilon_temp, epsilon_S, epsilon_Sf, epsilon_Sf_plus_St;
	int St_index;
	Dataset ds_reduced;

    count_classes(ds, &n_classes, &class_labels);
    
    // compute LOO-error on the whole training set
    epsilon_A = loo_score(ds, n_classes, class_labels, NULL, ds.n_instances);

    // construct the CCNN graph
	ccnn_graph = malloc(sizeof(int) * ds.n_instances * n_classes);
	within_in_degrees = malloc(sizeof(int) * ds.n_instances);
	between_in_degrees = malloc(sizeof(int) * ds.n_instances);
    min_squared_dists = malloc(sizeof(flpoint) * n_classes);
	total_within_in_degree = 0;
	total_between_in_degree = 0;
    for (i = 0; i < ds.n_instances; i++)
    {
        fill_flpoint_array(min_squared_dists, n_classes, DBL_MAX);
        for (j = 0; j < ds.n_instances; j++)
        {
            int current_class;
            flpoint current_squared_dist;

            if (j == i) continue;

            current_class =
                find_instances_class(ds, j, n_classes, class_labels);
            current_squared_dist = squared_dist(ds.n_features,
                    ds.X + i * ds.n_features, ds.X + j * ds.n_features);
            if (ccnn_graph[i * n_classes + current_class] < 0 ||
                    current_squared_dist < min_squared_dists[current_class])
            {
                ccnn_graph[i * n_classes + current_class] = j;
                min_squared_dists[current_class] = current_squared_dist;
            }
        }
        
        // at the same time computing within and between in-degrees
        // and total in-degrees for Gwc and Gbc
        for (j = 0; j < n_classes; j++)
        {
            int current_nn = ccnn_graph[i * n_classes + j];
            if (current_nn <= 0) continue;
            if (class_labels[j] == ds.y[i])
            {
                within_in_degrees[current_nn]++;
                total_within_in_degree++;
            }
            else
            {
                between_in_degrees[current_nn]++;
                total_between_in_degree++;
            }
        }
    }

    // score all elements and sort them according to score
    scored_instances = malloc(sizeof(CCNNInstance) *
            ds.n_instances);
    for (i = 0; i < ds.n_instances; i++)
        scored_instances[i].index = i;
        scored_instances[i].score = ccnn_score(
					(flpoint)within_in_degrees[i] / total_within_in_degree,
					(flpoint)between_in_degrees[i] / total_between_in_degree);

    qsort(scored_instances, ds.n_instances, sizeof(CCNNInstance),
            compare_ccnn_instances);
    k0 = (int)fmax(n_classes, ceil(epsilon_A * 0.5));

    // add to S k0 instances with highest scores
	S = malloc(sizeof(int) * ds.n_instances);
	S_size = 0;
    for (i = 0; i < k0; i++)
        S[S_size++] = scored_instances[i].index;

    // find number of instances with positive score
	upper_bound = ds.n_instances;
    for (i = k0; i < ds.n_instances; i++)
        if (scored_instances[i].score <= 0)
        {
            upper_bound = i;
            break;
        }

    // main loop of CC method
    for (i = k0 + 1; i < upper_bound; i++)
    {
        S[S_size] = scored_instances[i].index;
        epsilon_S = loo_score(ds, n_classes, class_labels, S, S_size);
        if (epsilon_S >= epsilon_A)
            break;
        epsilon_temp = loo_score(ds, n_classes, class_labels, S, S_size + 1);
        if (epsilon_temp < epsilon_S)
            S_size += 1;
        else
            break;
    }

	// THIN method
	Sprev_within = malloc(sizeof(int) * S_size);
	Sprev_between = malloc(sizeof(int) * S_size);
	S1_within = malloc(sizeof(int) * S_size);
	S1_between = malloc(sizeof(int) * S_size);

	compute_degrees(ds, S_size, S, n_classes, class_labels, Sprev_within,
			Sprev_between);

	Sf = malloc(sizeof(int) * ds.n_instances);
	S1 = malloc(sizeof(int) * ds.n_instances);
	Sf_size = 0;
	S1_size = 0;
	// Splitting S into Sf and S1
	for (i = 0; i < S_size; i++)
		if (Sprev_within[S[i]] > 0)
			Sf[Sf_size++] = S[i];
		else
			S1[S1_size++] = S[i];
	
	while (TRUE)
	{
		compute_degrees(ds, S1_size, S1, n_classes, class_labels, S1_within,
				S1_between);
		St_size = 0;
		for (i = 0; i < S1_size; i++)
			if (S1_between[S1[i]] > 0 &&
					(Sprev_between[S1[i]] > 0 ||
					 Sprev_within[S1[i]] > 0))
				Sf[Sf_size + St_size++] = S1[i];

		epsilon_Sf = loo_score(ds, n_classes, class_labels,
				Sf, Sf_size);
		epsilon_Sf_plus_St = loo_score(ds, n_classes, class_labels,
				Sf, Sf_size + St_size);

		if (epsilon_Sf_plus_St >= epsilon_Sf)
			break;

		// Sprev := S1
		compute_degrees(ds, S1_size, S1, n_classes, class_labels, Sprev_within,
				Sprev_between);
		// S1 := S - Sf which is equivalent to filtering out St elements
		// from S1
		St_index = 0;
		i = 0;
		while (i < S1_size)
		{
			if (St_index < St_size && S1[i] == Sf[Sf_size + St_index])
			{
				if (i + 1 != S1_size)
					S1[i] = S1[i + 1];

				S1_size--;
				St_index++;
			}
			else
				i++;
		}
		// Sf := Sf + St
		Sf_size = Sf_size + St_size;
	}

    // form a new dataset with only selected instances
    ds_reduced = alloc_dataset(ds.n_features, Sf_size);
    for (i = 0; i < Sf_size; i++)
    {
        memcpy(ds_reduced.X + ds.n_features * i,
                ds.X + ds.n_features * Sf[i], sizeof(flpoint) * ds.n_features);
        ds_reduced.y[i] = ds.y[Sf[i]];
    }

	free(class_labels);
	free(ccnn_graph);
	free(within_in_degrees);
	free(between_in_degrees);
	free(min_squared_dists);
	free(S);
	free(scored_instances);
	free(Sf);
	free(S1);
	free(Sprev_within);
	free(Sprev_between);
	free(S1_within);
	free(S1_between);

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
