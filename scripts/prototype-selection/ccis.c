#include "string.h"
#include "stdlib.h"
#include "math.h"
#include "float.h"
#include "stdio.h"

#include "psutils.h"

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
	// graph's ith row is related to ith element of S, not S[i]th
	// element of the dataset
	int* graph = malloc(sizeof(int) * S_size * n_classes);
	fill_int_array(graph, S_size * n_classes, -1);

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
            if (graph[i * n_classes + current_class] < 0 ||
                    current_squared_dist < min_squared_dists[current_class])
            {
                graph[i * n_classes + current_class] = S[j];
                min_squared_dists[current_class] = current_squared_dist;
            }
        }
        
        // at the same time computing within and between in-degrees
        // and total in-degrees for Gwc and Gbc
        for (j = 0; j < n_classes; j++)
        {
            int current_nn = graph[i * n_classes + j];
            if (current_nn <= 0)
				continue;
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

	// Scored instances should be sorted in order of decreasing score
	if (a_instance.score < b_instance.score)
		return 1;
	else if (a_instance.score > b_instance.score)
		return -1;
	
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
	fill_int_array(ccnn_graph, ds.n_instances * n_classes, -1);
	within_in_degrees = malloc(sizeof(int) * ds.n_instances);
	memset(within_in_degrees, 0, sizeof(int) * ds.n_instances);
	between_in_degrees = malloc(sizeof(int) * ds.n_instances);
	memset(between_in_degrees, 0, sizeof(int) * ds.n_instances);
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
    for (i = k0; i < upper_bound; i++)
    {
        S[S_size] = scored_instances[i].index;
        epsilon_S = loo_score(ds, n_classes, class_labels, S, S_size);
        if (epsilon_S <= epsilon_A)
            break;
        epsilon_temp = loo_score(ds, n_classes, class_labels, S, S_size + 1);
        if (epsilon_temp < epsilon_S)
            S_size += 1;
        else
            break;
    }

	// THIN method
	Sprev_within = malloc(sizeof(int) * ds.n_instances);
	memset(Sprev_within, 0, sizeof(int) * ds.n_instances);
	Sprev_between = malloc(sizeof(int) * ds.n_instances);
	memset(Sprev_between, 0, sizeof(int) * ds.n_instances);
	S1_within = malloc(sizeof(int) * ds.n_instances);
	S1_between = malloc(sizeof(int) * ds.n_instances);

	compute_degrees(ds, S_size, S, n_classes, class_labels, Sprev_within,
			Sprev_between);

	Sf = malloc(sizeof(int) * ds.n_instances);
	S1 = malloc(sizeof(int) * ds.n_instances);
	Sf_size = 0;
	S1_size = 0;
	// Splitting S into Sf and S1
	for (i = 0; i < S_size; i++)
		if (Sprev_between[S[i]] > 0)
			Sf[Sf_size++] = S[i];
		else
			S1[S1_size++] = S[i];
	
	while (TRUE)
	{
		int* temp;
		memset(S1_within, 0, sizeof(int) * ds.n_instances);
		memset(S1_between, 0, sizeof(int) * ds.n_instances);
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
		temp = Sprev_within;
		Sprev_within = S1_within;
		S1_within = temp;
		temp = Sprev_between;
		Sprev_between = S1_between;
		S1_between = temp;
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

