#ifndef PSUTILS_H
#define PSUTILS_H

#include "psmethods.h"

int compare_ints(const void* a, const void* b);
flpoint squared_dist(int N, flpoint* a, flpoint* b);
void fill_int_array(int* array, int N, int fill);
void fill_flpoint_array(flpoint* array, int N, flpoint fill);
void count_classes(const Dataset ds, int* n_classes, int** class_labels);
int find_instances_class(const Dataset ds, int instance, int n_classes,
		const int* class_labels);

#endif // PSUTILS_H

