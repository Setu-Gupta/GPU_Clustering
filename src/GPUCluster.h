#ifndef GPU_CLUSTER_H
#define GPU_CLUSTER_H

#include <math.h>
#include <float.h>
#include <ctime>

/*
Performs kmeans clustering on CPU
Args:
	x:	x coordinates of datapoints (n length float array)
	y:	y coordinates of datapoints (n length float array)
	map:	array to store cluster and datapoint mappings (n length unsigned int array)
	n:	number of datapoints
	k:	number of clusters to create
	num_iters:	number of iteration to run kmeans for
*/
float kmeansCPU(float *x, float *y, unsigned int *map, unsigned int n, unsigned int k, unsigned int num_iters);

#endif