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
Returns: Time spent in computation in milliseconds
*/
float kmeansCPU(float *x, float *y, unsigned int *map, unsigned int n, unsigned int k, unsigned int num_iters);

/*
Performs kmeans clustering on GPU
Args:
	x:	x coordinates of datapoints (n length float array)
	y:	y coordinates of datapoints (n length float array)
	map:	array to store cluster and datapoint mappings (n length unsigned int array)
	n:	number of datapoints
	k:	number of clusters to create
	num_iters:	number of iteration to run kmeans for
Returns: Time spent in computation in milliseconds
*/
float kmeansGPU(float *x, float *y, unsigned int *map, unsigned int n, unsigned int k, unsigned int num_iters);


/*
Performs DBSCAN clustering on CPU
Args:
	x:	x coordinates of datapoints (n length float array)
	y:	y coordinates of datapoints (n length float array)
	map:	array to store cluster and datapoint mappings (n length unsigned int array) . Value of 255 means Noise point.
	n:	number of datapoints
	minPts:	minimum number of neighbours for a point to be labelled as core
	R:	minimum euclidean distance between two points for them to be neighbours
*/
float dbscanCPU(float* x, float* y,	unsigned int* map,unsigned int n,int minPts,float R);
#endif
