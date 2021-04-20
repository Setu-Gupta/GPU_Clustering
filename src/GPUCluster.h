#ifndef GPU_CLUSTER_H
#define GPU_CLUSTER_H

#include <math.h>
#include <float.h>
#include <ctime>
#include <string.h>
#include <iostream>
#include <stdio.h>

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
	map:	array to store cluster and datapoint mappings (n length unsigned int array). Value of 0 means Noise point.
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
	map:	array to store cluster and datapoint mappings (n length unsigned int array). Value of 0 means Noise point.
	n:	number of datapoints
	minPts:	minimum number of neighbours for a point to be labelled as core
	R:	minimum euclidean distance between two points for them to be neighbours
Returns: Time spent in computation in milliseconds
*/
float dbscanCPU(float* x, float* y,	unsigned int* map, unsigned int n, int minPts, float R);

/*
Performs DBSCAN clustering on GPU
Args:
	x:	x coordinates of datapoints (n length float array)
	y:	y coordinates of datapoints (n length float array)
	map:	array to store cluster and datapoint mappings (n length unsigned int array). Value of 0 means Noise point.
	n:	number of datapoints
	minPts:	minimum number of neighbours for a point to be labelled as core
	R:	minimum euclidean distance between two points for them to be neighbours
Returns: Time spent in computation in milliseconds
*/
float dbscanGPU(float* x, float* y,	unsigned int* map, unsigned int n, int minPts, float R);

/*
Performs NIPAL's PCA on CPU
Args:
	x:	x coordinates of datapoints (n length float array)
	y:	y coordinates of datapoints (n length float array)
	new_x:	transformed x coordinates of datapoints (n length float array)
	new_y:	treansformed y coordinates of datapoints (n length float array)
	n:	number of datapoints
	num_iters:	number of iteration to run NIPALs for
Returns: Time spent in computation in milliseconds
*/
float nipalsCPU(float* x, float* y,	float* new_x, float* new_y, unsigned int n, unsigned int num_iters);

#endif
