#include "GPUCluster.h"

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
float kmeansCPU(float *x, float *y, unsigned int *map, unsigned int n, unsigned int k, unsigned int num_iters)
{
	float clusters_x[k];	// Holds the current x coordinate of k clusters
	float clusters_y[k];	// Holds the current y coordinate of k clusters
	
	struct timespec start_cpu, end_cpu;
	float msecs_cpu;
	clock_gettime(CLOCK_MONOTONIC, &start_cpu);

	// Step 1: Initialize clusters
	for(unsigned int i = 0; i < k; i++)
	{
		clusters_x[i] = x[i];
		clusters_y[i] = y[i];
	}

	// Allocate space to store the sum of coordinates in a cluster and the number of datapoints in each cluster
	float cluster_data_sum_x[k];
	float cluster_data_sum_y[k];
	unsigned int cluster_data_count[k];

	for(unsigned int iter = 0; iter < num_iters; iter++)
	{
		for(unsigned int i = 0; i < k; i++)
		{
			cluster_data_sum_x[i] = 0.0f;
			cluster_data_sum_y[i] = 0.0f;
			cluster_data_count[i] = 0;
		}

		// Step 2: Map points to the nearest cluster
		for(unsigned int i = 0; i < n; i++)
		{
			float min_dist = FLT_MAX;
			int min_cluster = -1;
			for(unsigned int j = 0; j < k; j++)
			{
				// Compute the Euclidean distance from each cluster
				float dist = pow((x[i] - clusters_x[j]), 2);
				dist += pow((y[i] - clusters_y[j]), 2);

				// Find the closest cluster
				if(dist < min_dist)
				{
					min_dist = dist;
					min_cluster = j;
				}
			}

			map[i] = min_cluster;	// Update map to reflect the closest cluster
			cluster_data_count[min_cluster]++;
			cluster_data_sum_x[min_cluster] += x[i];
			cluster_data_sum_y[min_cluster] += y[i];
		}

		// Step 3: Update clusters
		for(unsigned int i = 0; i < k; i++)
		{
			clusters_x[i] = cluster_data_sum_x[i] / (float)cluster_data_count[i];
			clusters_y[i] = cluster_data_sum_y[i] / (float)cluster_data_count[i];
		}
	}
	
	clock_gettime(CLOCK_MONOTONIC, &end_cpu);
	msecs_cpu = 1000.0 * (end_cpu.tv_sec - start_cpu.tv_sec) + (end_cpu.tv_nsec - start_cpu.tv_nsec)/1000000.0;
	return msecs_cpu;
}