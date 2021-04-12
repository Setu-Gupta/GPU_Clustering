#include "GPUCluster.h"

#define THREADS_PER_BLOCK	1024

// Store the x and y coordiates of datapoints in texture memory
texture<float, 1, cudaReadModeElementType> x_tex;
texture<float, 1, cudaReadModeElementType> y_tex;

__global__
void initDataToZero(unsigned int k, float *cluster_data_sum_x, float *cluster_data_sum_y, float *cluster_data_count)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < k)
	{
		cluster_data_sum_x[idx] = 0.0f;
		cluster_data_sum_y[idx] = 0.0f;
		cluster_data_count[idx] = 0.0;
	}
}

__global__
void assignCluster(unsigned int *map, unsigned int n, unsigned int k,
	float *clusters_x, float *clusters_y,
	float *cluster_data_sum_x, float *cluster_data_sum_y, float *cluster_data_count)
{
	extern __shared__ float local_cluster_x_y_count[];	// An array of size 5k with triplet elements for each cluster in the format x_old,y_old,x_new_sum,y_new_sum,count
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	// Step 1: Initialize local memory
	for(int i = threadIdx.x; i < k; i += blockDim.x)
	{
		local_cluster_x_y_count[5*i] = clusters_x[i];		// Stores the old cluster x coordinate 
		local_cluster_x_y_count[5*i + 1] = clusters_y[i];	// Stores the old cluster y coordinate
		local_cluster_x_y_count[5*i + 2] = 0.0f;			// Stores the new cluster x coordinate sum
		local_cluster_x_y_count[5*i + 3] = 0.0f;			// Stores the new cluster y coordinate sum
		local_cluster_x_y_count[5*i + 4] = 0.0f;			// Stores the new cluster count
	}
	__syncthreads();

	// Step 2: Compute the closest cluster for each point
	if(idx < n)
	{
		float min_dist = FLT_MAX;
		int min_cluster = -1;
		float cur_x = tex1Dfetch(x_tex, idx);
		float cur_y = tex1Dfetch(y_tex, idx);
		for(int j = 0; j < k; j++)
		{
			// Compute the Euclidean distance from each cluster
			float xdiff = (cur_x - local_cluster_x_y_count[5*j]);
			float ydiff = (cur_y - local_cluster_x_y_count[5*j + 1]);
			float dist = xdiff * xdiff;
			dist += ydiff * ydiff;

			// Find the closest cluster
			if(dist < min_dist)
			{
				min_dist = dist;
				min_cluster = j;
			}
		}	
		atomicAdd(&local_cluster_x_y_count[5*min_cluster + 2], cur_x);
		atomicAdd(&local_cluster_x_y_count[5*min_cluster + 3], cur_y);
		atomicAdd(&local_cluster_x_y_count[5*min_cluster + 4], 1.0f);
		map[idx] = (unsigned int) min_cluster;
		// printf("%d - > %f %d\n", idx, min_dist, min_cluster);
	}
	__syncthreads();

	//Step 3: Update state in global memory
	for(int i = threadIdx.x; i < k; i += blockDim.x)
	{
		atomicAdd(&cluster_data_sum_x[i], local_cluster_x_y_count[5*i + 2]);		// Stores the new cluster x coordinate sum
		atomicAdd(&cluster_data_sum_y[i], local_cluster_x_y_count[5*i + 3]);		// Stores the new cluster xy coordinate sum
		atomicAdd(&cluster_data_count[i], local_cluster_x_y_count[5*i + 4]);		// Stores the new cluster count
	}
}

__global__
void computeNewClusters(float *clusters_x, float *clusters_y,
	unsigned int k, float *cluster_data_sum_x, float *cluster_data_sum_y, float *cluster_data_count)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < k)
	{
		float x_sum = cluster_data_sum_x[idx];
		float y_sum = cluster_data_sum_y[idx];
		float count = cluster_data_count[idx];
		clusters_x[idx] = x_sum / count;
		clusters_y[idx] = y_sum / count;
	}	
}

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
float kmeansGPU(float *x, float *y, unsigned int *map, unsigned int n, unsigned int k, unsigned int num_iters)
{
	struct timespec start_gpu, end_gpu;
	float msecs_gpu;
	clock_gettime(CLOCK_MONOTONIC, &start_gpu);

	// Create pointers memory on device for various arrays
	float *d_clusters_x, *d_clusters_y;	// Holds the x and y coordinates of clusters
	float *d_cluster_data_sum_x, *d_cluster_data_sum_y, *d_cluster_data_count;	// Holds the sum and count required to calculate next clusters
	float *d_x, *d_y; 	// Holds the x and y coordinate of inut datapoints
	unsigned int *d_map;	// Holds the mappings

	// Allocate memory on device
	cudaMalloc(&d_clusters_x, k * sizeof(float));
	cudaMalloc(&d_clusters_y, k * sizeof(float));
	cudaMalloc(&d_cluster_data_sum_x, k * sizeof(float));
	cudaMalloc(&d_cluster_data_sum_y, k * sizeof(float));
	cudaMalloc(&d_cluster_data_count, k * sizeof(float));
	cudaMalloc(&d_x, n * sizeof(float));
	cudaMalloc(&d_y, n * sizeof(float));
	cudaMalloc(&d_map, n * sizeof(unsigned int));

	// Copy data onto the device
	cudaMemcpy(d_clusters_x, x, k * sizeof(float), cudaMemcpyHostToDevice);	// Initialize clusters as the first k datapoints
	cudaMemcpy(d_clusters_y, y, k * sizeof(float), cudaMemcpyHostToDevice);	// Initialize clusters as the first k datapoints
	cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice);

	// Bind Textures
	cudaBindTexture(NULL, x_tex, d_x, n * sizeof(float));
	cudaBindTexture(NULL, y_tex, d_y, n * sizeof(float));

	unsigned int k_grid_dim = (k + (THREADS_PER_BLOCK - 1)) / THREADS_PER_BLOCK;
	unsigned int n_grid_dim = (n + (THREADS_PER_BLOCK - 1)) / THREADS_PER_BLOCK;
	unsigned int shmem_size = 5 *k * sizeof(float);
	for(unsigned int iter = 0; iter < num_iters; iter++)
	{
		initDataToZero<<<k_grid_dim, THREADS_PER_BLOCK>>>(	k,
															d_cluster_data_sum_x,
															d_cluster_data_sum_y,
															d_cluster_data_count);
		assignCluster<<<n_grid_dim, THREADS_PER_BLOCK, shmem_size>>>(	d_map,
																		n,
																		k,
																		d_clusters_x,
																		d_clusters_y,
																		d_cluster_data_sum_x, 
																		d_cluster_data_sum_y, 
																		d_cluster_data_count);
		computeNewClusters<<<k_grid_dim, THREADS_PER_BLOCK>>>(	d_clusters_x,
																d_clusters_y,
																k,
																d_cluster_data_sum_x,
																d_cluster_data_sum_y,
																d_cluster_data_count);

	}

	// Bring back mappings to host
	cudaMemcpy(map, d_map, n * sizeof(unsigned int), cudaMemcpyDeviceToHost);

	// Unbind textures
	cudaUnbindTexture(x_tex);
	cudaUnbindTexture(y_tex);

	// Cleaup
	cudaFree(d_clusters_x);
	cudaFree(d_clusters_y);
	cudaFree(d_cluster_data_sum_x);
	cudaFree(d_cluster_data_sum_y);
	cudaFree(d_cluster_data_count);
	cudaFree(d_x);
	cudaFree(d_y);
	cudaFree(d_map);

	clock_gettime(CLOCK_MONOTONIC, &end_gpu);
	msecs_gpu = 1000.0 * (end_gpu.tv_sec - start_gpu.tv_sec) + (end_gpu.tv_nsec - start_gpu.tv_nsec)/1000000.0;

	return msecs_gpu;
}