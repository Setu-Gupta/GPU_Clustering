#include "../../src/GPUCluster.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdlib>
#include <cstring>
#include <map>
#include <cuda_runtime.h>

/*
I/O format:
A list of the following entries
x <space> y <space> cluster_number
x and y are floating point values
*/

void print_usage()
{
	std::cout << "Usage: test <path to dataset> <kmeans | dbscan | pca> [options]" << std::endl;
	std::cout << "Options for kmeans: <k> <number of iterations>" << std::endl;
	std::cout << "Options for dbscan: <minPts> <R>" << std::endl;
	std::cout << "Options for dbscan: <number of iterations> <dbscan|kmeans i.e. the clustering algoro=ithm to be used post PCA> [kmeans or dbscan options]" << std::endl;
}

float get_accuracy(unsigned int n, unsigned int *a, unsigned int *b)
{
	// Assign each mapping to same cluster IDs
	unsigned int *a_new, *b_new;
	a_new = (unsigned int *)malloc(n * sizeof(unsigned int));
	b_new = (unsigned int *)malloc(n * sizeof(unsigned int));

	std::map<unsigned int, unsigned int> cluster_mappings;
	unsigned int cluster_id = 0;
	for(unsigned int i = 0; i < n; i++)
	{
		if(cluster_mappings.find(a[i]) == cluster_mappings.end())
		{
			cluster_mappings[a[i]] = cluster_id;
			a_new[i] = cluster_id;
			cluster_id++;
		}
		else
			a_new[i] = cluster_mappings[a[i]];
	}

	cluster_mappings.clear();
	cluster_id = 0;
	for(unsigned int i = 0; i < n; i++)
	{
		if(cluster_mappings.find(b[i]) == cluster_mappings.end())
		{
			cluster_mappings[b[i]] = cluster_id;
			b_new[i] = cluster_id;
			cluster_id++;
		}
		else
			b_new[i] = cluster_mappings[b[i]];
	}

	// // Compare
	// unsigned int diff = 0;
	// for(unsigned int i = 0; i < n; i++)
	// 	diff += (unsigned int)(a_new[i] == b_new[i]);

	// Compare
	unsigned int same = 0;
	// std::cout << std::endl;
	for(unsigned int i = 0; i < n; i++)
	{
		same += (unsigned int)(a_new[i] == b_new[i]);
		if(a_new[i] != b_new[i])
			std::cout << "At index = " << i << " CPU:" << a_new[i] << " GPU:" << b_new[i] << std::endl;
	}
	// std::cout << a_new[78] << " " << b_new[78] << std::endl;
	// std::cout << a_new[98] << " " << b_new[98] << std::endl;
	// std::cout << a_new[94] << " " << b_new[94] << std::endl;

	free(a_new);
	free(b_new);
	
	return 100*(float)same/(float)n;
}

int main(int argc, char **argv)
{
	// Input parsing
	if(argc < 3)
	{
		print_usage();
		return -1;
	}

	std::string dataset_path(argv[1]);
	std::string test_name(argv[2]);

	if(test_name != "kmeans" && test_name != "dbscan" && test_name != "pca")
	{
		std::cout << "Please specify a valid test name out of <kmeans | dbscan | pca>" << std::endl;
		std::cout << "Usage: test <path to dataset> <kmeans | dbscan | pca> <k if kmeans is used>" << std::endl;
		return -1;
	}

	if(test_name == "kmeans" && argc < 5)
	{
		print_usage();
		return -1;
	}

	if(test_name == "dbscan" && argc < 5)
	{
		print_usage();
		return -1;
	}

	if(test_name == "pca" && argc < 7)
	{
		print_usage();
		return -1;
	}


	
	// Open files to read and write from
	std::ifstream dataset(dataset_path);
	std::ofstream mapCPU(test_name + "CPU" + ".clusters", std::ios_base::out | std::ios_base::trunc);
	std::ofstream mapGPU(test_name + "GPU" + ".clusters", std::ios_base::out | std::ios_base::trunc);

	// Read from input files
	std::vector<std::pair<float, float>> input;
	std::string line;
	while(std::getline(dataset, line))
	{
		std::stringstream linestream(line);

		float x, y;
		linestream >> x >> y;

		input.push_back(std::make_pair(x, y));
	}

	unsigned int n = input.size();
	float *x, *y;
	cudaHostAlloc((void **) &x, n * sizeof(float), cudaHostAllocDefault);
	cudaHostAlloc((void **) &y, n * sizeof(float), cudaHostAllocDefault);
	for(unsigned int i = 0; i < n; i++)
	{
		x[i] = input[i].first;
		y[i] = input[i].second;
	}

	unsigned int *map_from_CPU = (unsigned int *)malloc(n * sizeof(unsigned int));
	unsigned int *map_from_GPU;
	cudaHostAlloc((void **) &map_from_GPU, n * sizeof(unsigned int), cudaHostAllocDefault);
	memset(map_from_CPU, 0, n * sizeof(unsigned int));
	memset(map_from_GPU, 0, n * sizeof(unsigned int));

	unsigned int *new_x_from_CPU = (unsigned int *)malloc(n * sizeof(unsigned int));
	unsigned int *new_y_from_CPU = (unsigned int *)malloc(n * sizeof(unsigned int));
	memset(new_x_from_CPU, 0, n * sizeof(unsigned int));
	memset(new_y_from_CPU, 0, n * sizeof(unsigned int));

	if(test_name == "kmeans")
	{
		// Read arguments
		unsigned int k = atoi(argv[3]);
		unsigned int num_iters = atoi(argv[4]);

		float msecs_cpu = kmeansCPU(x, y, map_from_CPU, n, k, num_iters);
		std::cout << "CPU took " << msecs_cpu << "ms" << std::endl;
		for(unsigned int i = 0; i < n; i++)
			mapCPU << x[i] << " " << y[i] << " " << map_from_CPU[i] << std::endl;

		float msecs_gpu = kmeansGPU(x, y, map_from_GPU, n, k, num_iters);
		std::cout << "GPU took " << msecs_gpu << "ms" << std::endl;
		for(unsigned int i = 0; i < n; i++)
			mapGPU << x[i] << " " << y[i] << " " << map_from_GPU[i] << std::endl;

		float speedup = msecs_cpu / msecs_gpu;
		std::cout << "Speedup Obtained: " << speedup << "x" << std::endl;
		std::cout << "Accuracy: " << get_accuracy(n, map_from_CPU, map_from_GPU) << "%" << std::endl;

	}
	else if(test_name == "dbscan")
	{
		// Read arguments
		int minPts = atoi(argv[3]);
		char *endptr;
		float R = strtof(argv[4], &endptr);

		float msecs_cpu = dbscanCPU(x, y, map_from_CPU,n, minPts, R);
		std::cout<<"CPU Time "<<msecs_cpu<<"ms"<<std::endl;
		for(unsigned int i = 0; i < n; i++)
			mapCPU << x[i] << " " << y[i] << " " << map_from_CPU[i] << std::endl;
			

		float msecs_gpu = dbscanGPU(x, y, map_from_GPU, n, minPts, R);
		std::cout << "GPU took " << msecs_gpu << "ms" << std::endl;
		for(unsigned int i = 0; i < n; i++)
			mapGPU << x[i] << " " << y[i] << " " << map_from_GPU[i] << std::endl;

		float speedup = msecs_cpu / msecs_gpu;
		std::cout << "Speedup Obtained: " << speedup << "x" << std::endl;
		std::cout << "Accuracy: " << get_accuracy(n, map_from_CPU, map_from_GPU) << "%" << std::endl;
	}
	else
	{
		// Read arguments
		int num_iters = atoi(argv[3]);
		std::string clustering_algo(argv[4]);

		float msecs_cpu = nipalsCPU(x, y, new_x_from_CPU, new_y_from_CPU, n, num_iters);
		std::cout<<"CPU Time "<<msecs_cpu<<"ms"<<std::endl;

		// if(clustering_algo == "dbscan")
		// {
		// 	int minPts = atoi(argv[5]);
		// 	char *endptr;
		// 	float R = strtof(argv[6], &endptr);
		// 	dbscanGPU(x, y, map_from_GPU, n, minPts, R);
		// }
		// else
		// {
		// 	unsigned int k = atoi(argv[3]);
		// 	unsigned int num_iters = atoi(argv[4]);	
		// 	kmeansGPU(x, y, map_from_GPU, n, k, num_iters);
		// }
		// for(unsigned int i = 0; i < n; i++)
		// 	mapGPU << x[i] << " " << y[i] << " " << map_from_GPU[i] << std::endl;

		for(unsigned int i = 0; i < n; i++)
			mapGPU << x[i] << " " << y[i] << " 0" << std::endl;
	}

	cudaFreeHost(x);
	cudaFreeHost(y);
	free(map_from_CPU);
	cudaFreeHost(map_from_GPU);
	free(new_x_from_CPU);
	free(new_y_from_CPU);
	dataset.close();
	mapCPU.close();
	mapGPU.close();
}
