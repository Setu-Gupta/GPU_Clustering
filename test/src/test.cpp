#include "../../src/GPUCluster.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdlib>
#include <cstring>

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
	std::cout << "Options for dbscan: <minPts> <R>" <<std::endl;

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
	float *x = (float *)malloc(n * sizeof(float));
	float *y = (float *)malloc(n * sizeof(float));
	for(unsigned int i = 0; i < n; i++)
	{
		x[i] = input[i].first;
		y[i] = input[i].second;
	}

	unsigned int *map = (unsigned int *)malloc(n * sizeof(unsigned int));
	memset(map, 0, n * sizeof(unsigned int));

	if(test_name == "kmeans")
	{
		unsigned int k = atoi(argv[3]);
		unsigned int num_iters = atoi(argv[4]);

		float msecs_cpu = kmeansCPU(x, y, map, n, k, num_iters);
		std::cout << "CPU took " << msecs_cpu << "ms" << std::endl;
		for(unsigned int i = 0; i < n; i++)
			mapCPU << x[i] << " " << y[i] << " " << map[i] << std::endl;
		
		memset(map, 0, n * sizeof(unsigned int));

		float msecs_gpu = kmeansGPU(x, y, map, n, k, num_iters);
		std::cout << "GPU took " << msecs_gpu << "ms" << std::endl;
		for(unsigned int i = 0; i < n; i++)
			mapGPU << x[i] << " " << y[i] << " " << map[i] << std::endl;

		float speedup = msecs_cpu / msecs_gpu;
		std::cout << "Speedup Obtained: " << speedup << "x" << std::endl;

	}
	else if(test_name == "dbscan")
	{
	
		int minPts =atoi(argv[3]);
		char *endptr;
		float R = strtof(argv[4], &endptr);
		float msecs_cpu=dbscanCPU(x, y,map,n, minPts, R);
		// float msecs_cpu=1;

		std::cout<<"CPU Time "<<msecs_cpu<<"ms"<<std::endl;
		

		for(unsigned int i = 0; i < n; i++){
			mapCPU << x[i] << " " << y[i] << " " << map[i] << std::endl;
			// std::cout<< x[i] << " " << y[i] << " " << map[i] << std::endl;
			
		}

		memset(map, 0, n * sizeof(unsigned int));

		float msecs_gpu = dbscanGPU(x, y, map, n,minPts,R);
		std::cout << "GPU took " << msecs_gpu << "ms" << std::endl;
		for(unsigned int i = 0; i < n; i++)
			mapGPU << x[i] << " " << y[i] << " " << map[i] << std::endl;

		float speedup = msecs_cpu / msecs_gpu;
		std::cout << "Speedup Obtained: " << speedup << "x" << std::endl;





	}
	else
	{
		std::cout << "TODO" << std::endl;
	}

	free(x);
	free(y);
	free(map);
	dataset.close();
	mapCPU.close();
	mapGPU.close();
}
