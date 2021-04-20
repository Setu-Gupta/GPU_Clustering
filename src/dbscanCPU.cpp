#include "GPUCluster.h"
#include <assert.h>
#include <queue>

/*
Returns the euclidean distance betwee (xi, yi) and (xj, yj)
*/
float euclidean(float xi, float yi, float xj, float yj)
{
	float dist=pow((xi-xj),2)+pow((yi-yj),2);
	return sqrt(dist);
}

/*
Performs exclusive scan on V of length n. Stores result in indices
*/
void exclusive_scan(int* V, int* indices, unsigned int n)
{
	indices[0] = 0;
	for(int i = 1; i < n; i++)
		indices[i] = indices[i-1] + V[i-1];
}

/*
Performs BFS on graph (V, edges, indices) of n vertices starting from node v and assignes each point to cluster in map.
Only visits vertices not in visited.
*/
void bfs(int* V, int* edges, int* indices, char* visited, unsigned int* map, int v, unsigned int n, int cluster)
{
	std::queue<int> queue;
	queue.push(v);
	visited[v] = 1;

	while(!queue.empty())
	{
		int t = queue.front();
		queue.pop();
		map[t] = cluster;
		for(int off = 0; off < V[t]; off++)
		{
			int nbour = edges[indices[t] + off];
			if(!visited[nbour])
			{
				visited[nbour] = 1;
				map[nbour] = cluster;
				queue.push(nbour);
			}
		}
	}
}

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
float dbscanCPU(float* x, float* y,	unsigned int* map, unsigned int n, int minPts, float R)
{
	struct timespec start_cpu, end_cpu;
	float msecs_cpu;
	clock_gettime(CLOCK_MONOTONIC, &start_cpu);


	for(int i = 0; i < n; i++)
		map[i]=0; // initialise all points. 0 means noise

	int *V = (int* ) malloc(sizeof(int) * n);	// V holds the number of edges for every point
	memset(V, 0, n * sizeof(int));

	// Calculating the number of neighbours of each point
	for(int i = 0; i < n; i++)
	{
		for(int j = 0; j < n; j++)
		{
			if(j != i)	// No self edges
				if(euclidean(x[i], y[i], x[j], y[j]) <= R)
					V[i]++;
		}
	}

	// Array storing the index where the edges of a given element starts
	int* indices = (int*) malloc(sizeof(int) * n);
	exclusive_scan(V, indices, n);	// exclusive scan to populate the indices based on the values in V

	// total edges will be the indices value of last element plus the neighbours it has.
	int numEdges = indices[n-1] + V[n-1];
	int* edges = (int*)malloc(sizeof(int) * numEdges);
	int* core = (int*) malloc(sizeof(int) * n);
	
	// populating the edges array
	for(int i = 0; i < n; i++)
	{
		int count = 0;
		for(int j = 0; j < n; j++)
		{
			if(j != i)
			{	
				if(euclidean(x[i], y[i], x[j], y[j]) <= R)
				{
					// std::cout << i << " connected to " << j << " on CPU\n";
					edges[indices[i] + count] = j;	// Add an edge from i to j
					count++;
				}
			}
		}
		core[i] = (int)(count >= minPts);		// A point is core of it has more edges than minPts
	}
	// std::cout << numEdges << std::endl;
	int cluster = 1;
	char *visited = (char* ) malloc(n * sizeof(char));
	memset(visited, 0, n * sizeof(char));

	for(int v = 0; v < n; v++)
	{
		if(visited[v] == 0 && core[v] == 1)	//	bfs from all unvisited core points
		{
			bfs(V, edges, indices, visited, map, v, n, cluster);
			cluster++;
		}
	}

	clock_gettime(CLOCK_MONOTONIC, &end_cpu);
	msecs_cpu = 1000.0 * (end_cpu.tv_sec - start_cpu.tv_sec) + (end_cpu.tv_nsec - start_cpu.tv_nsec)/1000000.0;
	
	free(indices);
	free(edges);
	free(V);
	free(core);
	free(visited);

	return msecs_cpu;
}

