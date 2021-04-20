#include <thrust/scan.h>
#include "GPUCluster.h"

#define THREADS_PER_BLOCK   1024

// Texture memory for x and y coordinates of each point
static texture<float, 1, cudaReadModeElementType> x_tex;
static texture<float, 1, cudaReadModeElementType> y_tex;

// __global__
// void  kernel_getV(unsigned int n, int* V, unsigned char* core, int minPts, float R)
// {
// 	// Shared memory to store x and y coodinates of points
// 	__shared__ float x[THREADS_PER_BLOCK << 2];
// 	__shared__ float y[THREADS_PER_BLOCK << 2];

// 	int tx = threadIdx.x + (blockDim.x * blockIdx.x);
	
// 	// Local state
// 	float px = 0, py = 0;	// x and y coordinate of point assigned to tx
// 	int edge_count = 0;
// 	if(tx < n)
// 	{
// 		px = tex1Dfetch(x_tex, tx);
// 		py = tex1Dfetch(y_tex, tx);
// 	}

// 	for(int start = 0; start < n; start += (THREADS_PER_BLOCK << 2))	// Bring in chunks of THREADS_PER_BLOCK * 4
// 	{
// 		// Step 1: Bring some points to shared memory 
// 		for(int i = threadIdx.x; i < (THREADS_PER_BLOCK << 2) && (start + i) < n; i += THREADS_PER_BLOCK)
// 		{
// 			x[i] = tex1Dfetch(x_tex, start + i);
// 			y[i] = tex1Dfetch(y_tex, start + i);
// 		}
// 		__syncthreads();

// 		// Step 2: Find number of edges
// 		if(tx < n)
// 		{
// 			for(int i = 0; i < (THREADS_PER_BLOCK << 2) && (i + start) < n; i++)	// Iterate over locally stored points to see if are near (px, py)
// 			{
// 				float dist = sqrt(pow(px - x[i], 2) + pow(py - y[i], 2));
// 				if(dist <= R && tx != (start+i))
// 					edge_count++;
// 			}
// 		}
// 		__syncthreads();
// 	}

// 	// Update core and V in global memory
// 	if(tx < n)
// 	{
// 		V[tx] = edge_count;
// 		core[tx] = (edge_count >= minPts);
// 	}
// }

// __global__
// void kernel_getEdges(unsigned int n, int* indices, int* edges, float R)
// {
// 	// Shared memory to store x and y coodinates of points
// 	__shared__ float x[THREADS_PER_BLOCK << 2];
// 	__shared__ float y[THREADS_PER_BLOCK << 2];

// 	int tx = threadIdx.x + (blockDim.x * blockIdx.x);
	
// 	// Local state
// 	float px = 0, py = 0;	// x and y coordinate of point assigned to tx
// 	int edge_count = 0;
// 	int idx = 0;
// 	if(tx < n)
// 	{
// 		px = tex1Dfetch(x_tex, tx);
// 		py = tex1Dfetch(y_tex, tx);
// 		idx = indices[tx];
// 	}

// 	for(int start = 0; start < n; start += (THREADS_PER_BLOCK << 2))	// Bring in chunks of THREADS_PER_BLOCK * 4
// 	{
// 		// Step 1: Bring some points to shared memory 
// 		for(int i = threadIdx.x; i < (THREADS_PER_BLOCK << 2) && (start + i) < n; i += THREADS_PER_BLOCK)
// 		{
// 			x[i] = tex1Dfetch(x_tex, start + i);
// 			y[i] = tex1Dfetch(y_tex, start + i);
// 		}
// 		__syncthreads();

// 		// Step 2: Find edges
// 		if(tx < n)
// 		{
// 			for(int i = 0; i < (THREADS_PER_BLOCK << 2) && (i + start) < n; i++)	// Iterate over locally stored points to see if are near (px, py)
// 			{
// 				float dist = sqrt(pow(px - x[i], 2) + pow(py - y[i], 2));
// 				if(dist <= R && tx != (start+i))
// 				{
// 					edges[idx + edge_count] = (i+start); 
// 					edge_count++;
// 				}

// 			}
// 		}
// 		__syncthreads();
// 	}
// }


// /*
// Initializes array a of size to 0
// */
// __global__
// void kernel_init(unsigned char* a,int size)
// {
// 	int tx = threadIdx.x + (blockDim.x * blockIdx.x);
// 	if(tx < size)
// 		a[tx] = 0;
// }

// /*
// Perform a single frontier step computation of BFS
// */
// __global__
// void kernel_bfs_child(int* V, int* indices, int* edges, unsigned char* Fa, unsigned char* Xa, int* workToDo, unsigned int n)
// {
// 	int tx = threadIdx.x + (blockDim.x*blockIdx.x);

// 	int local_workToDo = 0;
// 	if(tx < n)
// 	{
// 		if(Fa[tx]==1)	// If current node in in frontier
// 		{
// 			Fa[tx]=0;
// 			Xa[tx]=1;	// Mark current node as visited
// 			int N_edges = V[tx];
// 			int idx = indices[tx];
// 			for(int i=0; i < N_edges; i++)
// 			{
// 				int neigh = edges[idx + i];
// 				if(Xa[neigh] == 0)	// If the neighbous has not been visted, add it to the next frontier
// 				{
// 					Fa[neigh] = 1;
// 					local_workToDo = 1;
// 				}
// 			}
// 		}
// 	}
// 	*workToDo = local_workToDo;
// }

// /*
// Updates the visited  and map array with Xa
// */
// __global__
// void kernel_updateVisited(unsigned char* visited, unsigned char* Xa, unsigned int* map, int cluster, unsigned int n)
// {
// 	int tx = threadIdx.x + (blockDim.x * blockIdx.x);
// 	if(tx < n)
// 	{
// 		visited[tx] = (Xa[tx]==1);
// 		map[tx] = (Xa[tx]==1) ? cluster : map[tx];
// 	}
// }


// /*
// Performs DBScan by BFS on a graph with n veritces
// Args:
// 	n:	number of datapoints
// 	V:	an array of length n where number of edges at node i = V[i]
// 	indices:	an array of length n which stores the first index into the edges array
// 	edges:	an array of all the edges
// 	core:	a boolean array of length n indicating whether a point is a core point or not
// 	Fa:	Frontier array of length n for BFS (length n)
// 	Xa: Visted array of length n for BFS (length n)
// 	map:	Mapping of points to clusters (length n)
// 	visited:	boolean array of length n indicating if a point is visted
// 	workToDo:	A flag which indicates if more levels of BFS need to be done
// */
// __global__
// void kernel_parent_bfs(	int n, int* V, int* indices, int* edges,
// 						unsigned char* core, unsigned char *Fa, unsigned char* Xa,
// 						unsigned int* map, unsigned char* visited, int* workToDo)
// {
// 	unsigned int n_grid_dim = (n + (THREADS_PER_BLOCK - 1)) / THREADS_PER_BLOCK;
// 	kernel_init<<<n_grid_dim,THREADS_PER_BLOCK>>>(visited, n);
// 	cudaDeviceSynchronize();

// 	int cluster = 1;	
// 	for(int v = 0; v < n; v++)
// 	{
// 		if(core[v] == 1 && visited[v] == 0)
// 		{

// 			visited[v]=1;
// 			kernel_init<<<n_grid_dim,THREADS_PER_BLOCK>>>(Fa, n);	// Tracks the frontier of BFS search
// 			kernel_init<<<n_grid_dim,THREADS_PER_BLOCK>>>(Xa, n);	// Tracks the visited node in current BFS search
// 			cudaDeviceSynchronize();
// 			Fa[v] = 1;

// 			*workToDo = 1;
// 			while(*workToDo)
// 			{
// 				*workToDo = 0;
// 				kernel_bfs_child<<<n_grid_dim,THREADS_PER_BLOCK>>>(V, indices, edges, Fa, Xa, workToDo,n);
// 				cudaDeviceSynchronize();
// 			}

// 			kernel_updateVisited<<<n_grid_dim,THREADS_PER_BLOCK>>>(visited, Xa, map, cluster, n);
// 			cudaDeviceSynchronize();
// 			cluster++;
// 		}
// 	}
// }

float dbscanGPU(float* x, float* y, unsigned int* map, unsigned int n, int minPts, float R)
{
	struct timespec start_gpu, end_gpu;
	float msecs_gpu;
	clock_gettime(CLOCK_MONOTONIC, &start_gpu);

	// Create pointers for memory to be allocated on device
	int* d_V;	// Holds the number of edges for every vertex v
	// unsigned char* d_core;	// Boolean array. If true, the point is a core point
	// float* d_x;	// x coordinate of each point
	// float* d_y;	// x coordinate of each point
	
	// Allocate memory on device
	cudaMalloc((void**)&d_V, sizeof(int) * n);
	// cudaMalloc((void**)&d_core, sizeof(unsigned char) * n );
	// cudaMalloc((void**)&d_x, sizeof(float) * n);
	// cudaMalloc((void**)&d_y, sizeof(float) * n);
	// cudaMemcpy(d_x, x, sizeof(float) * n, cudaMemcpyHostToDevice);-
	// cudaMemcpy(d_y, y, sizeof(float) * n, cudaMemcpyHostToDevice);

	// // Bind d_x and d_y to x_tex and y_tex	
	// cudaBindTexture(NULL, x_tex, d_x, n * sizeof(float));
	// cudaBindTexture(NULL, y_tex, d_y, n * sizeof(float));


	// // Compute the number of edges for every vertex. Also assigns core points
	// unsigned int n_grid_dim = (n + (THREADS_PER_BLOCK - 1)) / THREADS_PER_BLOCK;
	// kernel_getV<<<n_grid_dim, THREADS_PER_BLOCK>>>(n, d_V, d_core, minPts, R); // Returns Array when V[i] = number of edges of vertice i

	// // Computes exclusive scan
	// int* d_indices;	// x coordinate of each point
	// cudaMalloc((void**) &d_indices, sizeof(int) * n);
	// thrust::exclusive_scan(thrust::device, d_V, d_V + n, d_indices, 0);

	// // Get the total number of edges
	// int numEdges = 0;
	// int v_last = 0;
	// int indices_last = 0;
	// cudaMemcpy(&v_last, d_V+n-1, sizeof(int), cudaMemcpyDeviceToHost);
	// cudaMemcpy(&indices_last, d_indices+n-1, sizeof(int), cudaMemcpyDeviceToHost);

	// // Allocate space for edges on GPU
	// numEdges = v_last + indices_last;
	// std::cout << numEdges << std::endl;
	// int* d_edges;
	// cudaMalloc((void**) &d_edges, sizeof(int)*numEdges);
	
	// kernel_getEdges<<<n_grid_dim,THREADS_PER_BLOCK>>>(n, d_indices, d_edges, R);
	
	// // Allocate space for map, Xa and Fa
	// unsigned int *d_map;
	// unsigned char *Fa, *Xa;
	// cudaMalloc((void**)&d_map, sizeof(unsigned int)*n);
	// cudaMalloc((void**)&Fa, sizeof(unsigned char)*n);
	// cudaMalloc((void**)&Xa, sizeof(unsigned char)*n);

	// // Allocate sapce for visited, boolean array
	// unsigned char* visited;
	// cudaMalloc((void**)&visited, sizeof(unsigned char)*n);
	
	// // Alocate space for workToDoFlag
	// int* d_workToDo;
	// cudaMalloc((void**)&d_workToDo, sizeof(int));
	
	// kernel_parent_bfs<<<1,1>>>(	n, d_V, d_indices, d_edges,
	// 							d_core, Fa, Xa,
	// 							d_map, visited, d_workToDo);
	
	// // Bring map back to CPU
	// cudaMemcpy(map, d_map, sizeof(unsigned int)*n, cudaMemcpyDeviceToHost);

	// cudaUnbindTexture(x_tex);
	// cudaUnbindTexture(y_tex);
	// cudaFree(d_indices);
	cudaFree(d_V);
	// cudaFree(d_x);
	// cudaFree(d_y);
	// cudaFree(d_core);
	// cudaFree(d_edges);
	// cudaFree(Xa);
	// cudaFree(Fa);
	// cudaFree(d_workToDo);
	// cudaFree(visited);
	// cudaFree(d_map);

	clock_gettime(CLOCK_MONOTONIC, &end_gpu);
	msecs_gpu = 1000.0 * (end_gpu.tv_sec - start_gpu.tv_sec) + (end_gpu.tv_nsec - start_gpu.tv_nsec)/1000000.0;

	return msecs_gpu;

}