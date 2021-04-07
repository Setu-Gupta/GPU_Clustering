#include "GPUCluster.h"
#include<iostream>
/*
Performs DBSCAN  clustering on CPU
Args:
	x:	x coordinates of datapoints (n length float array)
	y:	y coordinates of datapoints (n length float array)
	map:	array to store cluster and datapoint mappings (n length unsigned int array)
	n:	number of datapoints
	minPts :  Minimum neighbours for a point to be classified as a "core " point
	R : min euclidean distance between 2 points for them to be called neighbours 
*/

float euclidean(float xi,float yi,float xj,float yj){
	float dist=pow((xi-xj),2)+pow((yi-yj),2);
	return sqrt(dist);
}

void exclusive_scan(int* V,int* indices,unsigned int n){
	indices[0]=0;
	for(int i=1;i<n;i++){
		indices[i]=indices[i-1]+V[i-1];
	}
}


void bfs(int* V,int* edges,int* indices,char* visited,unsigned int* map,int v,unsigned int n,int cluster){
	int* queue = (int*) malloc(sizeof(int)*n);
	queue[0]=v;
	int top=0;
	int bottom=1;
	while(top<bottom){
		int t=queue[top];
		map[t]=cluster;
		top++;
		for(int off=0;off<V[t];off++){
			int nbour=edges[indices[t]+off];
			if(visited[nbour]==0){
					//std::cout<<cluster<<std::endl;

				visited[nbour]=1;
				map[nbour]=cluster;
				//std::cout<<cluster<<std::endl;
				queue[bottom]=nbour;
				bottom++;
			}
		}
	}
	free(queue);



}

float dbscanCPU(float* x, float* y,	unsigned int* map,unsigned int n,int minPts,float R){
	struct timespec start_cpu, end_cpu;
	float msecs_cpu;
	clock_gettime(CLOCK_MONOTONIC, &start_cpu);


	for(int i=0;i<n;i++)map[i]=255; // initialise all points. 255 means noise
	int *V = (int* ) malloc(sizeof(int)*n);

// Calculating the number of neighbours of each point
	for(int i =0;i<n;i++){
		for(int j=0;j<n;j++){
			if(j!=i){	
				float dist=euclidean(x[i],y[i],x[j],y[j]);
				if(dist<=R){
					V[i]++; 
				}
			}
		}
	}
	// Array storing the index where the edges of a given element starts
	int* indices= (int*) malloc(sizeof(int)*n);
	// exclusie scan to populate the indices based on the values in V
	exclusive_scan(V,indices,n);

	// total edges will be the indices value of last element plus the neighbours it has.
	int* edges= (int* )malloc(sizeof(int)*(indices[n-1]+V[n-1]));
	int* core=(int*) malloc(sizeof(int)*n);
	

	// populating the edges array
	for(int i =0;i<n;i++){
		int count=0;
		for(int j=0;j<n;j++){
			if(j!=i){	
				float dist=euclidean(x[i],y[i],x[j],y[j]);
				if(dist<=R){
					edges[indices[i]+count]=j;
					count++;
				}
			}
		}
		if(count>=minPts)core[i]=1;
		else core[i]=0;
	}
	
	int cluster=1;
	char *visited = (char* ) malloc(sizeof(char)*n);
	for(int i=0;i<n;i++){
		visited[i]=0;
	}

	for(int v=0;v<n;v++){
			
		//bfs over all neighbours
		if(visited[v]==0 && core[v]==1){
			//std::cout<<cluster<<std::endl;
			bfs(V, edges, indices, visited,map, v, n,cluster);
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

