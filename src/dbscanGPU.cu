#include<stdio.h>
#include<iostream>
#include<cuda_runtime.h>

#include <thrust/scan.h>
#include <thrust/execution_policy.h>

#include "GPUCluster.h"
__global__ void  kernel_getV(float* x,float* y,unsigned int n,int* V,unsigned char* core,int minPts,float R){
   
    int tx=threadIdx.x+blockDim.x*blockIdx.x;
    // printf("%f\n",R);
   if(tx<n){
    V[tx]=0;
    core[tx]=0;
    float px=x[tx];
    float py=y[tx];
    int count=0;
    for(int i=0;i<n;i++){
        float dist=sqrt(pow(px-x[i],2)+pow(py-y[i],2));
        if(dist<=R && dist>0){
            V[tx]=count+1;
            count++;
        }
    }
    if(count>=minPts){
        core[tx]=1;
    }
    }
}

__global__ void kernel_getEdges(float* x,float* y,unsigned int n,int* V,int* indices,int* edges,float R) {
    int tx=threadIdx.x+blockDim.x*blockIdx.x;
    if(tx<n){
        int count =0;
        float px=x[tx];
        float py=y[tx];
        for(int i=0;i<n;i++){
            float dist=sqrt(pow(px-x[i],2)+pow(py-y[i],2));
            // printf("%f\n",dist);
            if(dist<=R && dist>0){
                edges[indices[tx]+count]=i;
                count++;
            }

        }
    }
    
}
__global__ void kernel_init(unsigned char* Fa,int size){
    int tx=threadIdx.x+blockDim.x*blockIdx.x;
    if(tx<size){
        Fa[tx]=0;
    }
}


__global__ void kernel_bfs_child(int* V,int* indices,int* edges,unsigned char* Fa,unsigned char* Xa,int* workToDo,unsigned int n){
    int tx=threadIdx.x+blockDim.x*blockIdx.x;

    if(tx<n){
    if(Fa[tx]==1){

        Fa[tx]=0;Xa[tx]=1;
        for(int i=0;i<V[tx];i++){
            int neigh=edges[indices[tx]+i];
            if(Xa[neigh]==0){

                Fa[neigh]=1;
                *workToDo=1;
            }


            }
        }
    }
}
__global__ void kernel_updateVisited(unsigned char* visited,unsigned char* Xa,unsigned int* map,int cluster,unsigned int n){
    int tx=threadIdx.x + blockDim.x*blockIdx.x;
    if(tx<n){
        map[tx]=255;
        if(Xa[tx]==1){
            visited[tx]=1;
            map[tx]=cluster;
        }
    }
}
__global__ void kernel_parent_bfs(int n,int* V,int* indices,int* edges,unsigned char* core,unsigned char *Fa,unsigned char* Xa, unsigned int* map,unsigned char* visited,int* workToDo){
    
    kernel_init<<<max(1,n/1024),min(n,1024)>>>(visited,n);
    cudaDeviceSynchronize();

    int cluster=1;
    for(int v=0;v<n;v++){

        if(core[v]==1 && visited[v]==0){

            visited[v]=1;
            kernel_init<<<max(1,n/1024),min(n,1024)>>>(Fa,n);
            kernel_init<<<max(1,n/1024),min(n,1024)>>>(Xa,n);
            cudaDeviceSynchronize();
            Fa[v]=1;

            *workToDo=1;
            while(*workToDo==1){
                *workToDo=0;
                kernel_bfs_child<<<max(1,n/1024),min(n,1024)>>>(V,indices,edges,Fa,Xa,workToDo,n);
                cudaDeviceSynchronize();

            }
            kernel_updateVisited<<<max(1,n/1024),min(n,1024)>>>(visited,Xa,map,cluster,n);
            cudaDeviceSynchronize();
            cluster++;
        }
    }

}

float dbscanGPU(float* x, float* y,	unsigned int* map,unsigned int n,int minPts,float R){
    // float x[12]={1,2,3,4,5,6,7,8,20,21,22,89};
    // float y[12]={0,0,0,0,0,0,0,0,0,0,0,0};
    // float R=1;
    // int minPts=2;
    // int n=12;
    struct timespec start_gpu, end_gpu;
	float msecs_gpu;
	clock_gettime(CLOCK_MONOTONIC, &start_gpu);
    int* d_V;
    unsigned char* d_core;
    float* d_x;
    float* d_y;
    cudaMalloc((void**)&d_V, sizeof(int)*n);
    cudaMalloc((void**)&d_core, sizeof(unsigned char)*n );
    cudaMalloc((void**)&d_x, sizeof(float)*n);
    cudaMalloc((void**)&d_y, sizeof(float)*n);
    
    
    cudaMemcpy(d_x, x, sizeof(float)*n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, sizeof(float)*n, cudaMemcpyHostToDevice);
    
    kernel_getV<<<max(1,n/1024),min(n,1024)>>>(d_x,d_y,n,d_V,d_core,minPts,R);
    cudaDeviceSynchronize();
    int* V= (int* )malloc(sizeof(int)*n);
    cudaMemcpy(V, d_V, sizeof(int)*n, cudaMemcpyDeviceToHost);
    
    
    int* indices=(int* )malloc(sizeof(int)*n);
    thrust::exclusive_scan(thrust::host, V, V + n, indices, 0);

    int numEdges=V[n-1]+indices[n-1];
   
    int* edges= (int* ) malloc(sizeof(int)*(numEdges));
    int* d_indices;
    int* d_edges;
    cudaMalloc((void**)&d_indices, sizeof(int)*n);
    cudaMalloc((void**)&d_edges, sizeof(int)*numEdges);
    
    
    cudaMemcpy(d_indices, indices , sizeof(int)*n, cudaMemcpyHostToDevice);

    kernel_getEdges<<<max(1,n/1024),min(n,1024)>>>(d_x,d_y,n,d_V,d_indices,d_edges,R);
    cudaDeviceSynchronize();
    cudaMemcpy(edges, d_edges, sizeof(int)*numEdges, cudaMemcpyDeviceToHost);
    

    unsigned int* d_map;
    unsigned char* Fa;unsigned char* Xa;
    cudaMalloc((void**)&d_map, sizeof(unsigned int)*n);
    cudaMalloc((void**)&Fa, sizeof(unsigned char)*n);
    cudaMalloc((void**)&Xa, sizeof(unsigned char)*n);

    unsigned char* visited;
    cudaMalloc((void**)&visited, sizeof(unsigned char)*n);
    
    int* d_workToDo;
    cudaMalloc((void**)&d_workToDo,sizeof(int));
    
    kernel_parent_bfs<<<1,1>>>(n,d_V,d_indices,d_edges,d_core,Fa,Xa,d_map,visited,d_workToDo);
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &end_gpu);
	msecs_gpu = 1000.0 * (end_gpu.tv_sec - start_gpu.tv_sec) + (end_gpu.tv_nsec - start_gpu.tv_nsec)/1000000.0;
    // unsigned int* map=(unsigned int*)malloc(sizeof(unsigned int)*n);
    cudaMemcpy(map, d_map, sizeof(unsigned int)*n, cudaMemcpyDeviceToHost);

    cudaFree(d_edges);
    cudaFree(d_indices);
    cudaFree(d_V);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_core);
    cudaFree(Xa);
    cudaFree(Fa);
    cudaFree(visited);

    free(edges);
    free(indices);
    free(V);

    

    return msecs_gpu;

}