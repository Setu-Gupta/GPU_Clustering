#include<stdio.h>
#include<iostream>
#include<cuda_runtime.h>

#include <thrust/scan.h>
#include <thrust/execution_policy.h>

#include "GPUCluster.h"
__global__ void  kernel_getV(float* x,float* y,int n,int* V,int* core,int minPts,float R){
    int tx=threadIdx.x+blockDim.x*blockIdx.x;
    // printf("%f\n",R);
    V[tx]=0;
    core[tx]=0;
    float px=x[tx];
    float py=y[tx];
    int count=0;
    for(int i=0;i<tx;i++){
        float dist=sqrt(pow(px-x[i],2)+pow(py-y[i],2));
        // printf("%f , %f",x[i],y[i]);
        // printf("%f\n",dist);
        if(dist<=R && dist!=0){
            printf("%d\n",tx);
            V[tx]++;
            count++;
        }
    }
    if(count>minPts){
        core[tx]=1;
    }
}

__global__ void kernel_getEdges(float* x,float* y,int n,int* V,int* indices,int* edges,float R) {
    int tx=threadIdx.x+blockDim.x*blockIdx.x;
    int count =0;
    float px=x[tx];
    float py=y[tx];
    for(int i=0;i<n;i++){
        float dist=sqrt(pow(px-x[i],2)+pow(py-y[i],2));
        if(dist<=R && dist>0){
            edges[indices[tx]+count]=i;
            count++;
        }

    }
    
}

int main(){
    float x[10]={1,2,3,4,5,6,7,8,10,20};
    float y[10]={0,0,0,0,0,0,0,0,0,0};
    float R=1;
    int minPts=2;
    int n=10;
    int* d_V;
    int* d_core;
    float* d_x;
    float* d_y;
    cudaMalloc((void**)&d_V, sizeof(int)*n);
    cudaMalloc((void**)&d_core, sizeof(int)*n );
    cudaMalloc((void**)&d_x, sizeof(float)*n);
    cudaMalloc((void**)&d_y, sizeof(float)*n);
    
    
    cudaMemcpy(d_x, x, sizeof(float)*n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, sizeof(float)*n, cudaMemcpyHostToDevice);
    
    kernel_getV<<<max(1,n/1024),max(n,1024)>>>(d_x,d_y,n,d_V,d_core,minPts,R);
    cudaDeviceSynchronize();
    int* V= (int* )malloc(sizeof(int)*n);
    int* core= (int* )malloc(sizeof(int)*n);

    cudaMemcpy(V, d_V, sizeof(int)*n, cudaMemcpyDeviceToHost);
    cudaMemcpy(core, d_core, sizeof(int)*n, cudaMemcpyDeviceToHost);
    
    for(int i=0;i<n;i++){
        std::cout<<V[i]<<std::endl;
    }
    // int* indices=(int* )malloc(sizeof(int)*n);
    // thrust::exclusive_scan(thrust::host, V, V + n, indices, 0);

    // int numEdges=V[n-1]+indices[n-1];
    // std::cout<<numEdges<<std::endl;
    // int* edges= (int* ) malloc(sizeof(int)*(numEdges));
    // int* d_indices;
    // int* d_edges;
    // cudaMalloc((void**)&d_indices, sizeof(int)*n);
    // cudaMalloc((void**)&d_edges, sizeof(int)*n);
    
    
    // cudaMemcpy(d_indices, indices , sizeof(int)*n, cudaMemcpyHostToDevice);

    // kernel_getEdges<<<max(1,n/1024),max(n,1024)>>>(d_x,d_y,n,d_V,d_indices,d_edges,R);
    // cudaDeviceSynchronize();
    // cudaMemcpy(edges, d_edges, sizeof(int)*n, cudaMemcpyDeviceToHost);
    


    // cudaFree(d_edges);
    // cudaFree(d_indices);
    // cudaFree(d_V);
    // cudaFree(d_x);
    // cudaFree(d_y);
    // cudaFree(d_core);
    // free(core);
    // free(edges);
    // free(indices);
    // free(V);

    

    return 0;

}