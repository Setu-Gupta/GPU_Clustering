#include "GPUCluster.h"
#include<iostream>
#include<stdio.h>
#include<cmath>
void moveColToVec(float* src,float* dest,int n,int K,int k_req){
    for(int i=0;i<n;i++){
        dest[i]=src[K*i+k_req];
    }
}
void transpose(float* A,float* At,int R,int C){
    // float* At=(float* )malloc(sizeof(float)*R*C);
    for(int i=0;i<R;i++){
        for(int j=0;j<C;j++){
            At[i+j*C] =A[j+C*i];
        }
    }
    // return At;
}

void matrix_multiply(float* A,float* B,int RA,int CA,int CB, float* dest){
    for(int i=0;i<RA;i++){
        for(int j=0;j<CB;j++){
            dest[j+i*CB]=0;
            for(int k=0;k<CA;k++){
            dest[j+i*CB]+=A[k+CA*i]*B[j+CB*k];
            // std::cout<<dest[j+i*CB]<<std::endl;
            }
            
        }
    }


}

void normalize(float* Pk,int k){
    float acc=0;
    for(int i=0;i<k;i++){
        acc+=Pk[i]*Pk[i];
    }
    acc=std::sqrt(acc);
    for(int i=0;i<k;i++)Pk[i]/=acc;

}
float norm(float* Pk,int k){
    float acc=0;
    for(int i=0;i<k;i++){
        acc+=Pk[i]*Pk[i];
    }
    acc=std::sqrt(acc); 
    return acc;
}
void subtract(float* A,float*B,int  RA,int CA,float*dest,float alpha){
    for(int i=0;i<RA;i++){
        for(int j=0;j<CA;j++){
            dest[j+CA*i]=(A[j+CA*i]-B[j+CA*i])*alpha;
        }
    }
}

void nipals(float* R,float* T,float* P,int J,int N,int K,float e){
    // float *Tk= (float*)malloc(sizeof(float)*N);
    // float *Pk= (float*)malloc(sizeof(float)*K);
    // float* Tpt= (float*)malloc(sizeof(float)*N*K);
    // for(int i=0;i<10;i++)std::cout<<R[i]<<std::endl;
    float Tk[N]= {0};
    float Pk[K]= {0};
    float Tpt[N*K]= {0};
    float Rt[K*N]={0};
    for(int k=0;k<2;k++){
        float lamda=0;
        moveColToVec(R,Tk,N,K,k);
        // std::cout<<"Printing"<<std::endl;
        // for(int i=0;i<10;i++){
        //     std::cout<<Tk[i]<<" "<<R[i*K+k]<<std::endl;
        // }
        for(int j=0;j<J;j++){
            // float * Rt=transpose(R,N,K);
            transpose(R,Rt,N,K);
            matrix_multiply(Rt,Tk,K,N,1,Pk);
            // free(Rt);
            normalize(Pk,k);
            matrix_multiply(R,Pk,N,K,1,Tk);
            float lamda_dash=norm(Tk,k);
            if(std::abs(lamda_dash-lamda)<=e)break;
            lamda=lamda_dash;
            // std::cout<<j<<" "<<k<<lamda<<std::endl;
        }
        for(int i=0;i<K;i++){
            P[i*K+k]=Pk[i];
            
        }
        for(int i=0;i<N;i++){
            T[i*K+k]=Tk[i];
            
        }
        // float* Pkt=transpose(Pk,K,1);
        float Pkt[1*K]={0};
        transpose(Pk,Pkt,K,1);
        matrix_multiply(Tk,Pkt,N,1,K,Tpt);
        subtract(R,Tpt,N,K,R,1e-3);
        // free(Pkt);

    }
    // free(Tk);
    // free(Pk);
    // free(Tpt);
}
void StandardScaler(float* X ,float *dest,int n,int DIM){
    // float* mean=(float*)malloc(sizeof(float)*DIM);
    float mean[DIM]={0};
    for(int i=0;i<DIM;i++){
        float temp_mean=0;
        for(int j=0;j<n;j++){
            temp_mean+=X[j*DIM+i];
        }
        mean[i]=temp_mean/((float)n);
    }
    for(int i=0;i<DIM;i++){
        float temp_std=0;
        for(int j=0;j<n;j++){
            temp_std+=(X[j*DIM+i]-mean[i])*(X[j*DIM+i]-mean[i]);
        }
        temp_std/=(float)n;
        temp_std=std::sqrt(temp_std);
        for(int j=0;j<n;j++){
            dest[i+j*DIM]=(X[i+j*DIM]-mean[i])/temp_std;
        }
    }
    // free(mean);


}
int main(){
    int DIM =2;
    int n=10;
    int K = DIM;
    // float* X=(float*)malloc(sizeof(float*)*n*DIM);
    // float* R=(float*)calloc(0,sizeof(float*)*n*DIM);
    // float* T=(float*)calloc(0,sizeof(float*)*n*DIM);
    float X[n*DIM]={0};
    float R[n*DIM]={0};
    float T[n*DIM]={0};
    float e=1e-5;
    // float* P=(float*)calloc(0,sizeof(float)*DIM*DIM);
    float P[DIM*DIM]={0};
    int J=1000;
    for(int i=0;i<n;i++){
        for(int j=0;j<DIM;j++){
            X[i*DIM+j]=(float)(i*DIM+j);
            // R[i*DIM+j]=i*DIM+j;
        }
    }
    StandardScaler(X,R,n,DIM);

    // for(int i=0;i<n;i++){
    //     for(int j=0;j<DIM;j++)std::cout<<R[j+DIM*i]<<std::endl;
    // }

    nipals(R,T,P,J,n,K,e);
    for(int i=0;i<n;i++){
        std::cout<<T[i*K+0]<<" "<<T[i*K+1]<<std::endl;
    }
    // free(X);
    // free(R);
    // free(P);
    // free(T);



}





