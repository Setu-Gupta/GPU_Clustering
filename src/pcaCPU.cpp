#include "GPUCluster.h"
#include<iostream>
#include<stdio.h>
#include<cmath>

#define DIMENSION 2	// dimentions of input and output data
#define MAX_CHANGE 1e-5	// Change should be more than this to have more iterations

// NOTE: All matrices are stores in row major format

/*
Moves the k_req column of matrix of dimension NxK to a column of dimention nx1
*/
void moveColToVec(float* src, float* dest, int N, int K, int k_req)
{
	for(int i = 0; i < N; i++)
		dest[i] = src[K*i + k_req];
}

/*
Computes At as transpose of A. Dimension of A is RxC. Dimension At is CxR
*/
void transpose(float* A, float* At, int R, int C)
{
	for(int i = 0; i < R; i++)
		for(int j = 0; j < C; j++)
			At[j*R + i] = A[C*i + j];
}

/*
Computes dest = AB. Dimension of A is RAxCA. Dimension B is CAxCB. Dimensions of dest is RAxCB
*/
void matrix_multiply(float* A, float* B, int RA, int CA, int CB, float* dest)
{
	for(int i = 0; i < RA; i++)
	{
		for(int j = 0; j < CB; j++)
		{
			dest[i*CB + j]=0;
			for(int k = 0; k < CA; k++)
				dest[i*CB + j] += A[i*CA + k] * B[k*CB + j];
		}
	}
}

/*
Computes the L2 norm of vector Pk of dimentions kx1.
*/
float norm(float* Pk, int k)
{
	float acc = 0;
	for(int i = 0; i < k; i++)
		acc += Pk[i] * Pk[i];
	return std::sqrt(acc); 
}

/*
Computes the L2 norm of vector Pk of dimentions kx1.
Then divides Pk by its norm, making it a unit vector.
*/
void normalize(float* Pk, int k)
{
	float l2 = norm(Pk, k);

	for(int i = 0; i < k; i++)
		Pk[i] /= l2;
}

/*
Computes dest = alpha(A-B). Dimensions of all A, B and dest is RAxCA
*/
void subtract(float* A, float*B, int RA, int CA, float*dest, float alpha)
{
	for(int i = 0; i < RA; i++)
		for(int j = 0; j < CA; j++)
			dest[i*CA + j] = alpha * (A[i*CA + j] - B[i*CA + j]);
}

/*
Computes NIPALs PCA.
Dimensions of R is NxK
Dimensions of T is NxK
Dimensions of P is KxK
J is max number of iterations
e is the change below which we can stop iterating
*/
void nipals(float* R, float* T, float* P, int J, int N, int K, float e)
{
	float Tk[N * 1] = {0};
	float Pk[K * 1] = {0};
	float Tpkt[N * K] = {0};
	float Rt[K * N] = {0};
	
	for(int k = 0; k < 2; k++)
	{
		float lamda = 0;
		moveColToVec(R, Tk, N, K, k);

		for(int j = 0; j < J; j++)
		{
			transpose(R, Rt, N, K);
			matrix_multiply(Rt, Tk, K, N, 1, Pk);
			normalize(Pk, k);
			matrix_multiply(R, Pk, N, K, 1, Tk);
			float lamda_dash = norm(Tk, k);
			if(std::abs(lamda_dash - lamda) <= e)
				break;
			lamda = lamda_dash;
		}

		for(int i = 0; i < K; i++)
			P[i*K+k] = Pk[i];
		
		for(int i = 0; i < N; i++)
			T[i*K + k]=Tk[i];
			
		float Pkt[1 * K] = {0};
		transpose(Pk, Pkt, K, 1);
		matrix_multiply(Tk, Pkt, N, 1, K, Tpkt);
		subtract(R, Tpkt, N, K, R, 1e-3);
	}
}

/*
Standard-scales matrix X of dimesions NxDIM column wise.
For every column i, performs (0 <= i < DIM)
	x_ji = (x_ji - mean_i) / std_dev_i
Store the result in dest of dimensions NxDIM
*/
void StandardScaler(float* X , float *dest, int N, int DIM)
{
	float mean[1 * DIM] = {0};

	for(int i = 0; i < DIM; i++)
	{
		float temp_mean = 0;
		for(int j = 0; j < N; j++)
			temp_mean += X[j*DIM + i];

		mean[i] = temp_mean / (float)N;
	}

	for(int i = 0; i < DIM; i++)
	{
		float temp_std = 0;
		for(int j = 0; j < N; j++)
			temp_std += (X[j*DIM + i] - mean[i]) * (X[j*DIM + i] - mean[i]);

		temp_std /= (float)N;
		temp_std = std::sqrt(temp_std);

		for(int j = 0; j < N; j++)
			dest[j*DIM + i] =  (X[j*DIM + i] - mean[i]) / temp_std;
	}
}

/*
Performs NIPAL's PCA on CPU
Args:
	x:	x coordinates of datapoints (n length float array)
	y:	y coordinates of datapoints (n length float array)
	new_x:	transformed x coordinates of datapoints (n length float array)
	new_y:	treansformed y coordinates of datapoints (n length float array)
	n:	number of datapoints
	num_iters:	number of iteration to run NIPALs for
Returns: Time spent in computation in milliseconds
*/
float nipalsCPU(float* x, float* y,	float* new_x, float* new_y, unsigned int n, unsigned int num_iters)
{
	int N = n;
	int DIM = DIMENSION;
	int K = DIM;
	float X[N*DIM] = {0};
	float R[N*DIM] = {0};
	float T[N*DIM] = {0};
	float P[DIM * DIM] = {0};
	float e = MAX_CHANGE;

	int J = num_iters;

	for(int i = 0; i < N; i++)
	{
		X[i*K + 0] = x[i];
		X[i*K + 1] = y[i];
	}

	for(int i = 0; i < N; i++)
	{
		for(int j = 0; j < DIM; j++)
			X[i*DIM + j] = (float)(i*DIM + j);	// From outside
	}

	StandardScaler(X, R, N, DIM);

	nipals(R, T, P, J, N, K, e);
	for(int i = 0; i < N; i++)
	{
		new_x = T[i*K + 0];
		new_y = X[i*K + 1];
	}
	// std::cout << "[" << std::endl;
	// for(int i = 0; i < N; i++)
	// 	std::cout << "["<<T[i*K + 0] << ", " << T[i*K + 1] << "]," << std::endl;
	// std::cout << "]" << std::endl;
}