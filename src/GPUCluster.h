#ifndef GPU_CLUSTER_H
#define GPU_CLUSTER_H

#include <math.h>
#include <float.h>
#include <ctime>

float kmeansCPU(float *x, float *y, unsigned int *map, unsigned int k, unsigned int n, unsigned int num_iters);

#endif