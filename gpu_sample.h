#ifndef _GPU_SAMPLE_H_
#define _GPU_SAMPLE_H_

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <time.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "basic_settings.h"
#include "timer.h"
#include "hash_graph.h"

//--------------------------------------------------//
extern __device__ int isInHashList(TripletList *gpu_list, int h, int r, int t, int hash);

//--------------------------------------------------//
__device__ int genRandomNumber(unsigned int *seed, int bound);

//--------------------------------------------------//
__global__ void kernelNegativeSampling(Triplet *triplets, TripletList *gpu_list, \
	Triplet *batch_triplets, Triplet *neg_triplets, \
	int num_ents, int num_rels, int num_triplets, \
	int batch_size, int neg_deg, int seed, int hash);

//--------------------------------------------------//
__host__ void gpuNegativeSampling(Triplet *triplets, TripletList *gpu_list, \
	Triplet *batch_triplets, Triplet *neg_triplets, \
	int num_ents, int num_rels, int num_triplets, \
	int batch_size, int neg_deg, int seed, int hash);

#endif