#ifndef _GPU_VALIDATE_H_
#define _GPU_VALIDATE_H_

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "basic_settings.h"
#include "init_embeds.h"
#include "validate.h"
#include "gpu_sample.h"
#include "gpu_score.h"
#include "hash_graph.h"

//--------------------------------------------------//
__global__ void kernelGenFilteredHeadTriplets(Triplet *triplets, Triplet *triplet, \
	TripletList *gpu_list, int num_ents, int hash);

//--------------------------------------------------//
__global__ void kernelGenFilteredTailTriplets(Triplet *triplets, Triplet *triplet, \
	TripletList *gpu_list, int num_ents, int hash);

//--------------------------------------------------//
__global__ void kernelClearRanks(int *ranks, int dim);

//--------------------------------------------------//
__global__ void kernelUpdateRanks(float *neg_scores, TripletLink *links, int *ranks, float *scores, int *cache,\
	int score_size, int link_idx);

//--------------------------------------------------//
__global__ void kernelComputeHit(int *ranks, int *cache, int dim, float* hit, int level);

//--------------------------------------------------//
__global__ void kernelComputeMRR(int *ranks, float *cache, int dim, float* mrr);

//--------------------------------------------------//
__global__ void kernelComputeMR(int *ranks, float *cache, int dim, float* mr);

//--------------------------------------------------//
__host__ void gpuComputeIndices(int *ranks, int dim, float* hit1, float* hit3, float* hit10, float *mr, float *mrr);

//--------------------------------------------------//
__host__ void gpuFastValidate(Embeds *embeds, TripletList *gpu_list, \
	Triplet *triplets, float *scores, Triplet *neg_triplets, float *neg_scores,\
	TripletLink *head_link, TripletLink *tail_link, \
	int head_size, int tail_size, \
	int *ranks, float *hits, float *mr, float *mrr, \
	int dim, int num_ents, int num_rels, int num_triplets, int hash);

#endif
