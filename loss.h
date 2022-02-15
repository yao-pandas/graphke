#ifndef _LOSS_H_
#define _LOSS_H_

#include "gpu_score.h"

//--------------------------------------------------//
__host__ float computeLoss(Embeds *embeds, Triplet *triplets, float *scores, int num_triplets);

//--------------------------------------------------//
__global__ void kernelComputeSum(float *scores, int dim);

#endif