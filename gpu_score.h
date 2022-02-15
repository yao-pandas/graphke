#ifndef _GPU_SCORE_H_
#define _GPU_SCORE_H_

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
//#include <math.h>
#include <time.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "user_interface.h"
#include "basic_settings.h"
//#include "gpu_sample.h"
#include "timer.h"
#include "hash_graph.h"

// 将cache中的元素值加到target中来，使用单个block
#define ScoreAllSingleBlock(target, cache, dim) {\
	int j_ = threadIdx.x; \
	for(int p_=j_+blockDim.x; p_<dim; p_+=blockDim.x){ \
		cache[j_] += cache[p_]; \
	} \
	__syncthreads(); \
	for(int sz_=blockDim.x/2; sz_>0; sz_=sz_>>1){ \
		if(j_<sz_) \
			cache[j_] += cache[j_+sz_];\
		__syncthreads(); \
	} \
	target = cache[0]; \
}

// 功能: 用于批量计算范数
//--------------------------------------------------//
__global__ void kernelNorm(float *dev_vectors, int dim, int size, float *dev_norms, int norm_type);

// 功能: 对size个长度为dim的向量进行批量归一化
//--------------------------------------------------//
__global__ void kernelNormalize(float *dev_vectors, int dim, int size, float *dev_norms);

// 功能: 包装kernelNorm函数用于测试
//--------------------------------------------------//
__host__ void gpuNorm(float *dev_vectors, int dim, int size, float *dev_norms, int norm_type);

// 功能: 对于显存中的embeds进行归一化
//--------------------------------------------------//
__host__ void gpuNormalize(Embeds *embeds, int norm_type);

// 功能: 将长度为dim的size个向量的范数合并
//--------------------------------------------------//
__host__ void gpuMergeNorms(cublasHandle_t handle, float *dev_norms, int dim, int size, int norm_type);

// 功能: 在主机中调用gpu代码
// 注意: 这里的分值和导数要考虑f中的范数
//--------------------------------------------------//
__host__ void gpuScore(Embeds *dev_embeds, Triplet *dev_triplets, float *dev_norms, \
	float *dev_gh, float *dev_gr, float *dev_gt, int size, int dim, int norm_type);

// 功能 	在GPU中创建一个cache
// 参数 	
// dim_ent 	实体的维度
// dim_rel 	关系的维度
// size 	dim_ent由多少个子分量拼接而成
//--------------------------------------------------//
__host__ void createGradCacheInGPU(GradCache *cache, int dim_ent, int dim_rel, int size);

//--------------------------------------------------//
__host__ void freeGradCacheInGPU(GradCache *cache);

//--------------------------------------------------//
__global__ void kernelClearGradCache(GradCache *cache);

//功能:	将显存中的embdes清零
//--------------------------------------------------//
__host__ void clearEmbedsInGPU(Embeds *embdes);

//功能:	在显存中构造Embdes
//--------------------------------------------------//
__host__ void createEmbedsInGPU(Graph* graph, int dim_ent, int dim_rel, int base_ent, int base_rel, Embeds **embeds);

//功能:	在显存中构造Embeds
//--------------------------------------------------//
__host__ void createTripletsInGPU(int size, Triplet **dev_triplets);

//功能:	将嵌入数据从内存拷贝到显存
//--------------------------------------------------//
__host__ void copyTripletsCPU2GPU(Triplet *dev_triplets, Triplet *triplets, int size);

//功能:	将嵌入数据从内存拷贝到显存
//--------------------------------------------------//
__host__ void copyEmbedsCPU2GPU(Embeds *dev_embeds, Embeds *embeds);

//功能:	将嵌入数据从显存拷贝到内存
//--------------------------------------------------//
__host__ void copyEmbedsGPU2CPU(Embeds *embeds, Embeds *dev_embeds);

//功能:	释放显存中的Embdes
//--------------------------------------------------//
__host__ void freeEmbedsInGPU(Embeds *dev_embeds);

// 功能: 打印当前可以使用的显存数理
//--------------------------------------------------//
__host__ void printAvailableGPUMem();

// 功能:	清空Grads
//--------------------------------------------------//
__global__ void kernelClearEmbeds(Embeds *embeds);

// 功能:	将Grads中的值更新到Embeds中
//--------------------------------------------------//
__global__ void kernelMergeGradsToEmbeds(Embeds *embeds, Embeds *grads);

// 功能:	通过细化缓存来优化分值的计算
//--------------------------------------------------//
__global__ void kernelComputeGradsInSingleGPU(Embeds *embeds, Embeds *grads,\
	Triplet *triplets, Triplet *neg_triplets, \
	int size, int neg_deg, int sigma);

// 功能:	负采样为0的时候更新梯度
//--------------------------------------------------//
__global__ void kernelComputeGradsZeroNegSample(Embeds *embeds, Embeds *grads,\
	Triplet *triplets, Triplet *neg_triplets, \
	int size, int neg_deg, int sigma);

//--------------------------------------------------//
__global__ void kernelComputeGrads(Embeds *embeds, Embeds *grads,\
	Triplet *triplets, Triplet *neg_triplets, GradCache *cache, \
	int size, int neg_deg, int sigma);

// 功能:	通过细化缓存来优化分值的计算
//--------------------------------------------------//
__host__ void gpuComputeGradsInSingleGPU(Embeds *embeds, Embeds *grads,\
	Triplet *triplets, Triplet *neg_triplets, \
	int size, int neg_deg, int sigma);

// 功能:	在核函数中计算所有样本的分值（调用的时候需要动态缓存）
// 参数
//--------------------------------------------------//
extern __global__ void kernelComputeScores(Embeds *embeds, Triplet *triplets, \
	float *scores, int num_triplets);

#endif

