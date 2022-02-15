#include "gpu_sample.h"

// 功能 	给定GPU中的tripet,判定其是否在gpu_list中
//--------------------------------------------------//
extern __device__ int isInHashList(TripletList *gpu_list, int h, int r, int t, int hash){
	int key = ((h+1)*(t+1) + r)%hash;
	int size = gpu_list[key].size;
	if(size==0)
		return 0;

	Triplet *triplets = gpu_list[key].triplets; 	
	for(int i=0; i<size; i++)
		if((triplets[i].h==h)&& \
			(triplets[i].r==r)&& \
			(triplets[i].t==t)){
			return 1;
		}

	return 0; 	
}

// 生成0到count-1之间的随机数
//--------------------------------------------------//
__device__ int genRandomNumber(unsigned int *seed, int bound){
	*seed = (*seed*1103515245 + 12345)%(1<<31);
	return *seed%bound; 
}

// 功能 	在GPU中进行负采样
// 参数	
// triplets 			所有的三元组
// batch_triplets 		采样的正triplets
// batch_neg_triplets 	采样的负triplets
// batch_size 			批次大小
// neg_deg 				负采样大小
// seed 				种子
//--------------------------------------------------//
__global__ void kernelNegativeSampling(Triplet *triplets, TripletList *gpu_list, \
	Triplet *batch_triplets, Triplet *neg_triplets, \
	int num_ents, int num_rels, int num_triplets, \
	int batch_size, int neg_deg, int seed, int hash){

	// 抽取正样本
	int index  = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	int h, r, t; 

	// 基于种子随机抽取正样本
	unsigned int rnd = seed*(threadIdx.x+1)*(blockIdx.x+gridDim.x-1);
	int k = index;
	int idx; 

	// 抽取正样本
	while(k<batch_size){
		if(CYCLIC)
			idx = (seed + k)%num_triplets;
		else{
			rnd = (rnd*1103515245 + 12345)%(1<<31);
			idx = rnd%num_triplets; 
		}
		//idx = genRandomNumber(&rnd, num_triplets); 
		
		batch_triplets[k].h = triplets[idx].h;
		batch_triplets[k].r = triplets[idx].r;
		batch_triplets[k].t = triplets[idx].t;

		k += stride; 
	}

	// 为每个正样本抽取负样本
	k = index; 
	int count = batch_size * neg_deg; 
	while(k<count){
		int j = k/neg_deg;
		h = batch_triplets[j].h;
		r = batch_triplets[j].r;
		t = batch_triplets[j].t;

		// 换头
		idx = genRandomNumber(&rnd, num_ents);
		if(threadIdx.x%2==0){
			h = idx;
			while(isInHashList(gpu_list, h, r, t, hash)){
				h = genRandomNumber(&rnd, num_ents);
			}
		}
		else{
			t = idx;
			while(isInHashList(gpu_list, h, r, t, hash)){
				t = genRandomNumber(&rnd, num_ents);
			}
		}

		neg_triplets[k].h = h;
		neg_triplets[k].r = r; 
		neg_triplets[k].t = t;
		k += stride; 
	}
}

//--------------------------------------------------//
__host__ void gpuNegativeSampling(Triplet *triplets, TripletList *gpu_list, \
	Triplet *batch_triplets, Triplet *neg_triplets, \
	int num_ents, int num_rels, int num_triplets, \
	int batch_size, int neg_deg, int seed, int hash){

	kernelNegativeSampling<<<grid_size, block_size>>>(triplets, gpu_list, \
		batch_triplets, neg_triplets, \
		num_ents, num_rels, num_triplets, \
		batch_size, neg_deg, seed, hash);

	gpuErrchk( cudaDeviceSynchronize() );
}