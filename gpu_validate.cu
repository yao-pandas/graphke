#include "gpu_validate.h"

// 功能 	生成所有换头负样本
// 参数
// triplets 	长度为num_ents的生成的负样本
// triplet 		给定的正样本
// gpu_list 	用于检查换头之后是否在训练集中
// num_ents 	总共实体的个数
// hash 		与gpu_list同时使用
//--------------------------------------------------//
__global__ void kernelGenFilteredHeadTriplets(Triplet *triplets, Triplet *triplet, \
	TripletList *gpu_list, int num_ents, int hash){
	
	int index  = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	int r, t;
	r = triplet->r; 
	t = triplet->t; 

	// 换头
	int i = index; 
	while(i<num_ents){
		if(isInHashList(gpu_list, i, r, t, hash)){
			triplets[i].r = -1;
		}
		else{
			triplets[i].h = i;
			triplets[i].r = r;
			triplets[i].t = t; 
		}
		i += stride; 
	}
}

// 功能 	生成所有换尾样本
// 参数
// triplets 	长度为num_ents;
//--------------------------------------------------//
__global__ void kernelGenFilteredTailTriplets(Triplet *triplets, Triplet *triplet, \
	TripletList *gpu_list, int num_ents, int hash){
	
	int index  = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	int h, r;
	h = triplet->h; 
	r = triplet->r; 

	// 换尾
	int i = index; 
	while(i<num_ents){
		if(isInHashList(gpu_list, h, r, i, hash)){
			triplets[i].r = -1;
		}
		else{
			triplets[i].h = h;
			triplets[i].r = r;
			triplets[i].t = i; 
		}
		i += stride; 
	}
	__syncthreads();
}

// 功能 	将数据清零
//--------------------------------------------------//
__global__ void kernelClearRanks(int *ranks, int dim){
	int index  = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	int i = index; 

	while(i<dim){
		ranks[i] = 0; 
		i += stride; 
	}	
}

// 功能:	根据scores的值对相应正样本的rank进行更新
// 参数
// neg_scores 	得到的所有负样本的分数值，长度为score_size
// links 		指向具有相同(h,r)的所有正样本，长度为link_size
// ranks 		所有正样本的ranks
// scores 		所有正样本的分数值
// score_size 	负样本的分数值的个数
// link_idx 	对应于links中的第idx项
// cache		长度为score_size，用于缓存0-1值
//--------------------------------------------------//
__global__ void kernelUpdateRanks(float *neg_scores, TripletLink *links, int *ranks, float *scores, int *cache,\
	int score_size, int link_idx){

	int index  = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	int *link = links[link_idx].link; 
	int link_size = links[link_idx].size; 

	for(int j=0; j<link_size; j++){

		// 读入当前待比较的正样本分数值
		float score = scores[link[j]]; 

		// 写入缓存
		int i = index; 
		while(i<score_size){
			if((neg_scores[i]>0)&&(neg_scores[i]<score)){
				//DEBUG
				//printf("$# %d %f ", score_size, neg_scores[i]);
				cache[i] = 1;
			}
			else
				cache[i] = 0; 
			i += stride; 
		}
		__threadfence();

		// 计算缓存中的总和
		for(int k=1; k<score_size; k*=2){
			i = index;
			while(i+k<score_size){
				if(i%(2*k)==0)
					cache[i] += cache[i+k];
				i += stride; 
			}
			__threadfence(); 
		}

		// 利用缓存总的值更新rank
		if((threadIdx.x==0)&&(blockIdx.x==0)){
			atomicAdd(ranks+link[j], cache[0]);
			//printf("*%d %d %d\t", cache[0], link[j], ranks[link[j]]);
		}
		__threadfence();
	}
}

//--------------------------------------------------//
__global__ void kernelComputeHit(int *ranks, int *cache, int dim, float* hit, int level){
	int index  = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	int i = index; 
	while(i<dim){
		if(ranks[i]<level)
			cache[i]=1;
		else
			cache[i]=0;
		i += stride; 
	}
	__threadfence();

	// 计算缓存中的总和
	for(int k=1; k<dim; k*=2){
		i = index;
		while(i+k<dim){
			if(i%(2*k)==0)
				cache[i] += cache[i+k];
			i += stride; 
		}
		__threadfence(); 
	}

	if((blockIdx.x==0)&&(threadIdx.x==0)){
		*hit = (float)cache[0]/dim; 
	}
}

//--------------------------------------------------//
__global__ void kernelComputeMR(int *ranks, float *cache, int dim, float* mr){
	int index  = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	
	int i = index; 
	while(i<dim){
		cache[i] = ranks[i] + 1;
		i += stride; 
	}
	__threadfence();

	// 计算缓存中的总和
	for(int k=1; k<dim; k*=2){
		i = index;
		while(i+k<dim){
			if(i%(2*k)==0)
				cache[i] += cache[i+k];
			i += stride; 
		}
		__threadfence(); 
	}

	if((blockIdx.x==0)&&(threadIdx.x==0)){
		*mr = cache[0]/dim; 
	}
}

//--------------------------------------------------//
__global__ void kernelComputeMRR(int *ranks, float *cache, int dim, float* mrr){
	int index  = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	
	int i = index; 
	while(i<dim){
		cache[i] = 1.0/(ranks[i] + 1);
		i += stride; 
	}
	__threadfence();

	// 计算缓存中的总和
	for(int k=1; k<dim; k*=2){
		i = index;
		while(i+k<dim){
			if(i%(2*k)==0)
				cache[i] += cache[i+k];
			i += stride; 
		}
		__threadfence(); 
	}

	if((blockIdx.x==0)&&(threadIdx.x==0)){
		*mrr = cache[0]/dim; 
	}
}

// 功能 	对kernelComputeInidice进行包装从而进行测试
//--------------------------------------------------//
__host__ void gpuComputeIndices(int *ranks, int dim, float* hit1, float* hit3, float* hit10, float *mr, float *mrr){
	int *cache0;
	float *cache1;
	gpuErrchk( CudaMalloc(cache0, dim, int) ); 
	gpuErrchk( CudaMalloc(cache1, dim, float) ); 

	kernelComputeHit<<<grid_size, block_size>>>(ranks, cache0, dim, hit1, 1);
	kernelComputeHit<<<grid_size, block_size>>>(ranks, cache0, dim, hit3, 3);
	kernelComputeHit<<<grid_size, block_size>>>(ranks, cache0, dim, hit10, 10);
	kernelComputeMRR<<<grid_size, block_size>>>(ranks, cache1, dim, mrr);
	kernelComputeMR<<<grid_size, block_size>>>(ranks, cache1, dim, mr);

	cudaFree(cache0);
	cudaFree(cache1); 
}

// 功能 	在GPU中进行验证
// 参数
// scores 		用于临时存储triplets对应的分值，长度为num_triplets，位于GPU中
// neg_triplets 用于临时存储单个样本对应的所有负样本，长度为num_ents，位于GPU
// ranks 		对应于triplets中的每个的rank，长度为num_triplets，位于GPU
// hits 		长度为3，保存所有triplets的Hit1,3,10的值，位于GPU
// flags		用于标识正样本是否需要被计算
// dim 			注意这里的dim是dim_ent/base_ent的结果
//--------------------------------------------------//
__host__ void gpuFastValidate(Embeds *embeds, TripletList *gpu_list, \
	Triplet *triplets, float *scores, Triplet *neg_triplets, float *neg_scores,\
	TripletLink *head_link, TripletLink *tail_link, \
	int head_size, int tail_size, \
	int *ranks, float *hits, float *mr, float *mrr, \
	int dim, int num_ents, int num_rels, int num_triplets, int hash){

	kernelClearRanks<<<grid_size, block_size>>>(ranks, num_triplets);
	kernelComputeScores<<<grid_size, block_size, dim*sizeof(float)>>>(embeds, \
		triplets, scores, num_triplets);

	float *cpu_scores = Malloc(num_triplets, float); 
	/*cudaMemcpy(cpu_scores, scores, num_triplets*sizeof(float), cudaMemcpyDeviceToHost);
	for(int j=0; j<num_triplets; j++)//num_triplets
		printf("%f\t", cpu_scores[j]);
	printf("\n--------------------------------------------------------------------\n");*/

	int *cache;
	gpuErrchk( CudaMalloc(cache, num_ents, int) );

	printf("num_triplets, head_size: %d, %d\n", num_triplets, head_size);

	for(int i=0; i<head_size; i++){	//head_size
		kernelGenFilteredHeadTriplets<<<grid_size, block_size>>>(neg_triplets, triplets+i, \
			gpu_list, num_ents, hash);

		/*Triplet *cpu_triplets = Malloc(num_ents, Triplet);
		cudaMemcpy(cpu_triplets, triplets+i, sizeof(Triplet), cudaMemcpyDeviceToHost);
		printf("(%d, %d, %d)\n", cpu_triplets[0].h, cpu_triplets[0].r, cpu_triplets[0].t);

		cudaMemcpy(cpu_triplets, neg_triplets, num_ents*sizeof(Triplet), cudaMemcpyDeviceToHost);
		for(int j=0; j<num_ents; j++){
			printf("(%d, %d, %d)\n", cpu_triplets[j].h, cpu_triplets[j].r, cpu_triplets[j].t);
		}*/

		kernelComputeScores<<<grid_size, block_size, dim*sizeof(float)>>>(embeds, \
			neg_triplets, neg_scores, num_ents); 

		/*cudaDeviceSynchronize();
		cudaMemcpy(cpu_scores, neg_scores, num_ents*sizeof(float), cudaMemcpyDeviceToHost);
		for(int j=0; j<num_ents; j++)//
			printf("%f\t", cpu_scores[j]);*/

		kernelUpdateRanks<<<grid_size, block_size>>>(neg_scores, head_link, ranks, scores, cache,\
			num_ents, i);
	}
	
	for(int i=0; i<tail_size; i++){
		kernelGenFilteredTailTriplets<<<grid_size, block_size>>>(neg_triplets, triplets+i, \
			gpu_list, num_ents, hash);
		kernelComputeScores<<<grid_size, block_size, dim*sizeof(float)>>>(embeds, \
			neg_triplets, neg_scores, num_ents); 
		kernelUpdateRanks<<<grid_size, block_size>>>(neg_scores, tail_link, ranks, scores, cache,\
			num_ents, i);
	}

	// DEBUG
	/*
	int *cpu_ranks = Malloc(num_triplets, int);
	cudaMemcpy(cpu_ranks, ranks, num_triplets*sizeof(int), cudaMemcpyDeviceToHost);
	for(int j=0; j<num_triplets; j++)
		printf("%d ", cpu_ranks[j]);
	printf("\n--------------------------------------------------------------------\n");	*/
	
	gpuComputeIndices(ranks, num_triplets, hits, hits+1, hits+2, mr, mrr);
	cudaFree(cache);
}