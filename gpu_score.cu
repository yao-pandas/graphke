#include "gpu_score.h"

// 功能: 用于批量计算范数（经测试该函数比batchNorm更快）
// 参数
// dev_vectors 	长度为dim*size的向量
// dim 			子向量维度
// size 		子向量个数
// dev_norms 	长度为size存储每个子向量的范数
// norm_type 	范数类型
//--------------------------------------------------//
__global__ void kernelNorm(float *dev_vectors, int dim, int size, float *dev_norms, int norm_type){
	int index  = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	int k, base, idx;

	k = index;
	if(norm_type==1)
		while(k<size){
			dev_norms[k] = 0;
			base = k*dim; 
			for(int i=0; i<dim; i++){
				dev_norms[k] += dev_vectors[base+i];
			}
			k += stride;
		}
	else if(norm_type==2)
		while(k<size){
			dev_norms[k] = 0;
			base = k*dim; 
			for(int i=0; i<dim; i++){
				idx = base + i;
				dev_norms[k] += dev_vectors[idx]*dev_vectors[idx];
			}
			dev_norms[k] = sqrt(dev_norms[k]);
			k += stride;
		}
	else if(index==0){
		printf("The norm type is illegal: %d\n", norm_type);
	}
}

// 功能: 包装kernelNorm函数用于测试
//--------------------------------------------------//
__host__ void gpuNorm(float *dev_vectors, int dim, int size, float *dev_norms, int norm_type){
	kernelNorm<<<grid_size, block_size>>>(dev_vectors, dim, size, dev_norms, norm_type);
	gpuErrchk( cudaDeviceSynchronize() );
}

// 功能: 对size个长度为dim的向量进行批量归一化
//--------------------------------------------------//
__global__ void kernelNormalize(float *dev_vectors, int dim, int size, float *dev_norms){
	int index  = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	int k = index; 	
	float *ptr; 
	while(k<size){
		ptr = dev_vectors + k*dim;
		for(int i=0; i<dim; i++){
			ptr[i] /= dev_norms[k];
		}
		k += stride;
	}
	return;
}

// 功能: 对于显存中的embeds进行归一化
//--------------------------------------------------//
__host__ void gpuNormalize(Embeds *embeds, int norm_type){
	Embeds host_embeds;
	gpuErrchk( cudaMemcpy(&host_embeds, embeds, sizeof(Embeds), cudaMemcpyDeviceToHost) ); 

	float *embed_ents, *embed_rels, *norms;
	int num_ents, num_rels, dim_ent, dim_rel;
	embed_ents = host_embeds.embed_ents;
	embed_rels = host_embeds.embed_rels;
	num_ents = host_embeds.num_ents;
	num_rels = host_embeds.num_rels;
	dim_ent = host_embeds.dim_ent; 
	dim_rel = host_embeds.dim_rel;

	CudaMalloc(norms, (num_ents>num_rels ? num_ents : num_rels), float);

	kernelNorm<<<grid_size, block_size>>>(embed_ents, dim_ent, num_ents, norms, norm_type);
	kernelNormalize<<<grid_size, block_size>>>(embed_ents, dim_ent, num_ents, norms);

	kernelNorm<<<grid_size, block_size>>>(embed_rels, dim_rel, num_rels, norms, norm_type);
	kernelNormalize<<<grid_size, block_size>>>(embed_rels, dim_rel, num_rels, norms);
	gpuErrchk( cudaDeviceSynchronize() );

	cudaFree(norms);
}

// 功能: 打印当前可以使用的显存数理
//--------------------------------------------------//
__host__ void printAvailableGPUMem(){
	size_t free_gpu_mem, total_gpu_mem;
	cudaMemGetInfo(&free_gpu_mem, &total_gpu_mem);
	printf("Available gpu memory is: %fGB\n", (double)free_gpu_mem/(1024*1024*1024));
}

// 功能 	在GPU中创建一个cache
// 参数 	
// dim_ent 	实体的维度
// dim_rel 	关系的维度
// size 	dim_ent由多少个子分量拼接而成
//--------------------------------------------------//
__host__ void createGradCacheInGPU(GradCache *cache, int dim_ent, int dim_rel, int size){
	GradCache *tmp = Malloc(1, GradCache);
	tmp->dim_ent = dim_ent;
	tmp->dim_rel = dim_rel;
	tmp->size = size; 

	gpuErrchk( CudaMalloc(tmp->gh, dim_ent*grid_size, float) );
	gpuErrchk( CudaMalloc(tmp->gr, dim_rel*grid_size, float) );
	gpuErrchk( CudaMalloc(tmp->gt, dim_ent*grid_size, float) );

	gpuErrchk( CudaMalloc(tmp->ngh, dim_ent*grid_size, float) );
	gpuErrchk( CudaMalloc(tmp->ngr, dim_rel*grid_size, float) );
	gpuErrchk( CudaMalloc(tmp->ngt, dim_ent*grid_size, float) );

	gpuErrchk( CudaMalloc(tmp->scores, size*grid_size, float) );
	cudaMemcpy(cache, tmp, sizeof(GradCache), cudaMemcpyHostToDevice);
	free(tmp);
}

//--------------------------------------------------//
__host__ void freeGradCacheInGPU(GradCache *cache){
	cudaFree(cache->gh);
	cudaFree(cache->gr);
	cudaFree(cache->gt);
	cudaFree(cache->ngh);
	cudaFree(cache->ngr);
	cudaFree(cache->ngt);
	cudaFree(cache->scores);
}

//--------------------------------------------------//
__global__ void kernelClearGradCache(GradCache *cache){
	int index  = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	int i = index;
	int dim_ent = grid_size*cache->dim_ent;
	int dim_rel = grid_size*cache->dim_rel; 
	int size = grid_size*cache->size;
	while(i<dim_ent){
		(cache->gh)[i] = 0;
		(cache->gt)[i] = 0;
		(cache->ngh)[i] = 0;
		(cache->ngt)[i] = 0;	
		i += stride; 		
	}
	__threadfence();

	i = index;
	while(i<dim_rel){
		(cache->gr)[i] = 0;
		(cache->ngr)[i] = 0;
		i += stride;
	}
	__threadfence();

	i = index;
	while(i<size){
		cache->scores[i] = 0; 
		i += stride; 
	}
	__threadfence(); 
}

// 功能 	在显存中构造Embeds
// 参数 	
// graph 	图结构
// dim_ent	实体的嵌入维度
// dim_rel 	关系的嵌入维度
// embeds 	显存中的嵌入结构体
//--------------------------------------------------//
__host__ void createEmbedsInGPU(Graph* graph, int dim_ent, int dim_rel, int base_ent, int base_rel, Embeds **embeds){
	int num_ents = graph->num_ents;
	int num_rels = graph->num_rels;
	cudaError_t err = cudaSuccess; 
	err = (err==cudaSuccess) ? CudaMalloc(*embeds, 1, Embeds) : err;

	float *embed_ents, *embed_rels;
	gpuErrchk( CudaMalloc(embed_ents, dim_ent*num_ents, float) );
	gpuErrchk( CudaMalloc(embed_rels, dim_rel*num_rels, float) );

	Embeds *tmp = Malloc(1, Embeds);
	tmp->dim_ent = dim_ent;
	tmp->dim_rel = dim_rel;
	tmp->base_ent = base_ent;
	tmp->base_rel = base_rel; 
	tmp->num_ents = num_ents;
	tmp->num_rels = num_rels;
	tmp->embed_ents = embed_ents;
	tmp->embed_rels = embed_rels;

	gpuErrchk( cudaMemcpy(*embeds, tmp, sizeof(Embeds), cudaMemcpyHostToDevice) );
}

//功能:	释放显存中的Embeds
//--------------------------------------------------//
__host__ void freeEmbedsInGPU(Embeds *embeds){
	Embeds *tmp = Malloc(1, Embeds);
	gpuErrchk( cudaMemcpy(tmp, embeds, sizeof(Embeds), cudaMemcpyDeviceToHost) );
	cudaFree(tmp->embed_ents);
	cudaFree(tmp->embed_rels);
	cudaFree(embeds);
	free(tmp);
}

//功能:	在显存中size个triplets
//--------------------------------------------------//
__host__ void createTripletsInGPU(int size, Triplet **dev_triplets){
	gpuErrchk( CudaMalloc(*dev_triplets, size, Triplet) );
}

// 功能 	将嵌入数据从内存拷贝到显存
//--------------------------------------------------//
__host__ void copyEmbedsCPU2GPU(Embeds *dev_embeds, Embeds *embeds){
	Embeds *tmp = Malloc(1, Embeds);
	gpuErrchk( cudaMemcpy(tmp, dev_embeds, sizeof(Embeds), cudaMemcpyDeviceToHost) );
	gpuErrchk( cudaMemcpy(tmp->embed_ents, embeds->embed_ents, sizeof(float)*embeds->dim_ent*embeds->num_ents, cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemcpy(tmp->embed_rels, embeds->embed_rels, sizeof(float)*embeds->dim_rel*embeds->num_rels, cudaMemcpyHostToDevice) );
	free(tmp);
}

// 功能 	将嵌入数据从显存拷贝到内存
//--------------------------------------------------//
__host__ void copyEmbedsGPU2CPU(Embeds *embeds, Embeds *dev_embeds){
	Embeds *tmp = Malloc(1, Embeds);
	gpuErrchk( cudaMemcpy(tmp, dev_embeds, sizeof(Embeds), cudaMemcpyDeviceToHost) );
	gpuErrchk( cudaMemcpy(embeds->embed_ents, tmp->embed_ents, sizeof(float)*embeds->dim_ent*embeds->num_ents, cudaMemcpyDeviceToHost) );
	gpuErrchk( cudaMemcpy(embeds->embed_rels, tmp->embed_rels, sizeof(float)*embeds->dim_rel*embeds->num_rels, cudaMemcpyDeviceToHost) );
	free(tmp);
}

// 功能 	将triplets从内存拷贝到显存
//--------------------------------------------------//
__host__ void copyTripletsCPU2GPU(Triplet *dev_triplets, Triplet *triplets, int size){
	gpuErrchk( cudaMemcpy(dev_triplets, triplets, sizeof(Triplet)*size, cudaMemcpyHostToDevice) );
}

// 功能 	将triplets从显存拷贝到内存
//--------------------------------------------------//
__host__ void copyTripletsGPU2CPU(Triplet *triplets, Triplet *dev_triplets, int size){
	gpuErrchk( cudaMemcpy(triplets, dev_triplets, sizeof(Triplet)*size, cudaMemcpyDeviceToHost) );
}

// 功能:	通过细化缓存来优化分值的计算
//--------------------------------------------------//
__host__ void gpuComputeGradsInSingleGPU(Embeds *embeds, Embeds *grads,\
	Triplet *triplets, Triplet *neg_triplets, \
	int size, int neg_deg, int sigma){

	kernelClearEmbeds<<<grid_size, block_size>>>(grads);

	kernelComputeGradsZeroNegSample<<<grid_size, block_size>>>(embeds, grads,\
		triplets, neg_triplets, \
		size, neg_deg, sigma);
	//kernelComputeGradsInSingleGPU<<<grid_size, block_size>>>(embeds, grads,\
		triplets, neg_triplets, \
		size, neg_deg, sigma);
	kernelMergeGradsToEmbeds<<<grid_size, block_size>>>(embeds, grads);

	// 引入cache计算分值
	//kernelComputeGrads<<<grid_size, block_size>>>(embeds, grads,\
		triplets, neg_triplets, cache, \
		size, neg_deg, sigma);
}

// 功能:	通过细化缓存来优化分值的计算
// 参数
// size 	总共的triplets的长度
// neg_deg 	每个正样本对应的负样本个数
//--------------------------------------------------//
__global__ void kernelComputeGrads(Embeds *embeds, Embeds *grads,\
	Triplet *triplets, Triplet *neg_triplets, GradCache *cache, \
	int size, int neg_deg, int sigma){

	int h, r, t, nh, nt;

	int i, j, base0, base1;
	float *addr_h, *addr_r, *addr_t, *grad_h, *grad_r, *grad_t, *tmp_addr_h, *tmp_addr_t;
	float *cache_gh, *cache_gr, *cache_gt, *cache_ngh, *cache_ngr, *cache_ngt, *cache_scores; 

	int dim_ent = embeds->dim_ent;
	int dim_rel = embeds->dim_rel; 
	int base_ent = embeds->base_ent;
	int base_rel = embeds->base_rel;
	int dim = dim_ent/base_ent; 

	// 加载地址
	float *embed_ents, *embed_rels, *grad_ents, *grad_rels;
	embed_ents = embeds->embed_ents;
	embed_rels = embeds->embed_rels;
	grad_ents = grads->embed_ents;
	grad_rels = grads->embed_rels;

	// 用于存储临时样本
	float positive_score, negative_score; 

	// 每个block负责1个正样本和其对应的所有负样本
	i = blockIdx.x;
	while(i<size){
		h = triplets[i].h;
		r = triplets[i].r;
		t = triplets[i].t;

		addr_h = embed_ents + h*dim_ent;
		addr_r = embed_rels + r*dim_rel;
		addr_t = embed_ents + t*dim_ent;

		grad_h = grad_ents + h*dim_ent;
		grad_r = grad_rels + r*dim_rel;
		grad_t = grad_ents + t*dim_ent;
				
		// 利用缓存计算正分
		j = threadIdx.x;
		cache_scores = cache->scores + i*dim;
		cache_gh = cache->gh + i*dim_ent;
		cache_gr = cache->gr + i*dim_rel;
		cache_gt = cache->gt + i*dim_ent;
		while(j<dim){
			ScoreFunc(addr_h+j*base_ent, addr_r+j*base_rel, addr_t+j*base_ent, cache_scores+j, \
				cache_gh+j*base_ent, cache_gr+j*base_rel, cache_gt+j*base_ent, sigma, true);
			j += blockDim.x;
		}
		__syncthreads();

		// 计算正样本总分 (这是一个宏，不是函数，单block规约)
		ScoreAllSingleBlock(positive_score, cache_scores, dim)
		__syncthreads();

		//用于统计正样本要被累加多少次
		int count = 0;

		// 利用缓存计算负分
		base0 = i*neg_deg;
		for(int k=0; k<neg_deg; k++){
			base1 = base0 + k;
			nh = neg_triplets[base1].h;
			nt = neg_triplets[base1].t;

			cache_ngh = cache->ngh + i*dim_ent;
			cache_ngr = cache->ngr + i*dim_rel;
			cache_ngt = cache->ngt + i*dim_ent;

			if(nh!=h){
				tmp_addr_h = embed_ents + nh*dim_ent;
				j = threadIdx.x;
				while(j<dim){
					ScoreFunc(tmp_addr_h+j*base_ent, addr_r+j*base_rel, addr_t+j*base_ent, cache_scores+j, \
						cache_ngh+j*base_ent, cache_ngr+j*base_rel, cache_ngt+j*base_ent, sigma, true);
					j += blockDim.x;
				}
			}
			else if(nt!=t){
				tmp_addr_t = embed_ents + nt*dim_ent;
				j = threadIdx.x;
				while(j<dim){
					ScoreFunc(addr_h+j, addr_r+j, tmp_addr_t+j, cache_scores+j, \
						cache_ngh+j, cache_ngr+j, cache_ngt+j, sigma, true);
					j += blockDim.x;
				}
			}
			__syncthreads();

			// 计算负样本的总分(这是一个宏，不是函数)
			ScoreAllSingleBlock(negative_score, cache_scores, dim)
			__syncthreads();

			// 将负样本的梯度更新到缓存和grads中
			float coef = GAMMA+positive_score-negative_score;
			if(coef>0){
				count += 1;				//为正样本梯度累积次数
				if(nh!=h){
					j = threadIdx.x;
					while(j<dim_ent){
						grad_ents[nh*dim_ent+j] -= cache_ngh[j];
						grad_t[j] -= cache_ngt[j];
						j += blockDim.x;
					}
					j = threadIdx.x;
					while(j<dim_rel){
						grad_r[j] -= cache_ngr[j]; 
						j += blockDim.x; 
					}

				}
				else if(nt!=t){
					j = threadIdx.x;
					while(j<dim_ent){
						grad_h[j] -= cache_ngh[j];
						grad_ents[nt*dim_ent+j] -= cache_ngh[j];
						j += blockDim.x;						
					}
					j = threadIdx.x;
					while(j<dim_rel){
						grad_r[j] -= cache_ngr[j];
						j += blockDim.x;						
					}
				}
			}
		}

		//更新正样本到grads
		j = threadIdx.x;
		while(j<dim_ent){
			grad_h[j] += count*cache_gh[j];
			grad_t[j] += count*cache_gt[j];
			j += blockDim.x; 
		}
		__syncthreads();
		j = threadIdx.x;
		while(j<dim_rel){
			grad_r[j] += count*cache_gr[j];
			j += blockDim.x; 
		}
		__syncthreads();

		// 计算下一个批次的梯度值
		i += gridDim.x;
	}
}

// 功能:	通过细化缓存来优化分值的计算
//--------------------------------------------------//
__global__ void kernelComputeGradsInSingleGPU(Embeds *embeds, Embeds *grads,\
	Triplet *triplets, Triplet *neg_triplets, \
	int size, int neg_deg, int sigma){

	__shared__ float cache_h[DIM_ENT], cache_r[DIM_REL], cache_t[DIM_ENT];		
	__shared__ float cache_gh0[DIM_ENT], cache_gr0[DIM_REL], cache_gt0[DIM_ENT];	//缓存原始的导数向量
	__shared__ float cache_gh[DIM_ENT], cache_gr[DIM_REL], cache_gt[DIM_ENT];		//缓存累积的导数向量
	__shared__ float cache_ngh[DIM_ENT], cache_ngr[DIM_REL], cache_ngt[DIM_ENT];
	__shared__ float cache_scores[DIM];
	int h, r, t, nh, nt;

	int i, j, base0, base1;
	float *addr_h, *addr_r, *addr_t, *grad_h, *grad_r, *grad_t;

	int dim_ent = embeds->dim_ent;
	int dim_rel = embeds->dim_rel; 
	int base_ent = embeds->base_ent;
	int base_rel = embeds->base_rel; 
	int dim = dim_ent/base_ent;

	// 将缓存清零
	j = threadIdx.x;
	while(j<dim_ent){
		cache_gh[j] = 0;
		cache_gt[j] = 0;
		j += blockDim.x; 
	}
	__syncthreads();

	j = threadIdx.x;
	while(j<dim_rel){
		cache_gr[j] = 0;
		j += blockDim.x; 
	}
	__syncthreads();

	// 加载地址
	float *embed_ents, *embed_rels, *grad_ents, *grad_rels;
	embed_ents = embeds->embed_ents;
	embed_rels = embeds->embed_rels;
	grad_ents = grads->embed_ents;
	grad_rels = grads->embed_rels;

	// 用于存储临时样本
	float positive_score, negative_score; 

	// 每个block负责1个正样本和其对应的所有负样本
	i = blockIdx.x;
	while(i<size){
		h = triplets[i].h;
		r = triplets[i].r;
		t = triplets[i].t;

		addr_h = embed_ents + h*dim_ent;
		addr_r = embed_rels + r*dim_rel;
		addr_t = embed_ents + t*dim_ent;

		grad_h = grad_ents + h*dim_ent;
		grad_r = grad_rels + r*dim_rel;
		grad_t = grad_ents + t*dim_ent;
		
		// 加载向量到缓存
		j = threadIdx.x;
		while(j<dim_ent){
			cache_h[j] = addr_h[j];
			cache_t[j] = addr_t[j];
			j += blockDim.x;
		}
		__syncthreads();

		j = threadIdx.x;
		while(j<dim_rel){
			cache_r[j] = addr_r[j];
			j += blockDim.x;
		}
		__syncthreads();
		
		// 利用缓存计算正分
		j = threadIdx.x;
		base0 = i*dim; 
		while(j<dim){
			ScoreFunc(cache_h+j*base_ent, cache_r+j*base_rel, cache_t+j*base_ent, cache_scores+j, \
				cache_gh0+j*base_ent, cache_gr0+j*base_rel, cache_gt0+j*base_ent, sigma, true);
			j += blockDim.x;
		}
		__syncthreads();

		// 计算正样本总分 (这是一个宏，不是函数，单block规约)
		ScoreAllSingleBlock(positive_score, cache_scores, dim)
		__syncthreads();

		//用于统计正样本要被累加多少次
		int count = 0;

		// 利用缓存计算负分
		base0 = i*neg_deg;
		for(int k=0; k<neg_deg; k++){
			base1 = base0 + k;
			nh = neg_triplets[base1].h;
			nt = neg_triplets[base1].t;

			if(nh!=h){
				addr_h = embed_ents + nh*dim_ent;
				j = threadIdx.x;
				while(j<dim){
					ScoreFunc(addr_h+j*base_ent, cache_r+j*base_rel, cache_t+j*base_ent, cache_scores+j, \
						cache_ngh+j*base_ent, cache_ngr+j*base_rel, cache_ngt+j*base_ent, sigma, true);
					j += blockDim.x;
				}
			}
			else if(nt!=t){
				addr_t = embed_ents + nt*dim;
				j = threadIdx.x;
				while(j<dim){
					ScoreFunc(cache_h+j*base_ent, cache_r+j*base_rel, addr_t+j*base_ent, cache_scores+j, \
						cache_ngh+j*base_ent, cache_ngr+j*base_rel, cache_ngt+j*base_ent, sigma, true);
					j += blockDim.x;
				}
			}
			__syncthreads();

			// 计算负样本的总分(这是一个宏，不是函数)
			ScoreAllSingleBlock(negative_score, cache_scores, dim)
			__syncthreads();

			// 将负样本的梯度更新到缓存和grads中
			float coef = GAMMA+positive_score-negative_score;
			//if(threadIdx.x==0)
			//	printf("*****%f %f-----\n", positive_score, negative_score);
			//printf("%f\n", coef);
			if(coef>0){
				count += 1;				//为正样本梯度累积次数
				if(nh!=h){
					j = threadIdx.x;
					while(j<dim_ent){
						grad_ents[nh*dim_ent+j] -= (-1)*STEP_SIZE*cache_ngh[j];
						cache_gt[j] -= cache_ngt[j];
						//printf("%f %f %f$$", coef, cache_ngh[j], cache_ngt[j]);
						j += blockDim.x;
					}
					j = threadIdx.x;
					while(j<dim_rel){
						cache_gr[j] -= cache_ngr[j];
						j += blockDim.x;
					}
				}
				else if(nt!=t){
					j = threadIdx.x;
					while(j<dim_ent){
						cache_gh[j] -= cache_ngh[j];
						grad_ents[nt*dim_ent+j] -= (-1)*STEP_SIZE*cache_ngt[j];
						j += blockDim.x;						
					}
					j = threadIdx.x;
					while(j<dim_rel){
						cache_gr[j] -= cache_ngr[j];
						j += blockDim.x;						
					}
				}
			}
		}
		__syncthreads();

		//更新正样本到grads
		j = threadIdx.x;
		while(j<dim_ent){
			grad_h[j] += count*cache_gh0[j];
			grad_t[j] += count*cache_gt0[j];
			grad_h[j] += cache_gh[j];
			grad_t[j] += cache_gt[j];

			//printf("#%f %f %f %f@\n", cache_gh0[j], cache_gt0[j], cache_gh[j], cache_gt[j]);
			j += blockDim.x; 
		}
		__syncthreads();
		j = threadIdx.x;
		while(j<dim_rel){
			grad_r[j] += count*cache_gr0[j];
			grad_r[j] += cache_gr[j];
			j += blockDim.x; 
		}
		__syncthreads();

		// 计算下一个批次的梯度值
		i += gridDim.x;
	}
}

// 功能:	负采样为0的时候更新梯度
//--------------------------------------------------//
__global__ void kernelComputeGradsZeroNegSample(Embeds *embeds, Embeds *grads,\
	Triplet *triplets, Triplet *neg_triplets, \
	int size, int neg_deg, int sigma){

	//__shared__ float cache_h[DIM_ENT], cache_r[DIM_REL], cache_t[DIM_ENT];		
	__shared__ float cache_gh0[DIM_ENT], cache_gr0[DIM_REL], cache_gt0[DIM_ENT];	//缓存原始的导数向量
	//__shared__ float cache_gh[DIM_ENT], cache_gr[DIM_REL], cache_gt[DIM_ENT];		//缓存累积的导数向量
	//__shared__ float cache_ngh[DIM_ENT], cache_ngr[DIM_REL], cache_ngt[DIM_ENT];
	__shared__ float cache_scores[DIM];
	int i, j, h, r, t;

	float *addr_h, *addr_r, *addr_t, *grad_h, *grad_r, *grad_t;

	int dim_ent = embeds->dim_ent;
	int dim_rel = embeds->dim_rel; 
	int base_ent = embeds->base_ent;
	int base_rel = embeds->base_rel; 
	int dim = dim_ent/base_ent;

	// 加载地址
	float *embed_ents, *embed_rels, *grad_ents, *grad_rels;
	embed_ents = embeds->embed_ents;
	embed_rels = embeds->embed_rels;
	grad_ents = grads->embed_ents;
	grad_rels = grads->embed_rels;

	// 每个block负责1个正样本和其对应的所有负样本
	i = blockIdx.x;
	while(i<size){
		h = triplets[i].h;
		r = triplets[i].r;
		t = triplets[i].t;

		addr_h = embed_ents + h*dim_ent;
		addr_r = embed_rels + r*dim_rel;
		addr_t = embed_ents + t*dim_ent;

		grad_h = grad_ents + h*dim_ent;
		grad_r = grad_rels + r*dim_rel;
		grad_t = grad_ents + t*dim_ent;
		
		// 利用缓存计算正分
		j = threadIdx.x;
		while(j<dim){
			ScoreFunc(addr_h+j*base_ent, addr_r+j*base_rel, addr_t+j*base_ent, cache_scores+j, \
				cache_gh0+j*base_ent, cache_gr0+j*base_rel, cache_gt0+j*base_ent, sigma, true);
			j += blockDim.x;
		}
		__syncthreads();

		//更新正样本到grads
		j = threadIdx.x;
		while(j<dim_ent){
			grad_h[j] += cache_gh0[j];
			grad_t[j] += cache_gt0[j];
			j += blockDim.x; 
		}
		__syncthreads();
		j = threadIdx.x;
		while(j<dim_rel){
			grad_r[j] += cache_gr0[j];
			j += blockDim.x; 
		}
		__syncthreads();

		// 计算下一个批次的梯度值
		i += gridDim.x;
	}
}

// 功能:	清空Grads
//--------------------------------------------------//
__global__ void kernelClearEmbeds(Embeds *embeds){
	int index  = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	int len_ent = embeds->dim_ent * embeds->num_ents;
	int len_rel = embeds->dim_rel * embeds->num_rels;

	int i = index;
	while(i<len_ent){
		embeds->embed_ents[i] = 0;
		i += stride;
	}

	i = index;
	while(i<len_rel){
		embeds->embed_rels[i] = 0;
		i += stride;
	}
}

// 功能:	将Grads中的值更新到Embeds中
//--------------------------------------------------//
__global__ void kernelMergeGradsToEmbeds(Embeds *embeds, Embeds *grads){
	float *embed, *grad; 
	int num_ents = embeds->num_ents;
	int num_rels = embeds->num_rels;
	int dim_ent = embeds->dim_ent;
	int dim_rel = embeds->dim_rel;

	__shared__ bool flag;

	int i = blockIdx.x;
	while(i<num_ents){
		if(threadIdx.x==0)
			if (grads->embed_ents[i*dim_ent]==0)
				flag = false;
			else
				flag = true;

		__syncthreads();
		if(flag){
			int j = threadIdx.x;
			embed = embeds->embed_ents + i*dim_ent;
			grad = grads->embed_ents + i*dim_ent; 
			while(j<dim_ent){
				embed[j] -= STEP_SIZE*grad[j];
				j += blockDim.x;
			}
		}
		i += gridDim.x;
	}

	i = blockIdx.x;
	while(i<num_rels){
		if(threadIdx.x==0)
			if (grads->embed_rels[i*dim_rel]==0)
				flag = false;
			else 
				flag = true;

		__syncthreads();
		if(flag){
			int j = threadIdx.x;
			embed = embeds->embed_rels + i*dim_rel;
			grad = grads->embed_rels + i*dim_rel; 
			while(j<dim_rel){
				embed[j] -= STEP_SIZE*grad[j];
				j += blockDim.x;
			}
		}
		i += gridDim.x;
	}
}

// 功能:	在核函数中计算所有样本的分值（调用的时候需要动态缓存）
// 参数
// 
//--------------------------------------------------//
__global__ void kernelComputeScores(Embeds *embeds, Triplet *triplets, \
	float *scores, int num_triplets){

	float *embed_ents, *embed_rels, local_score;
	embed_ents = embeds->embed_ents;
	embed_rels = embeds->embed_rels; 

	int dim_ent = embeds->dim_ent;  
	int dim_rel = embeds->dim_rel;
	int base_ent = embeds->base_ent; 
	int base_rel = embeds->base_rel; 
	int dim = dim_ent/base_ent; 

	int h, r, t; 
	float *addr_h, *addr_r, *addr_t;

	float *null; 
	extern __shared__ float cache_scores[]; 

	// 每个block处理一组(h,r,t)
	int i = blockIdx.x; 
	while(i<num_triplets){
		h = triplets[i].h; 
		r = triplets[i].r; 
		t = triplets[i].t; 
		if(r!=-1){
			addr_h = embed_ents+h*dim_ent;
			addr_r = embed_rels+r*dim_rel;
			addr_t = embed_ents+t*dim_ent;

			// 计算样本分值
			int j = threadIdx.x; 
			while(j<dim){
				ScoreFunc(addr_h+j*base_ent, addr_r+j*base_rel, addr_t+j*base_ent, cache_scores+j, \
					null, null, null, 1, false);
				j += blockDim.x;
			}
			__syncthreads();

			ScoreAllSingleBlock(local_score, cache_scores, dim)
			if(threadIdx.x==0){
				scores[i] = local_score; 
			}
			__syncthreads();
		}
		else{
			if(threadIdx.x==0)
				scores[i] = -1.0; 
			__syncthreads();
		}

		i += gridDim.x; 
	}
}