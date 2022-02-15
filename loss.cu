#include "loss.h"

//--------------------------------------------------//
__host__ float computeLoss(Embeds *embeds, Triplet *triplets, float *scores, int num_triplets){
	kernelComputeScores<<<grid_size, block_size, DIM*sizeof(float)>>>(embeds, triplets, scores, num_triplets);	
	kernelComputeSum<<<grid_size, block_size>>>(scores, num_triplets);
	float loss; 
	cudaMemcpy(&loss, scores, sizeof(float), cudaMemcpyDeviceToHost);
	return loss; 
}

//--------------------------------------------------//
__global__ void kernelComputeSum(float *scores, int dim){
	int index  = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	int i;

	// 计算缓存中的总和
	for(int k=1; k<dim; k*=2){
		i = index;
		while(i+k<dim){
			if(i%(2*k)==0)
				scores[i] += scores[i+k];
			i += stride; 
		}
		__threadfence(); 
	}
}