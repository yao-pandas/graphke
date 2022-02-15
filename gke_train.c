#include "gke_train.h"

//--------------------------------------------------//
void train(){

	// 构造hash图的参数
	int hash = HASH;
	int hash_r = HASH_R;

	char* filename = FILE_TRAIN;
	Graph graph = readGraphStruct(filename);

	printf("--------------------------------------------------\n");
	printf("Reading graph and preparing data ... \n");

	// 读入验证集的相关数据用于输出loss
	filename = FILE_VALID;
	TripletList triplets_valid; 
	readGraphAsTripletList(filename, &triplets_valid);

	Triplet *gpu_triplets_valid;
	gpuErrchk( CudaMalloc(gpu_triplets_valid, triplets_valid.size, Triplet) );
	copyTripletsCPU2GPU(gpu_triplets_valid, triplets_valid.triplets, triplets_valid.size);
	float *scores_valid; 
	gpuErrchk( CudaMalloc(scores_valid, triplets_valid.size, float) );

	HashGraph hash_graph = createHashGraph(&graph, hash, hash_r);
	
	// GPU中的hash表用于加速负样本采样
	TripletList *gpu_list;
	gpuErrchk( CudaMalloc(gpu_list, hash, TripletList) );
	copyHashListToGPU(gpu_list, &hash_graph);

	int num_ents = graph.num_ents;
	int num_rels = graph.num_rels;
	int num_triplets = graph.num_triplets; 
	int seed = SEED; 
	int sigma = SIGMA; 
	int batch_size = BATCH_SIZE; 
	int neg_deg = NEG_DEG; 
	int batches = BATCHES;
	int dim_ent = DIM_ENT;
	int dim_rel = DIM_REL;
	int dim = DIM; 
	if((DIM_ENT%DIM!=0)||(DIM_REL%DIM!=0)){
		printf("Parameter setting error!\n");
		exit(0);
	}

	int base_ent = DIM_ENT/DIM;
	int base_rel = DIM_REL/DIM; 

	// 创建CPU中的嵌入向量
	Embeds embeds = createEmbeds(&graph, dim_ent, dim_rel, base_ent, base_rel);
	initEmbeds(&embeds);

	// 在GPU中开辟嵌入向量存储空间
	Embeds *gpu_embeds, *gpu_grads;
	createEmbedsInGPU(&graph, dim_ent, dim_rel, base_ent, base_rel, &gpu_embeds);
	createEmbedsInGPU(&graph, dim_ent, dim_rel, base_ent, base_rel, &gpu_grads);
	copyEmbedsCPU2GPU(gpu_embeds, &embeds);

	// 为负采样准备的正负样本空间
	Triplet *batch_triplets, *neg_triplets, *gpu_triplets;
	gpuErrchk( CudaMalloc(batch_triplets, batch_size, Triplet) );
	gpuErrchk( CudaMalloc(neg_triplets, batch_size*neg_deg, Triplet) );
	gpuErrchk( CudaMalloc(gpu_triplets, num_triplets, Triplet) );
	gpuErrchk( cudaMemcpy(gpu_triplets, graph.triplets, \
		sizeof(Triplet)*num_triplets, cudaMemcpyHostToDevice) ); 

	printf("--------------------------------------------------\n");
	printf("The dataset contains %d entities, %d relations, %d triplets.\n", num_ents, num_rels, num_triplets);
	printf("Embedding dimension for entity is %d, for relation is %d. \n", dim_ent, dim_rel);
	printf("Training batch size: %d.\n", batch_size);
	printf("Training batches: %d.\n", batches);
	printf("Negative sampling degree: %d.\n", neg_deg);

	printf("--------------------------------------------------\n");
	printf("Start training ... \n");

	// 开始训练计时
	cudaDeviceSynchronize();
	double start, finish;
	GET_TIME(start);
	for(int i=0; i<batches; i++){
		if((i>0)&&(i%LOSS_STEP==0)){
			float loss = computeLoss(gpu_embeds, gpu_triplets_valid, scores_valid, triplets_valid.size);
			printf("The loss after training %d batches is: %f.\n", i, loss);
		}
		gpuNegativeSampling(gpu_triplets, gpu_list, \
			batch_triplets, neg_triplets, \
			num_ents, num_rels, num_triplets, \
			batch_size, neg_deg, seed, hash);
		gpuComputeGradsInSingleGPU(gpu_embeds, gpu_grads,\
			batch_triplets, neg_triplets, \
			batch_size, neg_deg, sigma);
	}
	cudaDeviceSynchronize();
	GET_TIME(finish);
	printf("Training done, time cost for training %d batches: %e.\n", batches, finish-start);

	// 释放训练中的显存
	cudaFree(batch_triplets);
	cudaFree(neg_triplets);
	cudaFree(gpu_triplets);
	cudaFree(gpu_triplets_valid);
	freeEmbedsInGPU(gpu_grads);

	printf("--------------------------------------------------\n");
	printf("Preparing data for testing ...\n");

	// 读入测试集的相关数据
	filename = FILE_TEST;

	TripletList triplets_test; 
	readGraphAsTripletList(filename, &triplets_test);

	TripletList *tail_list, *head_list; 
	TripletLink *tail_links, *head_links; 
	genTailTriplets(triplets_test.triplets, triplets_test.size, &tail_list);
	genTailTripletLink(triplets_test.triplets, triplets_test.size, tail_list, &tail_links);
	genHeadTriplets(triplets_test.triplets, triplets_test.size, &head_list);
	genHeadTripletLink(triplets_test.triplets, triplets_test.size, head_list, &head_links);

	// 为测试准备的临时数据空间
	int *ranks;
	float *scores, *neg_scores, *indices, *hits; 
	gpuErrchk( CudaMalloc(ranks, triplets_test.size, int) );
	gpuErrchk( CudaMalloc(hits, 3, float) );
	gpuErrchk( CudaMalloc(indices, 2, float) );
	gpuErrchk( CudaMalloc(scores, triplets_test.size, int) );
	gpuErrchk( CudaMalloc(neg_scores, num_ents, int) );

	gpuErrchk( CudaMalloc(neg_triplets, num_ents, Triplet) );

	Triplet *gpu_triplets_test;
	gpuErrchk( CudaMalloc(gpu_triplets_test, triplets_test.size, Triplet) );
	copyTripletsCPU2GPU(gpu_triplets_test, triplets_test.triplets, triplets_test.size);

	TripletLink *gpu_head_links, *gpu_tail_links; 
	gpuErrchk( CudaMalloc(gpu_head_links, head_list->size, TripletLink) );
	gpuErrchk( CudaMalloc(gpu_tail_links, tail_list->size, TripletLink) );
	copyLinksCPU2GPU(gpu_head_links, head_links, head_list->size);
	copyLinksCPU2GPU(gpu_tail_links, tail_links, tail_list->size);

	printf("--------------------------------------------------\n");
	printf("Start testing ... \n");

	gpuErrchk( cudaDeviceSynchronize() ); 
	GET_TIME(start);
	gpuFastValidate(gpu_embeds, gpu_list, \
		gpu_triplets_test, scores, neg_triplets, neg_scores,\
		gpu_head_links, gpu_tail_links, \
		head_list->size, tail_list->size, \
		ranks, hits, indices, indices+1, \
		dim, num_ents, num_rels, triplets_test.size, hash);
	gpuErrchk( cudaDeviceSynchronize() ); 
	GET_TIME(finish);
	printf("Testing done, time cost for testing: %e\n", finish-start);

	float cpu_hits[3], cpu_indices[2];
	cudaMemcpy(cpu_hits, hits, 3*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_indices, indices, 2*sizeof(float), cudaMemcpyDeviceToHost);
	printf("HITS@1: %f\nHITS@3: %f\nHITS@10: %f\nMR: %f\nMRR: %f\n", cpu_hits[0], cpu_hits[1], cpu_hits[2], cpu_indices[0], cpu_indices[1]); 

	// 程序在此处结束故不再进行相应的显存清理
}

//--------------------------------------------------//
int main(){
	train();
	return 0;
}