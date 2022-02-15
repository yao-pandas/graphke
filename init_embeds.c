#include "init_embeds.h"

// 功能: 为嵌入向量分配存储空间
//--------------------------------------------------//
Embeds createEmbeds(Graph* graph, int dim_ent, int dim_rel, int base_ent, int base_rel){
	float* embed_ents = (float *)malloc(sizeof(float)*dim_ent*graph->num_ents);
	float* embed_rels = (float *)malloc(sizeof(float)*dim_rel*graph->num_rels);
	Embeds embeds = {dim_ent, dim_rel, base_ent, base_rel, graph->num_ents, graph->num_rels, embed_ents, embed_rels};  
	if((embed_ents==NULL)||(embed_rels==NULL)){
		printf("Malloc failed in: \"creatEmbeddingSpace\"\n");
		exit(0); 
	}
	return embeds; 
}

// 功能: 释放存储空间
//--------------------------------------------------//
void freeEmbeds(Embeds *embeds){
	free(embeds->embed_ents);
	free(embeds->embed_rels);
}

// 功能: 将Embeds中的向量值清零
//--------------------------------------------------//
void clearEmbeds(Embeds *embeds){
	clearVector(embeds->embed_ents, embeds->num_ents*embeds->dim_ent);
	clearVector(embeds->embed_rels, embeds->num_rels*embeds->dim_rel);
	return;
}

// 功能: 打印嵌入向量的基本信息
//--------------------------------------------------//
void printEmbedsInfo(Embeds* embeds){
	printf("--------------------------------------------------\n");
	printf("The basic information of embeddings are: \n");
	printf("The embedding dimension for entity is: %d\n", embeds->dim_ent);
	printf("The embedding dimension for relation is: %d\n", embeds->dim_rel);
	printf("The number of entities is: %d\n", embeds->num_ents);
	printf("The number of relations is: %d\n", embeds->num_rels);
	printf("The first 10 values of entities embeddings are: \n");
	for(int i=0; i<min(10, embeds->num_ents*embeds->dim_ent); i++)
		printf("%f\t", embeds->embed_ents[i]);
	printf("\n");
	printf("The first 10 values of relations embeddings are: \n");
	for(int i=0; i<min(10, embeds->num_rels*embeds->dim_rel); i++)
		printf("%f\t", embeds->embed_rels[i]);
	printf("\n");
}

// 功能: 开始嵌入向量的初始化
//--------------------------------------------------//
void initEmbeds(Embeds* embeds){
	int dim_ent = embeds->dim_ent;
	int dim_rel = embeds->dim_rel;
	
	int threads = min(NUM_THREADS, embeds->num_ents);
	threads = min(threads, embeds->num_rels);

	#pragma omp parallel num_threads(threads)
	{
		int begin, end;
		int my_rank = omp_get_thread_num();
		unsigned int seed = (unsigned)(time(NULL)*(my_rank+1)*(my_rank+10)*2);

		int size = embeds->num_ents;
		splitDimByBeginEnd(size, threads, my_rank, &begin, &end);
		for(int i=begin; i<end; i++)
			sampleVector(&(embeds->embed_ents[i*dim_ent]), dim_ent, &seed);

		size = embeds->num_rels;
		splitDimByBeginEnd(size, threads, my_rank, &begin, &end);
		for(int i=begin; i<end; i++)
			sampleVector(&(embeds->embed_rels[i*dim_rel]), dim_rel, &seed);
	}
}

// 功能:	存储所有embeds
//--------------------------------------------------//
void saveEmbeds(Embeds *embeds, char *filename){
	FILE *file = fopen(filename,"wb"); 
	int size;
	size = embeds->num_ents * embeds->dim_ent;
	fwrite(embeds->embed_ents, sizeof(float), size, file);
	size = embeds->num_rels * embeds->dim_rel;
	fwrite(embeds->embed_rels, sizeof(float), size, file);
	fclose(file);
}

// 功能: 加载所有Embeds
//--------------------------------------------------//
void loadEmbeds(Embeds *embeds, char *filename){
	FILE *file = fopen(filename,"rb"); 
	int size;
	size = embeds->num_ents * embeds->dim_ent;
	fread(embeds->embed_ents, sizeof(float), size, file);
	size = embeds->num_rels * embeds->dim_rel;
	fread(embeds->embed_rels, sizeof(float), size, file);
	fclose(file);
}