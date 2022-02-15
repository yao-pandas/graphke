#ifndef _DEFINITIONS_H_
#define _DEFINITIONS_H_

#define MAX_WORD_LEN 20
#define NUM_THREADS 32

// Note: The User Defined Score Functions Are Given in user_interface.cu
//******************************BEGIN: Training Parameters
#define GAMMA 		16
#define STEP_SIZE 	0.01
#define DIM_ENT 	512
#define DIM_REL 	512
#define DIM 		512

#define SEED 		87231 		//GPU中采样种子
#define SIGMA 		1 			//采用L1范数
#define BATCH_SIZE 	420000
#define NEG_DEG 	1
#define BATCHES 	10
#define CYCLIC		true 		//取连续的样本作为正样本

#define LOSS_STEP	1 			//多少个batch输出一次loss

#define grid_size 	32
#define block_size 	128

#define HASH 		320001 		//构造Hash图加速查询
#define HASH_R 		1200 		

#define ScoreFunc 	ScoreTorusE //分值函数

#define FOLDER "../../data/"	//数据所在的文件夹
#define DATA "FB15k"			//数据集名称需要加引号

#define FILE_TRAIN 	FOLDER""DATA"/train.data"
#define FILE_TEST 	FOLDER""DATA"/test.data"
#define FILE_VALID 	FOLDER""DATA"/valid.data"
//******************************END: Training Parameters

#define DEBUG 0

typedef struct{
	int dim_ent; 
	int dim_rel;
	int size; 
	float *gh;
	float *gr; 
	float *gt; 
	float *ngh; 
	float *ngr;
	float *ngt;
	float *scores;
}GradCache; 

typedef struct{
	int h;
	int r;
	int t;
}Triplet; 

typedef struct{
	Triplet triplet;
	float score; 
}TripletScore;

typedef struct{
	int capacity;
	int size;
	Triplet *triplets; 
}TripletList;

typedef struct{
	int capacity;
	int size;
	TripletScore *triplets; 
}TripletScoreList;

typedef struct{
	Triplet *triplets;
	int num_ents;
	int num_rels;
	int num_triplets; 
}Graph;

// 主要用于存储一个图，加速对于图中数据的读取和写入过程
typedef struct{
	TripletList *list;		//给定三元组查询是否存在
	TripletList **list_r;	//指定关系快速查询到三元组
	Triplet *triplets;
	int hash;
	int hash_r;
	int num_ents; 
	int num_rels; 
	int num_triplets; 
}HashGraph;

// 用于通过给定的r去索引所有的三元组, 用于加速GPU计算
typedef struct 
{
	TripletList *list;	//这里的list是一个数组
	int num_ents;
	int num_rels;
	int num_triplets; 
}RelGraph;

typedef struct{
	int dim_ent;
	int dim_rel;
	int base_ent; 
	int base_rel; 
	int num_ents;
	int num_rels;
	float* embed_ents;
	float* embed_rels;
}Embeds;

#endif
