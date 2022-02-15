#ifndef _HASH_GRAPH_H_
#define _HASH_GRAPH_H_

#include <omp.h>
#include <stdio.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "graph_reader.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

//功能: 生成一个hash图，加速三元组的查询过程
//--------------------------------------------------//
HashGraph createHashGraph(Graph *graph, int hash, int hash_r);

//功能: 生成一个RelGraph，用于在GPU中加速计算
//--------------------------------------------------//
RelGraph createRelGraph(Graph *graph);

//功能: 生成一个不包含任何Triplet的RelGraph(存储负样本的时候用)
RelGraph createEmptyRelGraph(Graph *graph);

//--------------------------------------------------//
TripletList createTripletList(int capacity);

//--------------------------------------------------//
void createTripletScoreList(TripletScoreList **score_list, int capacity);

//--------------------------------------------------//
void freeTripletList(TripletList *list);

//--------------------------------------------------//
void addTriplet2HashGraph(Triplet *triplet, HashGraph *graph);

//--------------------------------------------------//
void addTriplet2TripletList(Triplet *triplet, TripletList *list);

// 功能 	添加triplet到list中，如果有重复则不添加
//--------------------------------------------------//
void mergeTriplet2TripletList(Triplet *triplet, TripletList *list);

//--------------------------------------------------//
void enlargeTripletList(TripletList *list, int incremental);

//--------------------------------------------------//
bool isInTripletList(Triplet *triplet, TripletList *list);

//--------------------------------------------------//
bool isInTripletScoreList(TripletScore *triplet, TripletScoreList *list);

//--------------------------------------------------//
bool isInHashGraph(Triplet *triplet, HashGraph *graph);

// 打印在graph中triplet所在的组的所有信息
//--------------------------------------------------//
void printTripletListInfo(Triplet *triplet, HashGraph *graph);

// 打印静态图中的信息
//--------------------------------------------------//
void printHashGraphInfo(HashGraph *graph);

// 获取静态图中已经插入的Triplet数目
//--------------------------------------------------//
int getNumberOfInsertedTriplets(HashGraph *graph);

// 功能 	主要用于在GPU中进行快速确定三元组的存在性
// 参数
// gpu_list 	由多个TripletList构成的数组
//--------------------------------------------------//
void copyHashListToGPU(TripletList *gpu_list, HashGraph *graph);

// 功能 	将HashList同时拷贝到多个GPU中
//--------------------------------------------------//
void copyHashListToGPUs(TripletList **gpu_list, HashGraph *graph, int num_gpus);

// 功能 用于生成Validation过程的flags
//--------------------------------------------------//
void genTripletsFlags4Validate(Triplet *triplets, bool *flags, int num_triplets);

// 功能 	用于生成Validation过程的indices
//		对于每个triplet，换头的时候其指向下标i的triplet的所有换头样本，换尾的时候指向下标j的换尾样本
//--------------------------------------------------//
void genTripletsIndices(Triplet *triplets, int *indices, int num_triplets);

//--------------------------------------------------//
void enlargeTripletScoreList(TripletScoreList *list, int incremental);

//--------------------------------------------------//
void addTripletScore2TripletScoreList(TripletScore *triplet, TripletScoreList *list);

//--------------------------------------------------//
void genTripletScoreList(Graph *graph, TripletScoreList **score_list, \
	Locks *locks, int list_hash);

//--------------------------------------------------//
void addTripletScore2TripletScoreList(TripletScore *triplet, TripletScoreList *list);

//--------------------------------------------------//
void enlargeTripletScoreList(TripletScoreList *list, int incremental);

#endif
