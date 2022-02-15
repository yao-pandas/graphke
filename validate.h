#ifndef _VALIDATE_H_
#define _VALIDATE_H_

#include "init_embeds.h"
#include "timer.h"
#include "gpu_score.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>

typedef struct{
	Triplet triplet;
	int *link; 
	int size; 
	int capacity; 
}TripletLink;

//--------------------------------------------------//
void genTailTripletLink(Triplet *triplets, int num_triplets, TripletList *tail_triplets, TripletLink **links);

//--------------------------------------------------//
void genHeadTripletLink(Triplet *triplets, int num_triplets, TripletList *head_triplets, TripletLink **links);

// 功能 	将tail变为-1, 生成所有的三元组
//--------------------------------------------------//
void genTailTriplets(Triplet *triplets, int num_triplets, TripletList **tail_triplets);

// 功能 	将head变为-1, 生成所有的三元组
//--------------------------------------------------//
void genHeadTriplets(Triplet *triplets, int num_triplets, TripletList **tail_triplets);

//--------------------------------------------------//
void addLink2TripletLink(TripletLink *triplet_link, int ptr);

//--------------------------------------------------//
void enlargeTripletLink(TripletLink *link, int incremental);

//--------------------------------------------------//
void copyLinksCPU2GPU(TripletLink *gpu_links, TripletLink *links, int size);

// 功能 	与copyLinksCPU2GPU相反主要用于测试
//--------------------------------------------------//
void copyLinksGPU2CPU(TripletLink *links, TripletLink *gpu_links, int size);

#endif
