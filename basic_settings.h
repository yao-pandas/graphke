#ifndef _BASIC_SETTINGS_H_
#define _BASIC_SETTINGS_H_

#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <cblas.h>
#include <string.h>
#include <stdio.h>
#include "definitions.h"

#define Malloc(n, type) ((type *)malloc(n*sizeof(type)))

#define CudaMalloc(x, n, type) (cudaMalloc((void**)&(x), sizeof(type)*(n)))

typedef struct{
	omp_lock_t *locks;
	int size;
}Locks;

// 功能: 计算2个数的最小值
//--------------------------------------------------//
int min(int x, int y);

// 打印向量中的结果; 
//--------------------------------------------------//
void printVector(float *list, int len);

// 将向量中的值置为0; 
//--------------------------------------------------//
void clearVector(float *vector, int dim);

// 生成一个随机的向量在[-1,1]之间均匀分布
//--------------------------------------------------//
void randVector(float *vector, int len); 

// 功能: 通过openmp实现向量的批量归一化
// 参数: 单个向量的长度是dim, 一共有size个向量
//--------------------------------------------------//
void normalizeVectors(float *vector, int dim, int size);

// 在单位球的表面随机采样一个向量(通过立方体截断的方法)
//--------------------------------------------------//
void sampleVector(float *list, int len, unsigned int *seed); 

// 采用Gibbs方法在单位球的表面随机采样一个向量
//--------------------------------------------------//
void sampleVectorGibbs(float *list, int len, float* sample, unsigned int *seed);

// 将dim平均分成size份, 返回第idx份的开始和结束值
//--------------------------------------------------//
void splitDimByBeginEnd(int dim, int size, int idx, int* begin, int *end); 

// 将dim平均分成size份, 返回第idx份的大小
//--------------------------------------------------//
int splitDimBySize(int dim, int size, int idx); 

// 功能: 给出数组长度, 判断某个整数是否在数组中
//--------------------------------------------------//
bool isInArray(int x, int* list, int len);

//功能:	判断某个整数在数组中的下标, 不在则返回-1
//--------------------------------------------------//
int indexInArray(int x, int* list, int len);

//功能:	生成长度为size的随机的向量，其中每个元素[0,range)且不重复
//--------------------------------------------------//
void randIndices(int *index, int range, int size);

void shuffle(int *array, int n);

// 功能: 将x限制到[0,level)的区间; 
//--------------------------------------------------//
float truncate(float x);

// 获取向量中每个元素的exp次方
//--------------------------------------------------//
void getExponentOfVector(float *list, int len, float exp);

// 创建一个有锁构成的数组
//--------------------------------------------------//
Locks createLocks(int size);

// 释放锁
//--------------------------------------------------//
void freeLocks(Locks *locks);

// 初始化
//--------------------------------------------------//
void initLocks(Locks *locks);

// 设置锁
//--------------------------------------------------//
void setLock(Locks *locks, int idx);

// 尝试加锁
//--------------------------------------------------//
bool testLock(Locks *locks, int idx);

// 解开锁
//--------------------------------------------------//
void unsetLock(Locks *locks, int idx);

#endif
