#include "basic_settings.h"

// 功能: 计算最小值
//--------------------------------------------------//
int min(int x, int y)
{
	return ((x > y) ? y : x);
}

// 打印向量中的结果; 
//--------------------------------------------------//
void printVector(float *list, int len){
	for(int i=0; i<len; i++)
		printf("%e\t", list[i]);
	printf("\n"); 
}

// 将向量中的值置为0; 
//--------------------------------------------------//
void clearVector(float *vector, int dim){
	memset(vector, 0, sizeof(float)*dim);
}

// 功能: 通过openmp实现向量的批量归一化
// 参数: 单个向量的长度是dim, 一共有size个向量
//--------------------------------------------------//
void normalizeVectors(float *vector, int dim, int size){
	int threads = min(NUM_THREADS, size); 

	#pragma omp parallel num_threads(threads)
	{
		int begin, end;
		int my_rank = omp_get_thread_num();
		splitDimByBeginEnd(size, threads, my_rank, &begin, &end);
		for(int k=begin; k<end; k++){
			float norm = cblas_snrm2(dim, &(vector[k*dim]), 1);
			cblas_sscal(dim, 1.0/norm, &(vector[k*dim]), 1);
		}
	}
	return;
}

// 生成一个随机的向量
//--------------------------------------------------//
void randVector(float *vector, int len){
	unsigned int seed = (unsigned)(time(NULL));
	for(int i=0; i<len; i++)
		vector[i] = 2.0*(float)rand_r(&seed)/RAND_MAX - 1;
	return;
}

//功能:	生成长度为size的随机的向量，其中每个元素[0,range)且不重复
//--------------------------------------------------//
void randIndices(int *index, int range, int size){
	int *list = Malloc(range, int); 
	for(int i=0; i<range; i++)
		list[i] = i;
	shuffle(list, range);
	for(int i=0; i<size; i++)
		index[i] = list[i]; 
	free(list);
}

void shuffle(int *array, int n){
	unsigned int seed = (unsigned)(time(NULL));
    if(n>1){
        for(int i=0; i<n-1; i++){
        	int j = i + rand_r(&seed)/(RAND_MAX/(n-i)+1);
          	int t = array[j];
          	array[j] = array[i];
          	array[i] = t;
        }
    }
}

// 在单位立方体内随机采一个向量
//--------------------------------------------------//
void sampleVector(float *list, int len, unsigned int *seed){
	for(int j=0; j<len; j++)
		list[j] = ((float)rand_r(seed))/RAND_MAX;
	//	list[j] = 2.0*(float)rand_r(seed)/RAND_MAX-1;
	//float val = cblas_snrm2(len, list, 1);
	//cblas_sscal(len, 1.0/val, list, 1);
}

// 采用Gibbs方法在单位球的表面随机采样一个向量
//--------------------------------------------------//
void sampleVectorGibbs(float *list, int len, float* sample, unsigned int *seed){
	float x, y, r, theta;
	float pi = 3.14159265360;
	for(int i=0; i<len; i++){
		theta = 2*pi*(float)rand_r(seed)/RAND_MAX;
		r = sqrt(list[i]*list[i] + list[(i+1)%len]*list[(i+1)%len]);
		x = r*cos(theta);
		y = r*sin(theta);
		sample[i] = x;
		sample[(i+1)%len] = y;
	}
	float norm = cblas_snrm2(len, sample, 1); 
	cblas_sscal(len, 1.0/norm, sample, 1);
}

// 将dim平均分成size份, 返回第idx份的开始和结束值
//--------------------------------------------------//
void splitDimByBeginEnd(int dim, int size, int idx, int* begin, int *end){
	int x = dim/size;
	int y = dim%size; 
	if((idx>=0)&&(idx<y)){
		*begin = (x+1)*idx;
		*end = (x+1)*(idx+1); 
	}
	else{
		*begin = y*(x+1) + (idx-y)*x;
		*end = y*(x+1) + (idx-y+1)*x;
	}
}

// 将dim平均分成size份, 返回第idx份的大小
//--------------------------------------------------//
int splitDimBySize(int dim, int size, int idx){
	int x = dim/size;
	int y = dim%size; 
	if((idx>=0)&&(idx<y))
		return x+1; 
	else
		return x; 
}

// 功能: 给出数组长度, 判断某个整数是否在数组中
//--------------------------------------------------//
bool isInArray(int x, int* list, int len){
	if(len==0)
		return false;
	for(int i=0; i<len; i++)
		if(list[i]==x)
			return true;
	return false; 
}

// 功能: 给出x在list中的位置, 如果不存在返回-1
//--------------------------------------------------//
inline int indexInArray(int x, int* list, int len){
	if(len==0)
		return -1;
	for(int i=0; i<len; i++)
		if(list[i]==x)
			return i;
	return -1; 
}

// 功能: 将x限制到[0,level)的区间, 注意x不会小于-1; 
//--------------------------------------------------//
inline float truncate(float x){
	float val = x+1;
	return val - (int)val; 
}

// 获取向量中每个元素的exp次方
//--------------------------------------------------//
void getExponentOfVector(float *list, int len, float exp){
	int i;

	#pragma omp parallel for num_threads(NUM_THREADS)
	for(i=0; i<len; i++)
		list[i] = pow(list[i], exp); 
}

// 创建一个有锁构成的数组
//--------------------------------------------------//
Locks createLocks(int size){
	omp_lock_t *locks_p = (omp_lock_t*)malloc(sizeof(omp_lock_t)*size); 
	Locks locks = {locks_p, size}; 
	return locks; 
}

// 释放锁
//--------------------------------------------------//
void freeLocks(Locks *locks){
	for(int i=0; i<locks->size; i++)
		omp_destroy_lock(&locks->locks[i]);
	free(locks->locks); 
}

// 初始化
//--------------------------------------------------//
void initLocks(Locks *locks){
	for(int i=0; i<locks->size; i++)
		omp_init_lock(&locks->locks[i]);
}

// 设置锁
//--------------------------------------------------//
void setLock(Locks *locks, int idx){
	omp_set_lock(&locks->locks[idx]); 
}

// 尝试加锁
//--------------------------------------------------//
bool testLock(Locks *locks, int idx){
	return omp_test_lock(&locks->locks[idx]); 
}

// 解开锁
//--------------------------------------------------//
void unsetLock(Locks *locks, int idx){
	omp_unset_lock(&locks->locks[idx]); 
}