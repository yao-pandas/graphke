#include "user_interface.h"

// 功能 	TransE分值计算
//--------------------------------------------------//
extern __device__ void ScoreTransE(float *h, float *r, float *t, float *score, \
	float *gh, float *gr, float *gt, int sigma, bool flag){
	*score = fabs(*h + *r - *t); 
	if(!flag)
		return;

	if(sigma==1){
		*gh = 1;
		*gr = 1; 
		*gt = -1;
	}
	if(sigma==2){
		*gh = *score;
		*gr = *score;
		*gt = (-1)*(*score);
	}
}

// 功能 	TorusE分值计算
//--------------------------------------------------//
extern __device__ void ScoreTorusE(float *h, float *r, float *t, float *score, \
	float *gh, float *gr, float *gt, int sigma, bool flag){
	float tmp = *h + *r - *t;
	while(tmp>=1)
		tmp -= 1;
	while(tmp<0)
		tmp += 1;
	*score = tmp<0.5 ? tmp : 1-tmp;

	if(!flag)
		return;

	if(tmp<0.5){
		*gh = 1;
		*gr = 1;
		*gt = -1;
	}
	else{
		*gh = -1;
		*gr = -1; 
		*gt = 1;
	}

	/*if(sigma==1){
		*gh = 1;
		*gr = 1; 
		*gt = -1;
	}
	if(sigma==2){
		*gh = *score;
		*gr = *score;
		*gt = (-1)*(*score);
	}*/
}

// 功能 	RotatE分值计算 |hor-t|
//--------------------------------------------------//
extern __device__ void ScoreRotatE(float *h, float *r, float *t, float *score, \
	float *gh, float *gr, float *gt, int sigma, bool flag){
	float real = h[0]*r[0] - h[1]*r[1] - t[0];
	float img = h[0]*r[1] + h[1]*r[0] - t[1];
	*score = real*real + img*img;

	if(!flag)
		return;

	if(sigma==1){
		float coef = sqrtf(*score);
		gh[0] = 2*coef*r[0];
		gh[1] = 2*coef*r[1];
		gr[0] = 2*coef*h[0];
		gr[1] = 2*coef*h[1];
		gt[0] = (-1)*2*coef;
		gt[1] = (-1)*2*coef;
	}
}

// 功能 	ComplEx分值计算 Re<r,h,\bar{t}>
//--------------------------------------------------//
extern __device__ void ScoreComplEx(float *h, float *r, float *t, float *score, \
	float *gh, float *gr, float *gt, int sigma, bool flag){
	float real = h[0]*r[0] - h[1]*r[1];
	float img  = h[0]*r[1] + h[1]*r[0];
	*score = real*t[0] + img*t[1];

	if(!flag)
		return;

	if(sigma==1){
		gh[0] = r[0]*t[0] + r[1]*t[1];
		gh[1] = r[1]*t[0] - r[0]*t[1];
		gr[0] = h[0]*t[0] + h[1]*t[1]; 
		gr[1] = h[1]*t[0] - h[0]*t[1];
		gt[0] = h[0]*r[0] - h[1]*r[1];
		gt[1] = h[0]*r[1] + h[1]*r[0];
	}
}