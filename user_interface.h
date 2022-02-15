#ifndef _USER_INTERFACE_H_
#define _USER_INTERFACE_H_

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

//--------------------------------------------------//
extern __device__ void ScoreTransE(float *h, float *r, float *t, float *score, \
	float *gh, float *gr, float *gt, int sigma, bool flag);

//--------------------------------------------------//
extern __device__ void ScoreTorusE(float *h, float *r, float *t, float *score, \
	float *gh, float *gr, float *gt, int sigma, bool flag);

//--------------------------------------------------//
extern __device__ void ScoreRotatE(float *h, float *r, float *t, float *score, \
	float *gh, float *gr, float *gt, int sigma, bool flag);

//--------------------------------------------------//
extern __device__ void ScoreComplEx(float *h, float *r, float *t, float *score, \
	float *gh, float *gr, float *gt, int sigma, bool flag);

#endif