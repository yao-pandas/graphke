#ifndef _INIT_EMBEDS_H_
#define _INIT_EMBEDS_H_

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <omp.h>

#include "basic_settings.h"

// 功能: 为嵌入向量分配存储空间
//--------------------------------------------------//
Embeds createEmbeds(Graph* graph, int dim_ent, int dim_rel, int base_ent, int base_rel);

// 功能: 开始嵌入向量的初始化
//--------------------------------------------------//
void initEmbeds(Embeds* embeds);

// 功能: 释放存储空间
//--------------------------------------------------//
void freeEmbeds(Embeds *embeds);

// 功能: 将Embeds中的向量值清零
//--------------------------------------------------//
void clearEmbeds(Embeds *embeds);

// 功能: 打印嵌入向量的基本信息
//--------------------------------------------------//
void printEmbedsInfo(Embeds* embeds); 

// 功能:	存储所有embeds
//--------------------------------------------------//
void saveEmbeds(Embeds *embeds, char *filename);

// 功能: 加载所有Embeds
//--------------------------------------------------//
void loadEmbeds(Embeds *embeds, char *filename);

#endif
