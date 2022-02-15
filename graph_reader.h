#ifndef _GRAPH_READER_H_
#define _GRAPH_READER_H_

#include <stdlib.h>
#include <stdio.h>
#include "basic_settings.h"

bool isDelimeter(char c); 
int str2Int(char *str);
int countLine(char *filename); 
bool readIntStr(FILE *fp, char *word);
Triplet* readGraph(char *filename); 
Graph readGraphStruct(char *filename); 
void printGraphInfo(Graph* graph);
void readGraphAsTripletList(char *filename, TripletList *list);

// 功能: 判断某个Triplet是否在Graph中
//--------------------------------------------------//
bool isInGraph(Triplet *triplet, Graph *graph);

// 功能: 获得图中的关系数目
//--------------------------------------------------//
int getNumberOfRelations(Graph *graph); 

#endif
