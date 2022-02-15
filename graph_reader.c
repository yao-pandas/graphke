#include "graph_reader.h"

//功能: 读取一个知识图谱数据集, 返回长度T的数组, 其中T标识triplet个数; 
//注意: 这个数据集只允许每行有3个元素, 中间通过空格或制表符隔开, 且最后一行没有换行符; 
//--------------------------------------------------//
Triplet* readGraph(char *filename){
	FILE *fp;
	int idx = 0; 
	int line = countLine(filename); 
	if((fp = fopen(filename, "r"))==NULL){
		printf("Cannot open the following file %s\n", filename);
		return NULL; 
	}
	Triplet *triplets = (Triplet *)malloc(line*sizeof(Triplet)); 
	char word[MAX_WORD_LEN + 1] = {'0'};

	int k, val; 
	while(readIntStr(fp, word)){
		val = str2Int(word); 
		k = idx/3; 
		switch((idx++)%3){
			case 0: triplets[k].h = val; break;
			case 1: triplets[k].r = val; break;
			case 2: triplets[k].t = val; break;
		}
	}

	fclose(fp);
	return triplets; 
}

//功能: 读取一个知识图谱数据集, 以TripletList的方式返回
//注意: 这个数据集只允许每行有3个元素, 中间通过空格或制表符隔开, 且最后一行没有换行符; 
//--------------------------------------------------//
void readGraphAsTripletList(char *filename, TripletList *list){
	FILE *fp;
	int idx = 0; 
	int line = countLine(filename); 
	if((fp = fopen(filename, "r"))==NULL){
		printf("Cannot open the following file %s\n", filename);
		return; 
	}
	Triplet *triplets = (Triplet *)malloc(line*sizeof(Triplet)); 
	char word[MAX_WORD_LEN + 1] = {'0'};

	int k, val; 
	while(readIntStr(fp, word)){
		val = str2Int(word); 
		k = idx/3; 
		switch((idx++)%3){
			case 0: triplets[k].h = val; break;
			case 1: triplets[k].r = val; break;
			case 2: triplets[k].t = val; break;
		}
	}

	fclose(fp);
	list->triplets = triplets;
	list->size = line;
	list->capacity = line; 
}

//功能: 将文件中的数据读入到一个Graph结构体中
//--------------------------------------------------//
Graph readGraphStruct(char *filename){
	Triplet* triplets = readGraph(filename);
	int num_triplets, num_rels, num_ents; 
	num_triplets = countLine(filename);

	num_ents = 0;
	num_rels = 0; 
	
	for(int i=0; i<num_triplets; i++){
		if(triplets[i].h > num_ents)
			num_ents = triplets[i].h;
		if(triplets[i].r > num_rels)
			num_rels = triplets[i].r;
		if(triplets[i].t > num_ents)
			num_ents = triplets[i].t;
	}
	Graph graph = {triplets, num_ents+1, num_rels+1, num_triplets}; 
	return graph; 
}

//功能: 打印一个图结构体中的基本信息
//--------------------------------------------------//
void printGraphInfo(Graph* graph){
	printf("--------------------------------------------------\n");
	printf("The basic information of the loaded graph is: \n");
	printf("The number of entities:\t %d\n", graph->num_ents);
	printf("The number of relations:\t %d\n", graph->num_rels);
	printf("The number of triplets:\t %d\n", graph->num_triplets);
	printf("The loaded triplets are: \n");
	for(int i=0; i<graph->num_triplets; i++)
		printf("%d\t%d\t%d\n", (graph->triplets[i]).h, (graph->triplets[i]).r, (graph->triplets[i]).t);
	return;
}

//将一个整数字符串转为对应的整数
//--------------------------------------------------//
int str2Int(char *str){
	int l = -1;
	int digits = 0; 
	while(str[++l]!='\0'); 
	for(int i=0; i<l; i++){
		digits = digits*10 + (str[i] - '0');
	}
	return digits; 
}

//从文件当前位置向下读出一个整数字符串
//--------------------------------------------------//
bool readIntStr(FILE *fp, char *word){
	char c; 
	bool flag1, flag2; 
	int idx = 0; 
	flag2 = true; 
	while(!feof(fp)){
		c = fgetc(fp);
		flag1 = flag2; 
		flag2 = isDelimeter(c);
		if(flag1 && !flag2){
			while(!feof(fp) && !isDelimeter(c)){
				word[idx++] = c; 
				c = fgetc(fp);
			}
			word[idx++] = '\0'; 
			return true; 
		}
	}
	return false;
}

//判断一个字符是否是空字符
//--------------------------------------------------//
bool isDelimeter(char c){
	return (c=='\n' || c=='\t' || c==' ' || c=='\r'); 
}

//计算一个文件中有多少行
//--------------------------------------------------//
int countLine(char *filename){
	FILE *fp;
	if((fp = fopen(filename, "r"))==NULL){
		printf("Cannot open the following file %s\n", filename);
		return -1; 
	}
	int line = 1;
	while(!feof(fp)){
		if((fgetc(fp)=='\n'))
			line++;
	}
	fclose(fp); 
	return line; 
}

// 功能: 判断某个Triplet是否在Graph中
//--------------------------------------------------//
bool isInGraph(Triplet *triplet, Graph *graph){
	for(int i=0; i<graph->num_triplets; i++)
		if ((graph->triplets[i].h == triplet->h)&&\
			(graph->triplets[i].r == triplet->r)&&\
			(graph->triplets[i].t == triplet->t))
			return true; 
	return false; 
}

// 功能: 获得图中的关系数目
//--------------------------------------------------//
int getNumberOfRelations(Graph *graph){
	int number = 0;
	for(int i=0; i<graph->num_triplets; i++){
		if(graph->triplets[i].r>number)
			number = graph->triplets[i].r;
	}
	return number+1; 
}