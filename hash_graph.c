#include "hash_graph.h"

//功能: 生成一个hash图，加速三元组的查询过程
//--------------------------------------------------//
HashGraph createHashGraph(Graph *graph, int hash, int hash_r){

	HashGraph hg;
	hg.num_rels = graph->num_rels;
	hg.num_ents = graph->num_ents; 
	hg.num_triplets = graph->num_triplets;
	hg.triplets = graph->triplets;
	hg.hash = hash;
	hg.hash_r = hash_r; 

	hg.list = (TripletList *)malloc(sizeof(TripletList)*hash); 
	hg.list_r = (TripletList **)malloc(sizeof(TripletList*)*graph->num_rels); 
	for(int i=0; i<graph->num_rels; i++){
		hg.list_r[i] = (TripletList *)malloc(sizeof(TripletList)*hash_r);
		for(int j=0; j<hash_r; j++)
			hg.list_r[i][j] = createTripletList(10); 
	}

	for(int i=0; i<hash; i++)
		hg.list[i] = createTripletList(20); 

	for(int i=0; i<graph->num_triplets; i++){
		addTriplet2HashGraph(&graph->triplets[i], &hg);
	}
	return hg; 
}

//功能: 生成一个RelGraph，用于在GPU中加速计算
//--------------------------------------------------//
RelGraph createRelGraph(Graph *graph){
	RelGraph rg;
	rg.num_rels = graph->num_rels;
	rg.num_ents = graph->num_ents; 
	rg.num_triplets = graph->num_triplets;

	rg.list = (TripletList *)malloc(sizeof(TripletList)*graph->num_rels); 
	for(int i=0; i<graph->num_rels; i++)
		rg.list[i] = createTripletList(20); 

	for(int i=0; i<graph->num_triplets; i++){
		addTriplet2TripletList(&(graph->triplets[i]), &(rg.list[graph->triplets[i].r])); 
	}

	return rg; 
}

//功能: 生成一个不包含任何Triplet的RelGraph(存储负样本的时候用)
//--------------------------------------------------//
RelGraph createEmptyRelGraph(Graph *graph){
	RelGraph rg;
	rg.num_rels = graph->num_rels;
	rg.num_ents = graph->num_ents; 
	rg.num_triplets = graph->num_triplets;

	rg.list = (TripletList *)malloc(sizeof(TripletList)*graph->num_rels); 
	for(int i=0; i<graph->num_rels; i++)
		rg.list[i] = createTripletList(20); 

	return rg; 
}

//--------------------------------------------------//
TripletList createTripletList(int capacity){
	TripletList list;
	list.size = 0;
	list.capacity = capacity;
	list.triplets = (Triplet *)malloc(sizeof(Triplet)*capacity);
	return list; 
}

//--------------------------------------------------//
void createTripletScoreList(TripletScoreList **score_list, int capacity){
	TripletScoreList *list = Malloc(1, TripletScoreList);
	list->size = 0;
	list->capacity = capacity;
	list->triplets = (TripletScore *)malloc(sizeof(TripletScore)*capacity);
	*score_list = list; 
}

//--------------------------------------------------//
void freeTripletList(TripletList *list){
	free(list->triplets);
}

//--------------------------------------------------//
void addTriplet2HashGraph(Triplet *triplet, HashGraph *graph){
	int key = ((triplet->h+1)*(triplet->t+1) + triplet->r)%graph->hash;
	addTriplet2TripletList(triplet, &(graph->list[key]));
	key = (triplet->h + triplet->t)%graph->hash_r;
	addTriplet2TripletList(triplet, &(graph->list_r[triplet->r][key])); 
}

//--------------------------------------------------//
void addTriplet2TripletList(Triplet *triplet, TripletList *list){
	if((list->size)==(list->capacity)){
		enlargeTripletList(list, 10);
	}
	int size = list->size; 
	list->triplets[size].h = triplet->h;
	list->triplets[size].t = triplet->t;
	list->triplets[size].r = triplet->r;
	list->size = list->size + 1;
	return; 
}

// 功能 	添加triplet到list中，如果有重复则不添加
//--------------------------------------------------//
void mergeTriplet2TripletList(Triplet *triplet, TripletList *list){
	for(int i=0; i<list->size; i++){
		if( ((triplet->h)==(list->triplets)[i].h) && \
			((triplet->r)==(list->triplets)[i].r) && \
			((triplet->t)==(list->triplets)[i].t) ){
			return; 
		}
	}

	if((list->size)==(list->capacity)){
		enlargeTripletList(list, 10);
	}
	int size = list->size; 
	list->triplets[size].h = triplet->h;
	list->triplets[size].t = triplet->t;
	list->triplets[size].r = triplet->r;
	list->size = list->size + 1;
	return; 
}

//--------------------------------------------------//
void addTripletScore2TripletScoreList(TripletScore *triplet, TripletScoreList *list){
	if((list->size)==(list->capacity)){
		enlargeTripletScoreList(list, 10);
	}
	int size = list->size;
	memcpy(list->triplets + size, triplet, sizeof(TripletScore));

	list->size = list->size + 1;
	return; 
}

//--------------------------------------------------//
void enlargeTripletList(TripletList *list, int incremental){
	list->capacity = list->capacity + incremental; 
	Triplet *triplets = (Triplet *)malloc(sizeof(Triplet)*(list->capacity)); 
	for(int i=0; i<list->size; i++){
		triplets[i].h = list->triplets[i].h;
		triplets[i].r = list->triplets[i].r;
		triplets[i].t = list->triplets[i].t;
	}
	free(list->triplets);
	list->triplets = triplets; 
}

//--------------------------------------------------//
void enlargeTripletScoreList(TripletScoreList *list, int incremental){
	list->capacity = list->capacity + incremental; 
	TripletScore *triplets = (TripletScore *)malloc(sizeof(TripletScore)*(list->capacity)); 
	memcpy(triplets, list->triplets, list->size*sizeof(TripletScore));
	
	free(list->triplets);
	list->triplets = triplets; 
}

//--------------------------------------------------//
bool isInTripletList(Triplet *triplet, TripletList *list){
	for(int i=0; i<list->size; i++)
		if((list->triplets[i].h==triplet->h)&&(list->triplets[i].t==triplet->t)&&(list->triplets[i].r==triplet->r))
			return true;
	return false;
}

//--------------------------------------------------//
bool isInTripletScoreList(TripletScore *triplet, TripletScoreList *list){
	for(int i=0; i<list->size; i++)
		if((list->triplets[i].triplet.h==triplet->triplet.h)&& \
			(list->triplets[i].triplet.t==triplet->triplet.t)&& \
			(list->triplets[i].triplet.r==triplet->triplet.r))
			return true;
	return false;	
}

//--------------------------------------------------//
bool isInHashGraph(Triplet *triplet, HashGraph *graph){
	int key = ((triplet->h+1)*(triplet->t+1) + triplet->r)%graph->hash;
	return isInTripletList(triplet, &graph->list[key]); 
}

// 功能: 查看属于同一个List中的所有三元组信息
//--------------------------------------------------//
void printTripletListInfo(Triplet *triplet, HashGraph *graph){
	int key = triplet->h + triplet->r + triplet->t; 
	int size = graph->list[key].size;
	for(int i=0; i<size; i++){
		Triplet triplet = (graph->list[key]).triplets[i];
		printf("(h, r, t): %d\t %d\t %d\n", triplet.h, triplet.r, triplet.t); 
	}
}

//--------------------------------------------------//
void printHashGraphInfo(HashGraph *graph){
	printf("--------------------------------------------------\n");
	printf("The information of hash graph is given as: \n");
	printf("num_rels:\t %d\n", graph->num_rels);
	printf("num_ents:\t %d\n", graph->num_ents);
	printf("num_triplets:\t %d\n", graph->num_triplets);
	printf("size of each triplet list in list_r: \n");
	for(int i=0; i<graph->num_rels; i++)
		for(int j=0; j<graph->hash_r; j++)
			printf("%d, ", graph->list_r[i][j].size); 
	printf("\n");
	printf("size of each triplet list in list: \n");
	int count = 0;
	for(int i=0; i<graph->hash; i++){
		if(graph->list[i].size>5){
			printf("%d, ", graph->list[i].size);
			count += 1;
		}
	}
	printf("\nThe count with hash clash greater than 5 is: %d\n", count);
	printf("\n");
}

//--------------------------------------------------//
int getNumberOfInsertedTriplets(HashGraph *graph){
	int size = 0;
	int dimx = graph->num_rels;
	int dimy = graph->hash_r;
	for(int i=0; i<dimx; i++)
		for(int j=0; j<dimy; j++){
			size += graph->list_r[i][j].size; 
		}
	return size; 
}

// 功能 	主要用于在GPU中进行快速确定三元组的存在性
// 参数
// gpu_list 	由多个TripletList构成的数组
//--------------------------------------------------//
void copyHashListToGPU(TripletList *gpu_list, HashGraph *graph){
	int hash = graph->hash;

	TripletList *list = Malloc(hash, TripletList);
	for(int i=0; i<hash; i++){
		list[i].size = (graph->list)[i].size;
		list[i].capacity = (graph->list)[i].capacity;
		Triplet *triplets;
		if(list[i].size!=0){
			gpuErrchk( CudaMalloc(triplets, (list[i].size), Triplet) );
			gpuErrchk( cudaMemcpy(triplets, (graph->list)[i].triplets, \
				list[i].size*sizeof(Triplet), cudaMemcpyHostToDevice) );
			list[i].triplets = triplets; 
		}
	}

	gpuErrchk( cudaMemcpy(gpu_list, list, sizeof(TripletList)*hash, cudaMemcpyHostToDevice) ); 
	free(list);
}

// 功能 	将HashList同时拷贝到多个GPU中
// 参数
// gpu_list 	本身是CPU中地址，但是gpu_list[i]是GPU中地址
//--------------------------------------------------//
void copyHashListToGPUs(TripletList **gpu_list, HashGraph *graph, int num_gpus){
	for(int i=0; i<num_gpus; i++){
		cudaSetDevice(i);
		copyHashListToGPU(gpu_list[i], graph); 
	}
	cudaSetDevice(0);
}

// 功能 用于生成Validation过程的flags
//--------------------------------------------------//
void genTripletsFlags4Validate(Triplet *triplets, bool *flags, int num_triplets){
	int threads = min(NUM_THREADS, num_triplets); 

	#pragma omp parallel num_threads(threads)
	{
		int i = omp_get_thread_num();

		for(int j=i; j<num_triplets; j+=threads){
			flags[j] = true;
			for(int k=j+1; k<num_triplets; k++){
				if(triplets[k].r==triplets[j].r)
					if((triplets[k].h==triplets[j].h)||(triplets[k].t==triplets[j].t)){
						flags[j] = false;
						break;
					}
			}
		}
	}
}

//--------------------------------------------------//
void genTripletScoreList(Graph *graph, TripletScoreList **score_list, \
	Locks *locks, int list_hash){

	int threads = min(NUM_THREADS, list_hash);
	
	#pragma omp parallel num_threads(threads)
 	{
 		int begin, end;
  		int my_rank = omp_get_thread_num();
  		splitDimByBeginEnd(graph->num_triplets, threads, my_rank, &begin, &end);
  		printf("%d: %d %d\n", my_rank, begin, end);

  		int h, r, t; 
  		Triplet triplet; 
		TripletScore triplet_score;
		triplet_score.score = 0; 
  		
  		//加入所有正样本
		for(int i=begin; i<end; i++){
			h = graph->triplets[i].h;
			r = graph->triplets[i].r; 
			t = graph->triplets[i].t; 
			int key = ((h+1)*(t+1) + r)%list_hash; 
			memcpy(&triplet_score, graph->triplets + i, sizeof(Triplet));
			setLock(locks, key);
			addTripletScore2TripletScoreList(&triplet_score, score_list[key]);
			unsetLock(locks, key);
		}
	}
}


/*

  		// 换头
  		for(int i=begin; i<end; i++){
			triplet.r = graph->triplets[i].r;
			triplet.t = graph->triplets[i].t;

			for(int j=0; j<graph->num_ents; j++){
				triplet.h = j; 
				int key = ((j+1)*(triplet.t+1) + tripet.r)%list_hash;
				memcpy(&triplet_score, &triplet, sizeof(Triplet));
				if(!isInTripletScoreList(&triplet_score, score_list[key])){
					setLock(locks, key); 
					addTripletScore2TripletScoreList(&triplet_score, score_list[key]);
					unsetLock(locks, key); 
				}
			}
		}

		// 换尾
  		for(int i=begin; i<end; i++){
			triplet.r = graph->triplets[i].r;
			triplet.h = graph->triplets[i].h;

			for(int j=0; j<graph->num_ents; j++){
				triplet.t = j; 
				int key = ((j+1)*(triplet.h+1) + tripet.r)%list_hash;
				memcpy(&triplet_score, &triplet, sizeof(Triplet));
				if(!isInTripletScoreList(&triplet_score, score_list[key])){
					setLock(locks, key); 
					addTripletScore2TripletScoreList(&triplet_score, score_list[key]);
					unsetLock(locks, key); 
				}
			}
		}
		*/