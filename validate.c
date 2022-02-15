#include "validate.h"

//--------------------------------------------------//
void copyTripletLinkCPU2GPU(TripletLink *link, TripletLink *gpu_link, int size){
	TripletLink *link_cache = Malloc(size, TripletLink);
	memcpy(link_cache, link, size*sizeof(TripletLink));
	for(int i=0; i<size; i++){
		int *links; 
		if(link[i].size==0){
			printf("Found an error!\n");
			exit(0);
		}
		gpuErrchk( CudaMalloc(links, link[i].size, int) );
		cudaMemcpy(links, link[i].link, link[i].size * sizeof(int), cudaMemcpyHostToDevice);
		link_cache[i].link = links; 
	}
	cudaMemcpy(gpu_link, link_cache, size*sizeof(TripletLink), cudaMemcpyHostToDevice);
}

// 功能 	将头部变为-1即为head_triplet, 生成合并的tripletlink指向triplets的链接
//--------------------------------------------------//
void genHeadTripletLink(Triplet *triplets, int num_triplets, TripletList *head_triplets, TripletLink **links){
	TripletLink *temp = Malloc(head_triplets->size, TripletLink);
	for(int i=0; i<head_triplets->size; i++){
		temp[i].size = 0; 
		temp[i].capacity = 2; 
		temp[i].link = Malloc(2, int); 
		memcpy( &(temp[i].triplet), head_triplets->triplets + i, sizeof(Triplet)); 
	}

	for(int i=0; i<num_triplets; i++){
		for(int j=0; j<head_triplets->size; j++){
			if( (triplets[i].t==(head_triplets->triplets)[j].t) && 
				(triplets[i].r==(head_triplets->triplets)[j].r) ){
				//printf("I AM HERE>>>>>>>>>>>>>>>>>>>>>%d %d\n", i, j);
				addLink2TripletLink(temp+j, i);
				break; 
			}
		}
	}

	*links = temp; 
}

// 功能 	将尾巴变为-1即为tail_triplet, 生成合并的tripletlink指向triplets的链接
//--------------------------------------------------//
void genTailTripletLink(Triplet *triplets, int num_triplets, TripletList *tail_triplets, TripletLink **links){
	TripletLink *temp = Malloc(tail_triplets->size, TripletLink);
	for(int i=0; i<tail_triplets->size; i++){
		temp[i].size = 0; 
		temp[i].capacity = 2; 
		temp[i].link = Malloc(2, int); 
		memcpy( &(temp[i].triplet), tail_triplets->triplets + i, sizeof(Triplet)); 
	}

	//printf("OK<<<<<<<<<<<<<<<<<<<< %d %d\n", tail_triplets->size, num_triplets);

	for(int i=0; i<num_triplets; i++){		//
		for(int j=0; j<tail_triplets->size; j++){	//
			if( (triplets[i].h==(tail_triplets->triplets)[j].h) && 
				(triplets[i].r==(tail_triplets->triplets)[j].r) ){
				//printf("I AM HERE>>>>>>>>>>>>>>>>>>>>>%d %d\n", i, j);
				addLink2TripletLink(temp+j, i);
				//printf("HERE END>>>>>>>>>>>>>>>>>>>>>\n");
				break; 
			}
		}
	}

	*links = temp; 

	//printf("FINISHED !!!!!!!!!!!!!\n");
}

// 功能 	将尾巴变为-1, 生成所有的三元组
//--------------------------------------------------//
void genTailTriplets(Triplet *triplets, int num_triplets, TripletList **tail_triplets){
	// 创建一个hash表
	int hash = num_triplets;
	TripletList *list = Malloc(hash, TripletList);
	for(int i=0; i<hash; i++){
		list[i].size = 0;
		list[i].capacity = 1; 
		list[i].triplets = Malloc(1, Triplet);
	}

	// 将所有的triplet去尾后加入hash表
	for(int i=0; i<num_triplets; i++){
		Triplet triplet;
		triplet.h = triplets[i].h;
		triplet.r = triplets[i].r;
		triplet.t = -1;
		int key = (triplet.h+1) * (triplet.r+1) % hash;

		mergeTriplet2TripletList(&triplet, list+key); 
	}

	// 开辟tail_triplets的空间
	*tail_triplets = Malloc(1, TripletList);
	(*tail_triplets)->size = 0; 
	(*tail_triplets)->capacity = 10; 
	(*tail_triplets)->triplets = Malloc(10, Triplet);
	for(int i=0; i<hash; i++)
		for(int j=0; j<list[i].size; j++)
			addTriplet2TripletList(list[i].triplets+j, *tail_triplets);

	for(int i=0; i<hash; i++){
		free(list[i].triplets);
	}
	free(list);
}

// 功能 	将head变为-1, 生成所有的三元组
//--------------------------------------------------//
void genHeadTriplets(Triplet *triplets, int num_triplets, TripletList **tail_triplets){
	// 创建一个hash表
	int hash = num_triplets;
	TripletList *list = Malloc(hash, TripletList);
	for(int i=0; i<hash; i++){
		list[i].size = 0;
		list[i].capacity = 1; 
		list[i].triplets = Malloc(1, Triplet);
	}

	// 将所有的triplet去尾后加入hash表
	for(int i=0; i<num_triplets; i++){
		Triplet triplet;
		triplet.h = -1;
		triplet.r = triplets[i].r;
		triplet.t = triplets[i].t;
		int key = (triplet.t+1) * (triplet.r+1) % hash;

		mergeTriplet2TripletList(&triplet, list+key); 
	}

	// 开辟tail_triplets的空间
	*tail_triplets = Malloc(1, TripletList);
	(*tail_triplets)->size = 0; 
	(*tail_triplets)->capacity = 10; 
	(*tail_triplets)->triplets = Malloc(10, Triplet);
	for(int i=0; i<hash; i++)
		for(int j=0; j<list[i].size; j++)
			addTriplet2TripletList(list[i].triplets+j, *tail_triplets);

	for(int i=0; i<hash; i++){
		free(list[i].triplets);
	}
	free(list);
}

// 功能 	将某个link添加到tripletlink中
//--------------------------------------------------//
void addLink2TripletLink(TripletLink *triplet_link, int ptr){
	int size = triplet_link->size; 

	if(size==triplet_link->capacity)
		enlargeTripletLink(triplet_link, 10);

	triplet_link->link[size] = ptr;
	triplet_link->size += 1; 
}

//--------------------------------------------------//
void enlargeTripletLink(TripletLink *link, int incremental){
	link->capacity = link->capacity + incremental; 
	int *tmp = Malloc(link->capacity, int);
	memcpy(tmp, link->link, link->size * sizeof(int));
	free(link->link);
	link->link = tmp; 
	return; 
}

//--------------------------------------------------//
void copyLinksCPU2GPU(TripletLink *gpu_links, TripletLink *links, int size){
	TripletLink *tmp_links = Malloc(size, TripletLink);
	memcpy(tmp_links, links, size*sizeof(TripletLink));
	for(int i=0; i<size; i++){
		int *tmp;
		gpuErrchk( CudaMalloc(tmp, links[i].size, int) ); 
		cudaMemcpy(tmp, links[i].link, links[i].size*sizeof(int), cudaMemcpyHostToDevice);
		tmp_links[i].link = tmp; 
	}
	cudaMemcpy(gpu_links, tmp_links, size*sizeof(TripletLink), cudaMemcpyHostToDevice); 
	free(tmp_links);
}

// 功能 	与copyLinksCPU2GPU相反主要用于测试
//--------------------------------------------------//
void copyLinksGPU2CPU(TripletLink *links, TripletLink *gpu_links, int size){
	cudaMemcpy(links, gpu_links, sizeof(TripletLink)*size, cudaMemcpyDeviceToHost); 
	for(int i=0; i<size; i++){
		int *tmp = Malloc(links[i].size, int);
		cudaMemcpy(tmp, links[i].link, links[i].size*sizeof(int), cudaMemcpyDeviceToHost);
		links[i].link = tmp; 
	}
}