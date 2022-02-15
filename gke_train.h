#include <cuda_runtime.h>
#include <cblas.h>
#include "timer.h"
#include "gpu_score.h"
#include "gpu_sample.h"
#include "hash_graph.h"
#include "basic_settings.h"
#include "init_embeds.h"
#include "validate.h"
#include "gpu_validate.h"
#include "loss.h"

//--------------------------------------------------//
void train();