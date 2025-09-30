#ifndef __SBTS__LEARNED_SCH_ENV_H__
#define __SBTS__LEARNED_SCH_ENV_H__

#include <vector>
#include <tuple>
#include <map>
#include <iostream>
#include "cn_api.h"

typedef struct {
    CNqueue hqueue;
    int kernel_num;
    int queue_sparsity;
    int queue_priority;
} Queue_st;

typedef struct {
    CNkernel hkernel;
    unsigned int dimx;
    unsigned int dimy;
    unsigned int dimz;
    int predict_time;  //actually size
    KernelClass c;
    Queue_st queueInfo;
} Kernel_st;

typedef std::vector<std::tuple<Queue_st, std::vector<Kernel_st>>> Param_vec;

typedef struct raw_info {
    int averageCoreUtilization;
    Param_vec Param_list;
} Raw_Info_st;

typedef struct observation {
    Raw_Info_st info;
    int kernelNum;
    int queueNum;
    int current_queue;
} Observe_st;


//secret but not necessary code, support api only
CNresult GetIPUUtilization(int *averageUtil, int *coreUtil, CNdev dev);
CNresult ObserveState(Observe_st *Obs);
CNresult AssignCaptureKernel(CNqueue hqueue, int num);
CNresult calcWaitTimeVariance(float *timeVariance);
CNresult calcWaitTimeCV(float *timeCV);

#endif /*__SBTS__LEARNED_SCH_ENV_H__*/
