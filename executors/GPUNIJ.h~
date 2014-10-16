/********************************
タプルの情報はここでまとめておく。

元のプログラムでは構造体のリストだったが、
GPUで動かすため配列のほうが向いていると思ったので
配列に変更している
********************************/

#ifndef GPUNIJ_H
#define GPUNIJ_H

#include <cuda.h>
#include "GPUTUPLE.h"

class GPUNIJ {

public:

#define JT_SIZE 120000000


    TUPLE *lt,*rt;
    JOIN_TUPLE *jt;

    int left,right;

    GPUNIJ(TUPLE *tlt,TUPLE *trt,int leftSize,int rightSize){
        
        lt = tlt;
        rt = trt;
        jt = (JOIN_TUPLE *)malloc(JT_SIZE*sizeof(JOIN_TUPLE));

        left = leftSize;
        right = rightSize;

    }

    ~GPUNIJ(){
        free(jt);
    }
    
    int join(JOIN_TUPLE *jt);

private:

//for partition execution
#define PART 4096    
   
//1blockでのスレッド数の定義。
#define BLOCK_SIZE_X 256  //outer ,left
#define BLOCK_SIZE_Y 512  //inner ,right

#define SELECTIVITY 1000000000
    


    CUresult res;
    CUdevice dev;
    CUcontext ctx;
    CUfunction function,c_function;
    CUmodule module,c_module;
    CUdeviceptr lt_dev, rt_dev, jt_dev,count_dev, pre_dev;
    CUdeviceptr ltn_dev, rtn_dev;
    unsigned int block_x, block_y, grid_x, grid_y;
    
    void printDiff(struct timeval begin, struct timeval end);

    uint iDivUp(uint dividend, uint divisor);


};

#endif
