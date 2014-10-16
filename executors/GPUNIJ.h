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

    GPUNIJ();
    ~GPUNIJ();
    
    int join();

    void setData(TUPLE *tlt,TUPLE *trt,int leftSize,int rightSize){
        
        lt = tlt;
        rt = trt;
        jt = (JOIN_TUPLE *)malloc(JT_SIZE*sizeof(JOIN_TUPLE));

        left = leftSize;
        right = rightSize;

        if(leftSize<524288&&rightSize<524288){
            PART = leftSize;
        }

    }

    JOIN_TUPLE *getResult(){
        return jt;
    }


private:

//for partition execution
   
//1blockでのスレッド数の定義。
#define BLOCK_SIZE_X 256  //outer ,left
#define BLOCK_SIZE_Y 512  //inner ,right

#define SELECTIVITY 1000000000

    int PART;

    CUresult res;
    CUdevice dev;
    CUcontext ctx;
    CUfunction function,c_function;
    CUmodule module,c_module;
    
    void printDiff(struct timeval begin, struct timeval end);

    uint iDivUp(uint dividend, uint divisor);


};

#endif
