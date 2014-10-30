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
#include "GPUetc/common/GNValue.h"

using namespace voltdb;


class GPUNIJ {

public:

#define JT_SIZE 120000000


    GPUNIJ();
    ~GPUNIJ();
    
    int join();

    void setTableData(GNValue *lGNV,GNValue *rGNV,int outerSize,int innerSize){
        
        left_GNV = lGNV;
        right_GNV = rGNV;
        left = outerSize;
        right = innerSize;

    }

    RESULT *getJoinResult(){
        return jt;
    }


private:

//for partition execution
   
    RESULT *jt;

    int left,right;

    GNValue *left_GNV;
    GNValue *right_GNV;

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
