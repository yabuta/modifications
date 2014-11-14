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
#include "GPUetc/expressions/Gcomparisonexpression.h"

using namespace voltdb;

class GPUNIJ{

public:

#define JT_SIZE 120000000


    GPUNIJ();
    ~GPUNIJ();

    bool initGPU();
    bool join();


/**
   outer tuple = left
   inner tuple = right
 */

    void setTableData(GNValue *oGNV,GNValue *iGNV,int outerSize,int innerSize){
        
        left_GNV = oGNV;
        right_GNV = iGNV;
        left = outerSize;
        right = innerSize;

        PART = 262144;
        
        uint biggerTupleSize = left;
        if(left < right) biggerTupleSize = right;

        for(int i=32768 ; i<=262144 ; i = i*2){
            if(biggerTupleSize<=i){
                PART = i;
                break;
            }
        }
        printf("PART : %d\n",PART);

    }
    void setExpression(GComparisonExpression *GC){
        expression = GC;        
    }

    RESULT *getResult(){
        return jt;
    }

    int getResultSize(){
        return total;
    }


private:

//for partition execution
   
    RESULT *jt;
    int total;

    uint left,right;
    GNValue *left_GNV;
    GNValue *right_GNV;

    GComparisonExpression *expression;

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
