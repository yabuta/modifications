#include <stdio.h>
#include <stdint.h>
#include <cuda.h>
#include <sys/time.h>
#include "GPUTUPLE.h"
#include "GPUetc/common/GNValue.h"
#include "GPUetc/expressions/Gcomparisonexpression.h"

using namespace voltdb;

extern "C" {

  /**
     called function is changed by join condition.
     
     if T1.val = T2.val, iocount and iojoin is called.
     if T.val1 = T.val2 , iicount and iijoin is called.
   */


__global__
void count(
          GNValue *ognv,
          GNValue *ignv,
          GComparisonExpression ex,
          int *count,
          int ltn,
          int rtn
          ) 

{

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * gridDim.x * blockDim.x;

  if(i<ltn){
    /*
    __shared__ GNValue tignv[BLOCK_SIZE_Y];
    for(uint j=0;threadIdx.x+j*blockDim.x<BLOCK_SIZE_Y&&(threadIdx.x+j*blockDim.x+BLOCK_SIZE_Y*blockIdx.y)<rtn;j++){
      tignv[threadIdx.x + j*blockDim.x] = ignv[threadIdx.x + j*blockDim.x + BLOCK_SIZE_Y * blockIdx.y];
    }
    __syncthreads();
    */
    //GNValue temp;
    /*
    __shared__ GNValue tignv[BLOCK_SIZE_Y];
    for(uint j=threadIdx.x ; j<BLOCK_SIZE_Y && BLOCK_SIZE_Y*blockIdx.y+j < rtn ; j += blockDim.x){
      tignv[j] = ignv[BLOCK_SIZE_Y*blockIdx.y+j];
      //tignv[j] = BLOCK_SIZE_Y*blockIdx.y+j < rtn ? ignv[BLOCK_SIZE_Y*blockIdx.y+j] : temp;
      if(blockIdx.x == 1 && blockIdx.y == 511){
        printf("%d %d %d\n",threadIdx.x,j,BLOCK_SIZE_Y*blockIdx.y+j);
      }
    }
    */

    __shared__ GNValue tignv[BLOCK_SIZE_Y];
    if(threadIdx.x==0){
      for(uint j=0 ; j<BLOCK_SIZE_Y && BLOCK_SIZE_Y*blockIdx.y+j<rtn ; j++){
        tignv[j] = ignv[BLOCK_SIZE_Y*blockIdx.y + j];
      }
    }

    __syncthreads();

    GNValue tognv=ognv[i];
    int rtn_g = rtn;
    int mcount = 0;
    for(uint j = 0; j<BLOCK_SIZE_Y && BLOCK_SIZE_Y*blockIdx.y+j<rtn_g;j++){      
      if(ex.eval(tognv,tignv[j])) {
      //      if(ex.eval(tognv,ignv[j+BLOCK_SIZE_Y*blockIdx.y])) {
        mcount++;
      }     
    }

    count[i+k] = mcount;
  }

  if(i+k == (blockDim.x*gridDim.x*gridDim.y-1)){
    count[i+k+1] = 0;
  }

}


__global__ void join(
          GNValue *ognv,
          GNValue *ignv,
          RESULT *p,
          GComparisonExpression ex,
          int *count,
          int ltn,
          int rtn,
          int ll,
          int rr
          ) 
{

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * gridDim.x * blockDim.x;

  if(i<ltn){
    /*
    __shared__ GNValue tignv[BLOCK_SIZE_Y];
    for(uint j=0;threadIdx.x+j*blockDim.x<BLOCK_SIZE_Y&&(threadIdx.x+j*blockDim.x+BLOCK_SIZE_Y*blockIdx.y)<rtn;j++){
      tignv[threadIdx.x + j*blockDim.x] = ignv[threadIdx.x + j*blockDim.x + BLOCK_SIZE_Y * blockIdx.y];
    }
    __syncthreads();
    */
    __shared__ GNValue tignv[BLOCK_SIZE_Y];
    if(threadIdx.x==0){
      for(uint j=0 ; j<BLOCK_SIZE_Y && BLOCK_SIZE_Y*blockIdx.y+j<rtn ; j++){
        tignv[j] = ignv[BLOCK_SIZE_Y*blockIdx.y + j];
      }
    }

    //GNValue temp;
    /*
    __shared__ GNValue tignv[BLOCK_SIZE_Y];

    for(uint j=threadIdx.x ; j<BLOCK_SIZE_Y ; j += blockDim.x){
      tignv[j] = ignv[BLOCK_SIZE_Y*blockIdx.y+j];
      //      tignv[j] = BLOCK_SIZE_Y*blockIdx.y+j < rtn ? ignv[BLOCK_SIZE_Y*blockIdx.y+j] : temp;
    }
    */

    __syncthreads();

    GNValue tognv = ognv[i];
    int rtn_g = rtn;
    int writeloc = count[i+k];
    for(uint j = 0; j<BLOCK_SIZE_Y && BLOCK_SIZE_Y*blockIdx.y+j<rtn_g;j++){
      //      if(ex.eval(tognv,ignv[j+BLOCK_SIZE_Y*blockIdx.y])){
      if(ex.eval(tognv,tignv[j])){
        p[writeloc].lkey = i+ll;
        p[writeloc].rkey = blockIdx.y*BLOCK_SIZE_Y+j+rr;
        writeloc++;
      }
    }
  }     
}

}
