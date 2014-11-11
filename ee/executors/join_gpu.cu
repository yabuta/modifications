#include <stdio.h>
#include <stdint.h>
#include <cuda.h>
#include <sys/time.h>
#include "GPUTUPLE.h"
#include "GPUetc/common/GNValue.h"
#include "GPUetc/expressions/Gcomparisonexpression.h"

using namespace voltdb;

//#define DISTANCE 1

extern "C" {


  /**
     called function is changed by join condition.
     
     if T1.val = T2.val, iocount and iojoin is called.
     if T.val1 = T.val2 , iicount and iijoin is called.
   */


__global__
void iicount(
          GNValue *ignv1,
          GNValue *ignv2,
          GComparisonExpression ex,
          int *count,
          int ltn,
          int rtn
          ) 

{

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * gridDim.x * blockDim.x;

  if(i<ltn){
    __shared__ GNValue tignv1[BLOCK_SIZE_Y];
    for(uint j=0;threadIdx.x+j*blockDim.x<BLOCK_SIZE_Y&&(threadIdx.x+j*blockDim.x+BLOCK_SIZE_Y*blockIdx.y)<rtn;j++){
      tignv1[threadIdx.x + j*blockDim.x] = ignv1[threadIdx.x + j*blockDim.x + BLOCK_SIZE_Y * blockIdx.y];
    }
    __shared__ GNValue tignv2[BLOCK_SIZE_Y];
    for(uint j=0;threadIdx.x+j*blockDim.x<BLOCK_SIZE_Y&&(threadIdx.x+j*blockDim.x+BLOCK_SIZE_Y*blockIdx.y)<rtn;j++){
      tignv2[threadIdx.x + j*blockDim.x] = ignv2[threadIdx.x + j*blockDim.x + BLOCK_SIZE_Y * blockIdx.y];
    }    
    __syncthreads();  
    int rtn_g = rtn;
    uint mcount = 0;
    for(uint j = 0; j<BLOCK_SIZE_Y &&((j+BLOCK_SIZE_Y*blockIdx.y)<rtn_g);j++){
      if(ex.eval(tignv1[j],tignv2[j])) {
        mcount++;
      }
    }
    count[i+k] = mcount;
  }

  
  if(i+k==blockDim.x*gridDim.x*gridDim.y-1){
    count[i+k+1]=0;
  }

}


__global__ void iijoin(
          GNValue *ignv1,
          GNValue *ignv2,
          RESULT *p,
          GComparisonExpression ex,
          int *count,
          int ltn,
          int rtn
          ) 
{

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * gridDim.x * blockDim.x;

  if(i<ltn){
    __shared__ GNValue tignv1[BLOCK_SIZE_Y];
    for(uint j=0;threadIdx.x+j*blockDim.x<BLOCK_SIZE_Y&&(threadIdx.x+j*blockDim.x+BLOCK_SIZE_Y*blockIdx.y)<rtn;j++){
      tignv1[threadIdx.x + j*blockDim.x] = ignv1[threadIdx.x + j*blockDim.x + BLOCK_SIZE_Y * blockIdx.y];
    }
    __shared__ GNValue tignv2[BLOCK_SIZE_Y];
    for(uint j=0;threadIdx.x+j*blockDim.x<BLOCK_SIZE_Y&&(threadIdx.x+j*blockDim.x+BLOCK_SIZE_Y*blockIdx.y)<rtn;j++){
      tignv2[threadIdx.x + j*blockDim.x] = ignv2[threadIdx.x + j*blockDim.x + BLOCK_SIZE_Y * blockIdx.y];
    }    
    __syncthreads();  
    int rtn_g = rtn;
    int writeloc = count[i+blockIdx.y*blockDim.x*gridDim.x];
    for(uint j = 0; j<BLOCK_SIZE_Y &&((j+BLOCK_SIZE_Y*blockIdx.y)<rtn_g);j++){
      if(ex.eval(tignv1[j],tignv2[j])) {
        p[writeloc].lkey = i;
        p[writeloc].rkey = blockIdx.y*BLOCK_SIZE_Y+j;
        writeloc++;
      }
    }
  }     
}    


__global__
void oicount(
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
    __shared__ GNValue tignv[BLOCK_SIZE_Y];
    for(uint j=0;threadIdx.x+j*blockDim.x<BLOCK_SIZE_Y&&(threadIdx.x+j*blockDim.x+BLOCK_SIZE_Y*blockIdx.y)<rtn;j++){
      tignv[threadIdx.x + j*blockDim.x] = ignv[threadIdx.x + j*blockDim.x + BLOCK_SIZE_Y * blockIdx.y];
    }
    __syncthreads();
    GNValue tognv=ognv[i];
    int rtn_g = rtn;
    int mcount = 0;
    for(uint j = 0; j<BLOCK_SIZE_Y &&((j+BLOCK_SIZE_Y*blockIdx.y)<rtn_g);j++){
      if(ex.eval(tognv,tignv[j])) {
        mcount++;
      }
    }
    count[i+k] = mcount;
  }

}


__global__ void oijoin(
          GNValue *ognv,
          GNValue *ignv,
          RESULT *p,
          GComparisonExpression ex,
          int *count,
          int ltn,
          int rtn
          ) 
{

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * gridDim.x * blockDim.x;

  if(i<ltn){
    __shared__ GNValue tignv[BLOCK_SIZE_Y];
    for(uint j=0;threadIdx.x+j*blockDim.x<BLOCK_SIZE_Y&&(threadIdx.x+j*blockDim.x+BLOCK_SIZE_Y*blockIdx.y)<rtn;j++){
      tignv[threadIdx.x + j*blockDim.x] = ignv[threadIdx.x + j*blockDim.x + BLOCK_SIZE_Y * blockIdx.y];
    }
    __syncthreads();
    GNValue tognv = ognv[i];
    int rtn_g = rtn;
    int writeloc = count[i+k];
    for(uint j = 0; j<BLOCK_SIZE_Y &&((j+BLOCK_SIZE_Y*blockIdx.y)<rtn_g);j++){
      if(ex.eval(tognv,tignv[j])){
        p[writeloc].lkey = i;
        p[writeloc].rkey = blockIdx.y*BLOCK_SIZE_Y+j;
        if(p[writeloc].lkey == 0){
          printf("%d %d %d\n",writeloc,p[writeloc].lkey,p[writeloc].rkey);
        }
        writeloc++;
      }
    }
  }     
}

}
