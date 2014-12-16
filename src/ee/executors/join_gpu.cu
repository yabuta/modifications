#include <stdio.h>
#include <stdint.h>
#include <cuda.h>
#include <sys/time.h>
#include "GPUTUPLE.h"
#include "GPUetc/common/GNValue.h"
#include "GPUetc/expressions/Gcomparisonexpression.h"

using namespace voltdb;

/**
count() is counting match tuple.
And in CPU, caluculate starting position using scan.
finally join() store match tuple to result array .

*/



extern "C" {


__global__
void count(
          COLUMNDATA *oCD,
          COLUMNDATA *iCD,
          GComparisonExpression ex,
          ulong *count,
          int ltn,
          int rtn
          ) 

{

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * gridDim.x * blockDim.x;

  __shared__ COLUMNDATA tiCD[BLOCK_SIZE_Y];

  /**
     TO DO : tiCD shoud be stored in parallel.but I have bug.
   */
  if(threadIdx.x==0){
    for(uint j=0 ; j<BLOCK_SIZE_Y && BLOCK_SIZE_Y*blockIdx.y+j<rtn ; j++){
      tiCD[j] = iCD[BLOCK_SIZE_Y*blockIdx.y + j];
    }
  }
  __syncthreads();

  count[i+k] = 0;

  if(i<ltn){


    //A global memory read is very slow.So repeating values is stored register memory
    COLUMNDATA toCD=oCD[i];
    int rtn_g = rtn;
    int mcount = 0;
    for(uint j = 0; j<BLOCK_SIZE_Y && BLOCK_SIZE_Y*blockIdx.y+j<rtn_g;j++){
      if(ex.eval(toCD.gn,tiCD[j].gn)){
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
          COLUMNDATA *oCD,
          COLUMNDATA *iCD,
          RESULT *p,
          GComparisonExpression ex,
          ulong *count,
          int ltn,
          int rtn
          ) 
{

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * gridDim.x * blockDim.x;

  __shared__ COLUMNDATA tiCD[BLOCK_SIZE_Y];
  if(threadIdx.x==0){
    for(uint j=0 ; j<BLOCK_SIZE_Y && BLOCK_SIZE_Y*blockIdx.y+j<rtn ; j++){
      tiCD[j] = iCD[BLOCK_SIZE_Y*blockIdx.y + j];
    }
  }
  __syncthreads();

  if(i<ltn){

    COLUMNDATA toCD = oCD[i];
    int rtn_g = rtn;
    ulong writeloc = count[i+k];
    for(uint j = 0; j<BLOCK_SIZE_Y && BLOCK_SIZE_Y*blockIdx.y+j<rtn_g;j++){
      if(ex.eval(toCD.gn,tiCD[j].gn)){
        p[writeloc].lkey = toCD.num;
        p[writeloc].rkey = tiCD[j].num;
        writeloc++;
      }
    }
  }     
}

}
