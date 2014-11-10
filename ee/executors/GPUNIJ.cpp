#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>
#include <error.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include "GPUTUPLE.h"
#include "GPUNIJ.h"
#include "scan_common.h"
#include "common/types.h"
#include "GPUetc/common/GNValue.h"
#include "GPUetc/expressions/Gcomparisonexpression.h"


using namespace voltdb;

GPUNIJ::GPUNIJ(){

  jt = NULL;
  total = 0;
  PART = 524288;

  char fname[256];
  const char *path="/home/yabuta/voltdb/voltdb";
  
  /******************** GPU init here ************************************************/
  //GPU仕様のために

  res = cuInit(0);
  if (res != CUDA_SUCCESS) {
    printf("cuInit failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }
  res = cuDeviceGet(&dev, 0);
  if (res != CUDA_SUCCESS) {
    printf("cuDeviceGet failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }
  res = cuCtxCreate(&ctx, 0, dev);
  if (res != CUDA_SUCCESS) {
    printf("cuCtxCreate failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }

  /*********************************************************************************/


  /*
   *指定したファイルからモジュールをロードする。これが平行実行されると思っていいもかな？
   *今回はjoin_gpu.cubinとcountJoinTuple.cubinの二つの関数を実行する
   */

  
  sprintf(fname, "%s/join_gpu.cubin", path);
  res = cuModuleLoad(&module, fname);
  if (res != CUDA_SUCCESS) {
    printf("cuModuleLoad(join) failed\n");
    exit(1);
  }
  res = cuModuleGetFunction(&iifunction, module, "iijoin");
  if (res != CUDA_SUCCESS) {
    printf("cuModuleGetFunction(iijoin) failed\n");
    exit(1);
  }
  res = cuModuleGetFunction(&iic_function, module, "iicount");
  if (res != CUDA_SUCCESS) {
    printf("cuModuleGetFunction(iicount) failed\n");
    exit(1);
  }
  res = cuModuleGetFunction(&oifunction, module, "oijoin");
  if (res != CUDA_SUCCESS) {
    printf("cuModuleGetFunction(oijoin) failed\n");
    exit(1);
  }
  res = cuModuleGetFunction(&oic_function, module, "oicount");
  if (res != CUDA_SUCCESS) {
    printf("cuModuleGetFunction(oicount) failed\n");
    exit(1);
  }
}


GPUNIJ::~GPUNIJ(){

  free(jt);
  free(left_GNV);
  free(right_GNV);

  //finish GPU   ****************************************************

  res = cuModuleUnload(module);
  if (res != CUDA_SUCCESS) {
    printf("cuModuleUnload module failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }  
  res = cuCtxDestroy(ctx);
  if (res != CUDA_SUCCESS) {
    printf("cuCtxDestroy failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }
}


void
GPUNIJ::printDiff(struct timeval begin, struct timeval end)
{
  long diff;
  
  diff = (end.tv_sec - begin.tv_sec) * 1000 * 1000 + (end.tv_usec - begin.tv_usec);
  printf("Diff: %ld us (%ld ms)\n", diff, diff/1000);
}

uint GPUNIJ::iDivUp(uint dividend, uint divisor)
{
  return ((dividend % divisor) == 0) ? (dividend / divisor) : (dividend / divisor + 1);
}



//HrightとHleftをそれぞれ比較する。GPUで並列化するforループもここにあるもので行う。
void GPUNIJ::join()
{

  //int i, j;
  uint jt_size,gpu_size;
  CUdeviceptr lt_dev, rt_dev, jt_dev,count_dev, pre_dev;
  CUdeviceptr ltn_dev, rtn_dev;
  unsigned int block_x, block_y, grid_x, grid_y;


  /************** block_x * block_y is decided by BLOCK_SIZE. **************/

  block_x = BLOCK_SIZE_X;
  block_y = BLOCK_SIZE_Y;
    grid_x = PART / block_x;
  if (PART % block_x != 0)
    grid_x++;
  grid_y = PART / block_y;
  if (PART % block_y != 0)
    grid_y++;
  block_y = 1;

  gpu_size = grid_x * grid_y * block_x * block_y;
  if(gpu_size>MAX_LARGE_ARRAY_SIZE){
    gpu_size = MAX_LARGE_ARRAY_SIZE * iDivUp(gpu_size,MAX_LARGE_ARRAY_SIZE);
  }else if(gpu_size > MAX_SHORT_ARRAY_SIZE){
    gpu_size = MAX_SHORT_ARRAY_SIZE * iDivUp(gpu_size,MAX_SHORT_ARRAY_SIZE);
  }else{
    gpu_size = MAX_SHORT_ARRAY_SIZE;
  }


  /********************************************************************************/

  res = cuMemAlloc(&lt_dev, PART * sizeof(GNValue));
  if (res != CUDA_SUCCESS) {
    printf("cuMemAlloc (lefttuple) failed\n");
    exit(1);
  }
  res = cuMemAlloc(&rt_dev, PART * sizeof(GNValue));
  if (res != CUDA_SUCCESS) {
    printf("cuMemAlloc (righttuple) failed\n");
    exit(1);
  }
  res = cuMemAlloc(&count_dev, gpu_size * sizeof(int));
  if (res != CUDA_SUCCESS) {
    printf("cuMemAlloc (count) failed\n");
    exit(1);
  }

  /********************** upload lt , rt and count***********************/

  for(uint ll = 0; ll < left ; ll += PART){
    for(uint rr = 0; rr < right ; rr += PART){

      uint lls=PART,rrs=PART;
      if((ll+PART) >= left){
        lls = left - ll;
      }
      if((rr+PART) >= right){
        rrs = right - rr;
      }

      block_x = lls < BLOCK_SIZE_X ? lls : BLOCK_SIZE_X;
      block_y = rrs < BLOCK_SIZE_Y ? rrs : BLOCK_SIZE_Y;      
      grid_x = lls / block_x;
      if (lls % block_x != 0)
        grid_x++;
      grid_y = rrs / block_y;
      if (rrs % block_y != 0)
        grid_y++;
      block_y = 1;

      printf("\nStarting...\nll = %d\trr = %d\tlls = %d\trrs = %d\n",ll,rr,lls,rrs);
      printf("grid_x = %d\tgrid_y = %d\tblock_x = %d\tblock_y = %d\n",grid_x,grid_y,block_x,block_y);
      gpu_size = grid_x * grid_y * block_x * block_y+1;
      printf("gpu_size = %d\n",gpu_size);


      res = cuMemcpyHtoD(lt_dev, &(left_GNV[ll]), lls * sizeof(GNValue));
      if (res != CUDA_SUCCESS) {
        printf("cuMemcpyHtoD (lt) failed: res = %lu\n", res);//conv(res));
        exit(1);
      }
      res = cuMemcpyHtoD(rt_dev, &(right_GNV[rr]), rrs * sizeof(GNValue));
      if (res != CUDA_SUCCESS) {
        printf("cuMemcpyHtoD (rt) failed: res = %lu\n", (unsigned long)res);
        exit(1);
      }

      void *count_args[]={    
        (void *)&lt_dev,
        (void *)&rt_dev,
        (void *)expression,
        (void *)&count_dev,
        (void *)&lls,
        (void *)&rrs        
      };


      if(conditionflag = II){
        res = cuLaunchKernel(
                             iic_function,    // CUfunction f
                             grid_x,        // gridDimX
                             grid_y,        // gridDimY
                             1,             // gridDimZ
                             block_x,       // blockDimX
                             block_y,       // blockDimY
                             1,             // blockDimZ
                             0,             // sharedMemBytes
                             NULL,          // hStream
                             count_args,   // keunelParams
                             NULL           // extra
                             );
        if(res != CUDA_SUCCESS) {
          printf("cuLaunchKernel(count) failed: res = %lu\n", (unsigned long int)res);
          exit(1);
        }      
      
        res = cuCtxSynchronize();
        if(res != CUDA_SUCCESS) {
          printf("cuCtxSynchronize(count) failed: res = %lu\n", (unsigned long int)res);
          exit(1);
        }  
      }else{
        res = cuLaunchKernel(
                             oic_function,    // CUfunction f
                             grid_x,        // gridDimX
                             grid_y,        // gridDimY
                             1,             // gridDimZ
                             block_x,       // blockDimX
                             block_y,       // blockDimY
                             1,             // blockDimZ
                             0,             // sharedMemBytes
                             NULL,          // hStream
                             count_args,   // keunelParams
                             NULL           // extra
                             );
        if(res != CUDA_SUCCESS) {
          printf("cuLaunchKernel(count) failed: res = %lu\n", (unsigned long int)res);
          exit(1);
        }      
      
        res = cuCtxSynchronize();
        if(res != CUDA_SUCCESS) {
          printf("cuCtxSynchronize(count) failed: res = %lu\n", (unsigned long int)res);
          exit(1);
        }  

      }

      /**************************** prefix sum *************************************/
      if(!(presum(&count_dev,(uint)gpu_size))){
        printf("count scan error.\n");
        exit(1);
      }
      /********************************************************************/      

      if(!transport(count_dev,(uint)gpu_size,&jt_size)){
        printf("transport error.\n");
        exit(1);
      }


      /************************************************************************
      jt memory alloc and jt upload

      ************************************************************************/

      if(jt_size<=0){
        total += jt_size;
        jt_size = 0;
      }else{
        jt = (RESULT *)malloc(jt_size*sizeof(RESULT));
        res = cuMemAlloc(&jt_dev, jt_size*sizeof(RESULT));
        if (res != CUDA_SUCCESS) {
          printf("cuMemAlloc (join) failed\n");
          exit(1);
        }      
        
        void *kernel_args[]={
          (void *)&lt_dev,
          (void *)&rt_dev,
          (void *)&jt_dev,
          (void *)expression,
          (void *)&count_dev,
          (void *)&lls,
          (void *)&rrs,    
        };

        if(conditionflag = II){
          res = cuLaunchKernel(
                               iifunction,      // CUfunction f
                               grid_x,        // gridDimX
                               grid_y,        // gridDimY
                               1,             // gridDimZ
                               block_x,       // blockDimX
                               block_y,       // blockDimY
                               1,             // blockDimZ
                               0,             // sharedMemBytes
                               NULL,          // hStream
                               kernel_args,   // keunelParams
                               NULL           // extra
                               );
          if(res != CUDA_SUCCESS) {
            printf("cuLaunchKernel() failed: res = %lu\n", (unsigned long int)res);
            exit(1);
          }  
        
          res = cuCtxSynchronize();
          if(res != CUDA_SUCCESS) {
            printf("cuCtxSynchronize() failed: res = %lu\n", (unsigned long int)res);
            exit(1);
          }  
        }else{
          res = cuLaunchKernel(
                               oifunction,      // CUfunction f
                               grid_x,        // gridDimX
                               grid_y,        // gridDimY
                               1,             // gridDimZ
                               block_x,       // blockDimX
                               block_y,       // blockDimY
                               1,             // blockDimZ
                               0,             // sharedMemBytes
                               NULL,          // hStream
                               kernel_args,   // keunelParams
                               NULL           // extra
                               );
          if(res != CUDA_SUCCESS) {
            printf("cuLaunchKernel() failed: res = %lu\n", (unsigned long int)res);
            exit(1);
          }  
        
          res = cuCtxSynchronize();
          if(res != CUDA_SUCCESS) {
            printf("cuCtxSynchronize() failed: res = %lu\n", (unsigned long int)res);
            exit(1);
          }  

        }
        res = cuMemcpyDtoH(&(jt[total]), jt_dev, jt_size * sizeof(RESULT));
        if (res != CUDA_SUCCESS) {
          printf("cuMemcpyDtoH (jt) failed: res = %lu\n", (unsigned long)res);
          exit(1);
        }
        cuMemFree(jt_dev);
        total += jt_size;
        printf("End...\n jt_size = %d\ttotal = %d\n",jt_size,total);
        jt_size = 0;
        
      }
    }
    

  }

  /***************************************************************/

  //free GPU memory***********************************************


  res = cuMemFree(lt_dev);
  if (res != CUDA_SUCCESS) {
    printf("cuMemFree (lt) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }
  res = cuMemFree(rt_dev);
  if (res != CUDA_SUCCESS) {
    printf("cuMemFree (rt) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }
  res = cuMemFree(count_dev);
  if (res != CUDA_SUCCESS) {
    printf("cuMemFree (count) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }

}


