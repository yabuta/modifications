/*
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#ifndef SCAN_COMMON_H
#define SCAN_COMMON_H

#include <stdlib.h>
#include <cuda.h>

/*
namespace voltdb{

class GPUScan{

public:

    GPUScan();

////////////////////////////////////////////////////////////////////////////////
// Shortcut typename
////////////////////////////////////////////////////////////////////////////////
#define SUCCESS 1
#define FALSE 0


////////////////////////////////////////////////////////////////////////////////
// Shortcut typename
////////////////////////////////////////////////////////////////////////////////
    typedef unsigned int uint;

////////////////////////////////////////////////////////////////////////////////
// Implementation limits
////////////////////////////////////////////////////////////////////////////////
    uint MAX_BATCH_ELEMENTS;
    uint MIN_SHORT_ARRAY_SIZE;
    uint MAX_SHORT_ARRAY_SIZE;
    uint MIN_LARGE_ARRAY_SIZE;
    uint MAX_LARGE_ARRAY_SIZE;
    uint MAX_LL_SIZE;
    uint MIN_LL_SIZE;
    
////////////////////////////////////////////////////////////////////////////////
// CUDA scan
////////////////////////////////////////////////////////////////////////////////
    void initScan(void);
    void closeScan(void);
    
    size_t scanExclusiveShort(
        uint *d_Dst,
        uint *d_Src,
        uint arrayLength
        );

    size_t scanExclusiveLarge(
        uint *d_Dst,
        uint *d_Src,
        uint arrayLength
        );

    size_t scanExclusiveLL(
        uint *d_Dst,
        uint *d_Src,
        uint arrayLength
        );

    size_t diff_Part(
        uint *d_Dst,
        uint *d_Src,
        uint diff,
        uint arrayLength,
        uint size
        );

    void transport_gpu(
        uint *d_Dst,
        uint *d_Src,
        uint diff,
        uint arrayLength,
        uint size
        );


    void getValue_gpu(
        uint *d_Dst,
        uint *d_Src,
        uint loc
        );

    uint presum(
        CUdeviceptr *d_Input,
        uint arrayLength
        );

    CUdeviceptr diff_part(
        CUdeviceptr d_Input,
        uint tnum,
        uint arrayLength,
        uint size
        );

    CUdeviceptr transport(
        CUdeviceptr d_Input,
        uint tnum,
        uint arrayLength,
        uint size
        );

    uint getValue(
        CUdeviceptr d_Input,
        uint tnum,
        uint *res
        );

};

}

#endif
*/


////////////////////////////////////////////////////////////////////////////////
// Shortcut typename
////////////////////////////////////////////////////////////////////////////////
#define SUCCESS 1
#define FALSE 0


////////////////////////////////////////////////////////////////////////////////
// Shortcut typename
////////////////////////////////////////////////////////////////////////////////
//    typedef unsigned int uint;

////////////////////////////////////////////////////////////////////////////////
// Implementation limits
////////////////////////////////////////////////////////////////////////////////
    extern "C" const uint MAX_BATCH_ELEMENTS;
    extern "C" const uint MIN_SHORT_ARRAY_SIZE;
    extern "C" const uint MAX_SHORT_ARRAY_SIZE;
    extern "C" const uint MIN_LARGE_ARRAY_SIZE;
    extern "C" const uint MAX_LARGE_ARRAY_SIZE;
    extern "C" const uint MAX_LL_SIZE;
    extern "C" const uint MIN_LL_SIZE;
    
////////////////////////////////////////////////////////////////////////////////
// CUDA scan
////////////////////////////////////////////////////////////////////////////////
    extern "C" void initScan(void);
    extern "C" void closeScan(void);
    
    extern "C" size_t scanExclusiveShort(
        uint *d_Dst,
        uint *d_Src,
        uint arrayLength
        );

    extern "C" size_t scanExclusiveLarge(
        uint *d_Dst,
        uint *d_Src,
        uint arrayLength
        );

    extern "C" size_t scanExclusiveLL(
        uint *d_Dst,
        uint *d_Src,
        uint arrayLength
        );

    extern "C" size_t diff_Part(
        uint *d_Dst,
        uint *d_Src,
        uint diff,
        uint arrayLength,
        uint size
        );

    extern "C" void transport_gpu(
        uint *d_Dst,
        uint *d_Src,
        uint loc
        );

    extern "C" uint presum(
        CUdeviceptr *d_Input,
        uint arrayLength
        );

    extern "C" CUdeviceptr diff_part(
        CUdeviceptr d_Input,
        uint tnum,
        uint arrayLength,
        uint size
        );

    extern "C" uint transport(
        CUdeviceptr d_Input,
        uint loc,
        uint *res
        );


#endif
