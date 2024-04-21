/* Copyright 2019 Inspur Corporation. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
    
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef __GOOGLENET__
#define __GOOGLENET__

//---------------------------------------------------------------------//
//                                                                     //
//              DEVICE/HOST COMMON GOOGLENET PARAMETERS                //
//                                                                     //
//---------------------------------------------------------------------//

// Note: This file will be automatically generated by programs rather than configured mannually in the future.

//
// Debug Parameters
//

//#define CONCAT_LAYER_DEBUG

#define STATIC_CYCLE
//#define PRINT_N 1
//#define PRINT_CYCLE
//#define PRINT_SEQUENCER_INDEX
//#define PRINT_IPOOL_INPUT
//#define PRINT_PE_INPUT
//#define PRINT_PE_OUTPUT
//#define PRINT_POOL_INPUT
//#define PRINT_POOL_OUTPUT

//
// Configuration Parameters
//

#define NUM_LAYER 67
#define NUM_CONVOLUTIONS 67
#define NUM_Q_LAYERS (NUM_CONVOLUTIONS + 1 + 9) // 1 is for input data Quantization value, 9 is for concatenate layer, their qs are stored seperately.

#define INPUT_IMAGE_C 3
#define INPUT_IMAGE_H 224
#define INPUT_IMAGE_W 224
#define FIRST_FILTER_SIZE 7

#define MAX_OUT_CHANNEL 1024
#define MAX_POOL_OUTPUT_WVEC CEIL(56, W_VECTOR)

// the maximum pool window size
#define POOL_WINDOW_MAX 3

// set size of feature map DDR
#define DDR_PAGE_SIZE0 (CEIL(480, C_VECTOR) * 56 * CEIL(56, W_VECTOR))
#define DDR_PAGE_SIZE1 (CEIL(256, C_VECTOR) * 56 * CEIL(56, W_VECTOR))
#define DDR_SIZE (DDR_PAGE_SIZE0 + DDR_PAGE_SIZE1)

// set size of feature map cache
#define CACHE_PAGE_SIZE (CEIL(27, C_VECTOR) * 114 * CEIL(114, W_VECTOR)) // single buffer size: be calculated from the layer of max slice size
#define CACHE_SIZE (CACHE_PAGE_SIZE * 3)

// set size of filter buffer
#define FILTER_CACHE_PAGE_SIZE1 (NEXT_DIVISIBLE(192, C_VECTOR) * 3 * CEIL(3, FW_VECTOR))
#define FILTER_CACHE_PAGE_SIZE2 (NEXT_DIVISIBLE(1024, C_VECTOR * FW_VECTOR))
#define FILTER_CACHE_PAGE_SIZE  (MYMAX2(FILTER_CACHE_PAGE_SIZE1, FILTER_CACHE_PAGE_SIZE2))

#define FILTER_CACHE_PAGE_DEPTH (NEXT_POWER_OF_2(CEIL(FILTER_CACHE_PAGE_SIZE, C_VECTOR))) // size of every cache is FW_VECTOR * C_VECTOR
#define FILTER_CACHE_DEPTH (FILTER_CACHE_PAGE_DEPTH * DOUBLE_BUFFER_DIM)

// read filter data from ddr with this many cycle intervals in order not to
// get stalled because of ddr bandwidth bottleneck
// N_VECTOR kernels are each reading in round-robin order C_VECTOR*FW_VECTOR floats.
// We want to wait ceil( (C_VECTOR * FW_VECTOR) / DDR_BANDWIDTH_IN_FLOATS ) cycles between each ddr access from kernels
#define FILTER_DDR_READ_STEP1 (CEIL((C_VECTOR * FW_VECTOR), DDR_BANDWIDTH_IN_BYTES))
#define FILTER_DDR_READ_STEP2 (CEIL((C_VECTOR * 1), DDR_BANDWIDTH_IN_BYTES))

#define FEATURE_DDR_READ_STEP 1//( CEIL((C_VECTOR * W_VECTOR), DDR_BANDWIDTH_IN_BYTES) )

// Set size of host filter and bias buffer of each layer.
#define MAX_FILTER_SIZE1 (NEXT_POWER_OF_2(CEIL(2048, C_VECTOR) * 1 * CEIL(1, FW_VECTOR) * NEXT_DIVISIBLE(2048, N_VECTOR) * NEXT_POWER_OF_2(FW_VECTOR * C_VECTOR)))
#define MAX_FILTER_SIZE2 (NEXT_POWER_OF_2(CEIL(1024, C_VECTOR) * 3 * CEIL(3, FW_VECTOR) * NEXT_DIVISIBLE(1024, N_VECTOR) * NEXT_POWER_OF_2(FW_VECTOR * C_VECTOR)))
#define MAX_FILTER_SIZE_TEMP ((MAX_FILTER_SIZE1 > MAX_FILTER_SIZE2) ? MAX_FILTER_SIZE1 : MAX_FILTER_SIZE2)
#define MAX_FILTER_SIZE (CEIL(MAX_FILTER_SIZE_TEMP, NEXT_POWER_OF_2(FW_VECTOR * C_VECTOR)))

#define MAX_BIAS_SIZE NEXT_DIVISIBLE(2048, N_VECTOR)
/*
// used by pool.cl
#define EDGE_H (POOL_WINDOW_MAX - 1)
#define EDGE_W (POOL_WINDOW_MAX - 1)
#define WVEC_ITER (CEIL(kOwEndWithOffsetMax, OW_VECTOR))
#define NNVEC_ITER (CEIL(N_VECTOR, NARROW_N_VECTOR))
#define EDGE_H_BUFFER_SIZE (WVEC_ITER * NNVEC_ITER)
#define EDGE_W_BUFFER_SIZE (NNVEC_ITER)
*/
#define DDR_BLOCK_SIZE DDR_PAGE_SIZE0
#define D0 0
#define D1 0
#define D2 DDR_BLOCK_SIZE

#define OUTPUT_OFFSET (3 * DDR_BLOCK_SIZE * NEXT_POWER_OF_2(W_VECTOR * NARROW_N_VECTOR))

#define C1 0
#define C2 CACHE_PAGE_SIZE
#define C3 (2 * CACHE_PAGE_SIZE)

//
// Convolution Parametres of each layer
//

CONSTANT int kCacheReadBase[NUM_CONVOLUTIONS] = {
  C1,         
  C2,         
  C1, 
  C2, C2, C3, C2, C3, C2, C3, 
  C1, C1, C3, C1, C3, C1, C3, 
  C2, C2, C3, C2, C3, C2, C3, 
  C1, C1, C3, C1, C3, C1, C3, 
  C2, C2, C3, C2, C3, C2, C3, 
  C1, C1, C3, C1, C3, C1, C3, 
  C2, C2, C3, C2, C3, C2, C3, 
  C1, C1, C3, C1, C3, C1, C3, 
  C2, C2, C3, C2, C3, C2, C3,
  C1
};

CONSTANT int kCacheWriteBase[NUM_CONVOLUTIONS] = {
  C2,         
  C1,         
  C2, 
  C1, C3, C1, C3, C1, C3, C1, 
  C2, C3, C2, C3, C2, C3, C2,
  C1, C3, C1, C3, C1, C3, C1, 
  C2, C3, C2, C3, C2, C3, C2,
  C1, C3, C1, C3, C1, C3, C1, 
  C2, C3, C2, C3, C2, C3, C2,
  C1, C3, C1, C3, C1, C3, C1, 
  C2, C3, C2, C3, C2, C3, C2,
  C1, C3, C1, C3, C1, C3, C1,
  C2
};

CONSTANT int kDDRReadBase[NUM_CONVOLUTIONS] = {
  D0,         
  D0,         
  D0, 
  D0, D0, D0, D0, D0, D0, D0, 
  D0, D0, D0, D0, D0, D0, D0, 
  D0, D0, D0, D0, D0, D0, D0, 
  D0, D0, D0, D0, D0, D0, D0, 
  D0, D0, D0, D0, D0, D0, D0, 
  D0, D0, D0, D0, D0, D0, D0, 
  D0, D0, D0, D0, D0, D0, D0, 
  D0, D0, D0, D0, D0, D0, D0, 
  D0, D0, D0, D0, D0, D0, D0, 
  D0
};

CONSTANT int kDDRWriteBase[NUM_CONVOLUTIONS] = {
  D0,         
  D0,         
  D0, 
  D0, D0, D0, D0, D0, D0, D0, 
  D0, D0, D0, D0, D0, D0, D0, 
  D0, D0, D0, D0, D0, D0, D0, 
  D0, D0, D0, D0, D0, D0, D0, 
  D0, D0, D0, D0, D0, D0, D0, 
  D0, D0, D0, D0, D0, D0, D0, 
  D0, D0, D0, D0, D0, D0, D0, 
  D0, D0, D0, D0, D0, D0, D0, 
  D0, D0, D0, D0, D0, D0, D0, 
  D0
};

CONSTANT bool kCacheWriteEnable[NUM_CONVOLUTIONS] = {
  1,      
  1,      
  1, 
  1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1,
  0
};

#ifdef CONCAT_LAYER_DEBUG

CONSTANT bool kDDRWriteEnable[NUM_CONVOLUTIONS] = {
  0,       
  0,       
  0, 
  0, 0, 0, 0, 0, 0, 0, 
  1, 0, 1, 0, 1, 0, 1, 
  1, 0, 1, 0, 1, 0, 1, 
  1, 0, 1, 0, 1, 0, 1, 
  1, 0, 1, 0, 1, 0, 1, 
  1, 0, 1, 0, 1, 0, 1, 
  1, 0, 1, 0, 1, 0, 1, 
  1, 0, 1, 0, 1, 0, 1, 
  1, 0, 1, 0, 1, 0, 1, 
  0  
};

#else
  
CONSTANT bool kDDRWriteEnable[NUM_CONVOLUTIONS] = {
  0,       
  0,       
  0, 
  0, 0, 0, 0, 0, 0, 0, 
  0, 0, 0, 0, 0, 0, 0, 
  0, 0, 0, 0, 0, 0, 0, 
  0, 0, 0, 0, 0, 0, 0, 
  0, 0, 0, 0, 0, 0, 0, 
  0, 0, 0, 0, 0, 0, 0, 
  0, 0, 0, 0, 0, 0, 0, 
  0, 0, 0, 0, 0, 0, 0, 
  0, 0, 0, 0, 0, 0, 0,
  0
  };
  
#endif

CONSTANT bool kEndPoolEnable[NUM_CONVOLUTIONS] = {
  0,       
  0,      
  0, 
  0, 0, 0, 0, 0, 0, 0, 
  0, 0, 0, 0, 0, 0, 0, 
  0, 0, 0, 0, 0, 0, 0, 
  0, 0, 0, 0, 0, 0, 0, 
  0, 0, 0, 0, 0, 0, 0, 
  0, 0, 0, 0, 0, 0, 0, 
  0, 0, 0, 0, 0, 0, 0, 
  0, 0, 0, 0, 0, 0, 0, 
  //0, 0, 0, 0, 0, 0, 0, 
  1, 0, 1, 0, 1, 0, 1, 
  0
};

CONSTANT bool kAdditionEnable[NUM_CONVOLUTIONS] = {
  0,       
  0,       
  0, 
  0, 0, 0, 0, 0, 0, 0, 
  0, 0, 0, 0, 0, 0, 0, 
  0, 0, 0, 0, 0, 0, 0, 
  0, 0, 0, 0, 0, 0, 0, 
  0, 0, 0, 0, 0, 0, 0, 
  0, 0, 0, 0, 0, 0, 0, 
  0, 0, 0, 0, 0, 0, 0, 
  0, 0, 0, 0, 0, 0, 0, 
  0
};

CONSTANT bool kAdditionReluEnable[NUM_CONVOLUTIONS] = {
  0,       
  0,       
  0, 
  0, 0, 0, 0, 0, 0, 0, 
  0, 0, 0, 0, 0, 0, 0, 
  0, 0, 0, 0, 0, 0, 0, 
  0, 0, 0, 0, 0, 0, 0, 
  0, 0, 0, 0, 0, 0, 0, 
  0, 0, 0, 0, 0, 0, 0, 
  0, 0, 0, 0, 0, 0, 0, 
  0, 0, 0, 0, 0, 0, 0, 
  0, 0, 0, 0, 0, 0, 0, 
  0
};

CONSTANT bool kReluEnable[NUM_CONVOLUTIONS] = {
  true,             
  true,             
  true, 
  true, true, true, true, true, false, true, 
  true, true, true, true, true, false, true, 
  true, true, true, true, true, false, true, 
  true, true, true, true, true, false, true, 
  true, true, true, true, true, false, true, 
  true, true, true, true, true, false, true, 
  true, true, true, true, true, false, true, 
  true, true, true, true, true, false, true, 
  true, true, true, true, true, false, true, 
  false
};

CONSTANT int kFilterSize[NUM_CONVOLUTIONS] = {
  3, 
  1, 
  3, 
  1, 1, 3, 1, 5, 1, 1, 
  1, 1, 3, 1, 5, 1, 1, 
  1, 1, 3, 1, 5, 1, 1, 
  1, 1, 3, 1, 5, 1, 1, 
  1, 1, 3, 1, 5, 1, 1, 
  1, 1, 3, 1, 5, 1, 1, 
  1, 1, 3, 1, 5, 1, 1, 
  1, 1, 3, 1, 5, 1, 1, 
  1, 1, 3, 1, 5, 1, 1, 
  1
};
CONSTANT int kFilterSizeMax = 5;

// Conv pad
CONSTANT int kPadWidth[NUM_CONVOLUTIONS] = {
  0, 
  0, 
  1, 
  0, 0, 1, 0, 2, 0, 0, 
  0, 0, 1, 0, 2, 0, 0, 
  0, 0, 1, 0, 2, 0, 0, 
  0, 0, 1, 0, 2, 0, 0, 
  0, 0, 1, 0, 2, 0, 0, 
  0, 0, 1, 0, 2, 0, 0, 
  0, 0, 1, 0, 2, 0, 0, 
  0, 0, 1, 0, 2, 0, 0, 
  0, 0, 1, 0, 2, 0, 0, 
  0
};

// Conv pad
CONSTANT int kPadHeight[NUM_CONVOLUTIONS] = {
  0, 
  0, 
  1, 
  0, 0, 1, 0, 2, 0, 0, 
  0, 0, 1, 0, 2, 0, 0, 
  0, 0, 1, 0, 2, 0, 0, 
  0, 0, 1, 0, 2, 0, 0, 
  0, 0, 1, 0, 2, 0, 0, 
  0, 0, 1, 0, 2, 0, 0, 
  0, 0, 1, 0, 2, 0, 0, 
  0, 0, 1, 0, 2, 0, 0, 
  0, 0, 1, 0, 2, 0, 0, 
  0
};

// input image of each convolution stage
CONSTANT int kInputWidth[NUM_CONVOLUTIONS] = {
  114,
   56,
   56, 
   28, 28, 28, 28, 28, 28, 28, 
   28, 28, 28, 28, 28, 28, 28, 
   14, 14, 14, 14, 14, 14, 14,
   14, 14, 14, 14, 14, 14, 14,
   14, 14, 14, 14, 14, 14, 14,
   14, 14, 14, 14, 14, 14, 14,
   14, 14, 14, 14, 14, 14, 14,
    7,  7,  7,  7,  7,  7,  7,
    7,  7,  7,  7,  7,  7,  7,
    1
};
CONSTANT int kInputWidthMax = 114;

CONSTANT int kInputHeight[NUM_CONVOLUTIONS] = {
  114,
   56,
   56, 
   28, 28, 28, 28, 28, 28, 28, 
   28, 28, 28, 28, 28, 28, 28, 
   14, 14, 14, 14, 14, 14, 14,
   14, 14, 14, 14, 14, 14, 14,
   14, 14, 14, 14, 14, 14, 14,
   14, 14, 14, 14, 14, 14, 14,
   14, 14, 14, 14, 14, 14, 14,
    7,  7,  7,  7,  7,  7,  7,
    7,  7,  7,  7,  7,  7,  7,
    1
};
CONSTANT int kInputHeightMax = 114;

// output image of each convolution stage
CONSTANT int kOutputWidth[NUM_CONVOLUTIONS] = {
  112,
   56,
   56,
   28, 28, 28, 28, 28, 28, 28, 
   28, 28, 28, 28, 28, 28, 28, 
   14, 14, 14, 14, 14, 14, 14,
   14, 14, 14, 14, 14, 14, 14,
   14, 14, 14, 14, 14, 14, 14,
   14, 14, 14, 14, 14, 14, 14,
   14, 14, 14, 14, 14, 14, 14,
    7,  7,  7,  7,  7,  7,  7,
    7,  7,  7,  7,  7,  7,  7,
    1
 };
CONSTANT int kOutputWidthMax = 112;

CONSTANT int kOutputHeight[NUM_CONVOLUTIONS] = {
  112,
   56,
   56, 
   28, 28, 28, 28, 28, 28, 28, 
   28, 28, 28, 28, 28, 28, 28, 
   14, 14, 14, 14, 14, 14, 14,
   14, 14, 14, 14, 14, 14, 14,
   14, 14, 14, 14, 14, 14, 14,
   14, 14, 14, 14, 14, 14, 14,
   14, 14, 14, 14, 14, 14, 14,
    7,  7,  7,  7,  7,  7,  7,
    7,  7,  7,  7,  7,  7,  7,
    1
};
CONSTANT int kOutputHeightMax = 112;

CONSTANT int kInputChannels[NUM_CONVOLUTIONS] = {
  27,
  64,
  64,   
  192,  192,  96,  192,  16,  1,  192,   
  256,  256, 128,  256,  32,  1,  256,   
  480,  480,  96,  480,  16,  1,  480,   
  512,  512, 112,  512,  24,  1,  512,   
  512,  512, 128,  512,  24,  1,  512,   
  512,  512, 144,  512,  32,  1,  512,   
  528,  528, 160,  528,  32,  1,  528,   
  832,  832, 160,  832,  32,  1,  832,   
  832,  832, 192,  832,  48,  1,  832,
  1024
};

// how much filter data (number of float_vec_t reads) we need to prefetch at each stage of convolution
// formula : kCvecEnd * R * kFWvecEnd
CONSTANT int kFilterLoadSize[NUM_CONVOLUTIONS] = {
  CEIL(27,   C_VECTOR) * 3 * CEIL(3, FW_VECTOR), 
  CEIL(64,   C_VECTOR * FW_VECTOR) ,   
  CEIL(64,   C_VECTOR) * 3 * CEIL(3, FW_VECTOR),  
  
  CEIL(192,  C_VECTOR * FW_VECTOR),                 // inception3a_1x1
  CEIL(192,  C_VECTOR * FW_VECTOR), 
  CEIL(96,   C_VECTOR) * 3 * CEIL(3, FW_VECTOR),  
  CEIL(192,  C_VECTOR * FW_VECTOR), 
  CEIL(16,   C_VECTOR) * 5 * CEIL(5, FW_VECTOR),  
  0,
  CEIL(192,   C_VECTOR * FW_VECTOR), 
  
  CEIL(256,  C_VECTOR * FW_VECTOR),                 // inception3b_1x1
  CEIL(256,  C_VECTOR * FW_VECTOR), 
  CEIL(128,   C_VECTOR) * 3 * CEIL(3, FW_VECTOR),  
  CEIL(256,  C_VECTOR * FW_VECTOR), 
  CEIL(32,   C_VECTOR) * 5 * CEIL(5, FW_VECTOR),  
  0,
  CEIL(256,   C_VECTOR * FW_VECTOR), 
  
  CEIL(480,  C_VECTOR * FW_VECTOR),                 // inception4a_1x1
  CEIL(480,  C_VECTOR * FW_VECTOR), 
  CEIL(96,   C_VECTOR) * 3 * CEIL(3, FW_VECTOR),  
  CEIL(480,  C_VECTOR * FW_VECTOR), 
  CEIL(16,   C_VECTOR) * 5 * CEIL(5, FW_VECTOR),  
  0,
  CEIL(480,   C_VECTOR * FW_VECTOR), 
  
  CEIL(512,  C_VECTOR * FW_VECTOR),                 // inception4b_1x1
  CEIL(512,  C_VECTOR * FW_VECTOR), 
  CEIL(112,   C_VECTOR) * 3 * CEIL(3, FW_VECTOR),  
  CEIL(512,  C_VECTOR * FW_VECTOR), 
  CEIL(24,   C_VECTOR) * 5 * CEIL(5, FW_VECTOR),  
  0,
  CEIL(512,   C_VECTOR * FW_VECTOR), 
  
  CEIL(512,  C_VECTOR * FW_VECTOR),                 // inception4c_1x1
  CEIL(512,  C_VECTOR * FW_VECTOR), 
  CEIL(128,   C_VECTOR) * 3 * CEIL(3, FW_VECTOR),  
  CEIL(512,  C_VECTOR * FW_VECTOR), 
  CEIL(24,   C_VECTOR) * 5 * CEIL(5, FW_VECTOR),  
  0,
  CEIL(512,   C_VECTOR * FW_VECTOR), 
  
  CEIL(512,  C_VECTOR * FW_VECTOR),                 // inception4d_1x1
  CEIL(512,  C_VECTOR * FW_VECTOR), 
  CEIL(144,   C_VECTOR) * 3 * CEIL(3, FW_VECTOR),  
  CEIL(512,  C_VECTOR * FW_VECTOR), 
  CEIL(32,   C_VECTOR) * 5 * CEIL(5, FW_VECTOR),  
  0,
  CEIL(512,   C_VECTOR * FW_VECTOR), 
  
  CEIL(528,  C_VECTOR * FW_VECTOR),                 // inception4e_1x1
  CEIL(528,  C_VECTOR * FW_VECTOR), 
  CEIL(160,   C_VECTOR) * 3 * CEIL(3, FW_VECTOR),  
  CEIL(528,  C_VECTOR * FW_VECTOR), 
  CEIL(32,   C_VECTOR) * 5 * CEIL(5, FW_VECTOR),  
  0,
  CEIL(528,   C_VECTOR * FW_VECTOR), 
  
  CEIL(832,  C_VECTOR * FW_VECTOR),                 // inception5a_1x1
  CEIL(832,  C_VECTOR * FW_VECTOR), 
  CEIL(160,   C_VECTOR) * 3 * CEIL(3, FW_VECTOR),  
  CEIL(832,  C_VECTOR * FW_VECTOR), 
  CEIL(32,   C_VECTOR) * 5 * CEIL(5, FW_VECTOR),  
  0,
  CEIL(832,   C_VECTOR * FW_VECTOR), 
  
  CEIL(832,  C_VECTOR * FW_VECTOR),                 // inception3b_1x1
  CEIL(832,  C_VECTOR * FW_VECTOR), 
  CEIL(192,   C_VECTOR) * 3 * CEIL(3, FW_VECTOR),  
  CEIL(832,  C_VECTOR * FW_VECTOR), 
  CEIL(48,   C_VECTOR) * 5 * CEIL(5, FW_VECTOR),  
  0,
  CEIL(832,   C_VECTOR * FW_VECTOR), 
  
  CEIL(1024,   C_VECTOR * FW_VECTOR), 
  };

CONSTANT int kOutputChannels[NUM_CONVOLUTIONS] = {
   64,  
   64,
  192,   
   64,   96,  128,  16,  32,  192,  32,  
  128,  128,  192,  32,  96,  256,  64,  
  192,   96,  208,  16,  48,  480,  64,  
  160,  112,  224,  24,  64,  512,  64,  
  128,  128,  256,  24,  64,  512,  64,  
  112,  144,  288,  32,  64,  512,  64,  
  256,  160,  320,  32, 128,  528,  128,  
  256,  160,  320,  32, 128,  832,  128,  
  384,  192,  384,  48, 128,  832,  128,
  1000
};
CONSTANT int kOutputChannelsMax = 1000;

CONSTANT int kWvecEnd[NUM_CONVOLUTIONS] = {
  CEIL(114, W_VECTOR),
  CEIL(56, W_VECTOR),
  CEIL(56, W_VECTOR), 
  CEIL(28, W_VECTOR), CEIL(28, W_VECTOR), CEIL(28, W_VECTOR), CEIL(28, W_VECTOR), CEIL(28, W_VECTOR), CEIL(28, W_VECTOR), CEIL(28, W_VECTOR), 
  CEIL(28, W_VECTOR), CEIL(28, W_VECTOR), CEIL(28, W_VECTOR), CEIL(28, W_VECTOR), CEIL(28, W_VECTOR), CEIL(28, W_VECTOR), CEIL(28, W_VECTOR), 
  CEIL(14, W_VECTOR), CEIL(14, W_VECTOR), CEIL(14, W_VECTOR), CEIL(14, W_VECTOR), CEIL(14, W_VECTOR), CEIL(14, W_VECTOR), CEIL(14, W_VECTOR),
  CEIL(14, W_VECTOR), CEIL(14, W_VECTOR), CEIL(14, W_VECTOR), CEIL(14, W_VECTOR), CEIL(14, W_VECTOR), CEIL(14, W_VECTOR), CEIL(14, W_VECTOR),
  CEIL(14, W_VECTOR), CEIL(14, W_VECTOR), CEIL(14, W_VECTOR), CEIL(14, W_VECTOR), CEIL(14, W_VECTOR), CEIL(14, W_VECTOR), CEIL(14, W_VECTOR),
  CEIL(14, W_VECTOR), CEIL(14, W_VECTOR), CEIL(14, W_VECTOR), CEIL(14, W_VECTOR), CEIL(14, W_VECTOR), CEIL(14, W_VECTOR), CEIL(14, W_VECTOR),
  CEIL(14, W_VECTOR), CEIL(14, W_VECTOR), CEIL(14, W_VECTOR), CEIL(14, W_VECTOR), CEIL(14, W_VECTOR), CEIL(14, W_VECTOR), CEIL(14, W_VECTOR),
  CEIL(7,  W_VECTOR), CEIL(7,  W_VECTOR), CEIL(7,  W_VECTOR), CEIL(7,  W_VECTOR), CEIL(7,  W_VECTOR), CEIL(7,  W_VECTOR), CEIL(7,  W_VECTOR),
  CEIL(7,  W_VECTOR), CEIL(7,  W_VECTOR), CEIL(7,  W_VECTOR), CEIL(7,  W_VECTOR), CEIL(7,  W_VECTOR), CEIL(7,  W_VECTOR), CEIL(7,  W_VECTOR),
  CEIL(1, W_VECTOR)  // fc1000
};
CONSTANT int kWvecEndMax = CEIL(114, W_VECTOR);

CONSTANT int kConvStride[NUM_CONVOLUTIONS] = {
  1,      
  1,       
  1, 
  1, 1, 1, 1, 1, 1, 1, 
  1, 1, 1, 1, 1, 1, 1, 
  1, 1, 1, 1, 1, 1, 1, 
  1, 1, 1, 1, 1, 1, 1, 
  1, 1, 1, 1, 1, 1, 1, 
  1, 1, 1, 1, 1, 1, 1, 
  1, 1, 1, 1, 1, 1, 1, 
  1, 1, 1, 1, 1, 1, 1, 
  1, 1, 1, 1, 1, 1, 1, 
  1        // FC1000
};

//
// POOL
//

CONSTANT bool kPoolEnable[NUM_CONVOLUTIONS] = {
   true,             
  false,            
   true, 
  false, false, false, false, false, true, false, 
  //false, false, false, false, false, true, false, 
   true, false,  true, false,  true, true,  true, 
  false, false, false, false, false, true, false, 
  false, false, false, false, false, true, false, 
  false, false, false, false, false, true, false, 
  false, false, false, false, false, true, false, 
  //false, false, false, false, false, true, false, 
  true, false,  true, false,  true, true,  true, 
  false, false, false, false, false, true, false, 
  false, false, false, false, false, true, false, 
  false             // fc1000
};

CONSTANT int kPoolWindow[NUM_CONVOLUTIONS] = {
  3, 
  3, 
  3, 
  3, 3, 3, 3, 3, 3, 3, 
  3, 3, 3, 3, 3, 3, 3, 
  3, 3, 3, 3, 3, 3, 3, 
  3, 3, 3, 3, 3, 3, 3, 
  3, 3, 3, 3, 3, 3, 3, 
  3, 3, 3, 3, 3, 3, 3, 
  3, 3, 3, 3, 3, 3, 3, 
  3, 3, 3, 3, 3, 3, 3, 
  3, 3, 3, 3, 3, 3, 3, 
  3  // fc1000
};

// 0 - max pooling
// 1 - average pooling
CONSTANT int kPoolType[NUM_CONVOLUTIONS] = {
  0, 
  0, 
  0, 
  0, 0, 0, 0, 0, 0, 0, 
  0, 0, 0, 0, 0, 0, 0, 
  0, 0, 0, 0, 0, 0, 0, 
  0, 0, 0, 0, 0, 0, 0, 
  0, 0, 0, 0, 0, 0, 0, 
  0, 0, 0, 0, 0, 0, 0, 
  0, 0, 0, 0, 0, 0, 0, 
  0, 0, 0, 0, 0, 0, 0, 
  0, 0, 0, 0, 0, 0, 0, 
  0  // fc1000
};

CONSTANT bool kPoolStride2[NUM_CONVOLUTIONS] = {
  true,       
  false,       
  true, 
  false, false, false, false, false, false, false, 
  //false, false, false, false, false, false, false, 
   true, false,  true, false,  true, false,  true, 
  false, false, false, false, false, false, false, 
  false, false, false, false, false, false, false, 
  false, false, false, false, false, false, false, 
  false, false, false, false, false, false, false, 
  //false, false, false, false, false, false, false, 
  true, false,  true, false,  true, false,  true, 
  false, false, false, false, false, false, false, 
  false, false, false, false, false, false, false, 
};

CONSTANT int kPoolOutputWidth[NUM_CONVOLUTIONS] = {
  56,
  56,
  28, 
  28, 28, 28, 28, 28, 28, 28, 
  //28, 28, 28, 28, 28, 28, 28, 
  14, 28, 14, 28, 14, 28, 14, 
  14, 14, 14, 14, 14, 14, 14,
  14, 14, 14, 14, 14, 14, 14,
  14, 14, 14, 14, 14, 14, 14,
  14, 14, 14, 14, 14, 14, 14,
  // 14, 14, 14, 14, 14, 14, 14,
  7, 14,  7, 14,  7, 14,  7,
   7,  7,  7,  7,  7,  7,  7,
   7,  7,  7,  7,  7,  7,  7,
   1   // fc1000
};
CONSTANT int kPoolOutputWidthMax = 56;

CONSTANT int kPoolOutputHeight[NUM_CONVOLUTIONS] = {
  56,
  56,
  28, 
  28, 28, 28, 28, 28, 28, 28, 
  //28, 28, 28, 28, 28, 28, 28, 
  14, 28, 14, 28, 14, 28, 14, 
  14, 14, 14, 14, 14, 14, 14,
  14, 14, 14, 14, 14, 14, 14,
  14, 14, 14, 14, 14, 14, 14,
  14, 14, 14, 14, 14, 14, 14,
 //14, 14, 14, 14, 14, 14, 14,
   7, 14,  7, 14,  7, 14,  7,
   7,  7,  7,  7,  7,  7,  7,
   7,  7,  7,  7,  7,  7,  7,
   1   // fc1000
};
CONSTANT int kPoolOutputHeightMax = 56;

CONSTANT int kPoolOutputWvecEnd[NUM_CONVOLUTIONS] = {
  CEIL(56, W_VECTOR),
  CEIL(56, W_VECTOR), 
  CEIL(28, W_VECTOR),
  CEIL(28, W_VECTOR), CEIL(28, W_VECTOR), CEIL(28, W_VECTOR), CEIL(28, W_VECTOR), CEIL(28, W_VECTOR), CEIL(28, W_VECTOR), CEIL(28, W_VECTOR), 
  //CEIL(28, W_VECTOR), CEIL(28, W_VECTOR), CEIL(28, W_VECTOR), CEIL(28, W_VECTOR), CEIL(28, W_VECTOR), CEIL(28, W_VECTOR), CEIL(28, W_VECTOR), 
  CEIL(14, W_VECTOR), CEIL(28, W_VECTOR), CEIL(14, W_VECTOR), CEIL(28, W_VECTOR), CEIL(14, W_VECTOR), CEIL(28, W_VECTOR), CEIL(14, W_VECTOR), 
  CEIL(14, W_VECTOR), CEIL(14, W_VECTOR), CEIL(14, W_VECTOR), CEIL(14, W_VECTOR), CEIL(14, W_VECTOR), CEIL(14, W_VECTOR), CEIL(14, W_VECTOR),
  CEIL(14, W_VECTOR), CEIL(14, W_VECTOR), CEIL(14, W_VECTOR), CEIL(14, W_VECTOR), CEIL(14, W_VECTOR), CEIL(14, W_VECTOR), CEIL(14, W_VECTOR),
  CEIL(14, W_VECTOR), CEIL(14, W_VECTOR), CEIL(14, W_VECTOR), CEIL(14, W_VECTOR), CEIL(14, W_VECTOR), CEIL(14, W_VECTOR), CEIL(14, W_VECTOR),
  CEIL(14, W_VECTOR), CEIL(14, W_VECTOR), CEIL(14, W_VECTOR), CEIL(14, W_VECTOR), CEIL(14, W_VECTOR), CEIL(14, W_VECTOR), CEIL(14, W_VECTOR),
 // CEIL(14, W_VECTOR), CEIL(14, W_VECTOR), CEIL(14, W_VECTOR), CEIL(14, W_VECTOR), CEIL(14, W_VECTOR), CEIL(14, W_VECTOR), CEIL(14, W_VECTOR),
  CEIL(7,  W_VECTOR), CEIL(14, W_VECTOR), CEIL(7,  W_VECTOR), CEIL(14, W_VECTOR), CEIL(7, W_VECTOR),  CEIL(14, W_VECTOR), CEIL(7,  W_VECTOR),
  CEIL(7,  W_VECTOR), CEIL(7,  W_VECTOR), CEIL(7,  W_VECTOR), CEIL(7,  W_VECTOR), CEIL(7,  W_VECTOR), CEIL(7,  W_VECTOR), CEIL(7,  W_VECTOR),
  CEIL(7,  W_VECTOR), CEIL(7,  W_VECTOR), CEIL(7,  W_VECTOR), CEIL(7,  W_VECTOR), CEIL(7,  W_VECTOR), CEIL(7,  W_VECTOR), CEIL(7,  W_VECTOR),
  CEIL(1, W_VECTOR) // fc1000
};

//pool output feature map height
CONSTANT int kOhEndWithOffset[NUM_CONVOLUTIONS] = {
  112 + POOL_OFFSET_P,
  56 + POOL_OFFSET_P,
  56 + POOL_OFFSET_P, 
  28 + POOL_OFFSET_P, 28 + POOL_OFFSET_P, 28 + POOL_OFFSET_P, 28 + POOL_OFFSET_P, 28 + POOL_OFFSET_P, 28 + POOL_OFFSET_P, 28 + POOL_OFFSET_P, 
  28 + POOL_OFFSET_P, 28 + POOL_OFFSET_P, 28 + POOL_OFFSET_P, 28 + POOL_OFFSET_P, 28 + POOL_OFFSET_P, 28 + POOL_OFFSET_P, 28 + POOL_OFFSET_P, 
  14 + POOL_OFFSET_P, 14 + POOL_OFFSET_P, 14 + POOL_OFFSET_P, 14 + POOL_OFFSET_P, 14 + POOL_OFFSET_P, 14 + POOL_OFFSET_P, 14 + POOL_OFFSET_P, 
  14 + POOL_OFFSET_P, 14 + POOL_OFFSET_P, 14 + POOL_OFFSET_P, 14 + POOL_OFFSET_P, 14 + POOL_OFFSET_P, 14 + POOL_OFFSET_P, 14 + POOL_OFFSET_P, 
  14 + POOL_OFFSET_P, 14 + POOL_OFFSET_P, 14 + POOL_OFFSET_P, 14 + POOL_OFFSET_P, 14 + POOL_OFFSET_P, 14 + POOL_OFFSET_P, 14 + POOL_OFFSET_P, 
  14 + POOL_OFFSET_P, 14 + POOL_OFFSET_P, 14 + POOL_OFFSET_P, 14 + POOL_OFFSET_P, 14 + POOL_OFFSET_P, 14 + POOL_OFFSET_P, 14 + POOL_OFFSET_P, 
  14 + POOL_OFFSET_P, 14 + POOL_OFFSET_P, 14 + POOL_OFFSET_P, 14 + POOL_OFFSET_P, 14 + POOL_OFFSET_P, 14 + POOL_OFFSET_P, 14 + POOL_OFFSET_P, 
   7 + POOL_OFFSET_P,  7 + POOL_OFFSET_P,  7 + POOL_OFFSET_P,  7 + POOL_OFFSET_P,  7 + POOL_OFFSET_P,  7 + POOL_OFFSET_P,  7 + POOL_OFFSET_P,  
   7 + POOL_OFFSET_P,  7 + POOL_OFFSET_P,  7 + POOL_OFFSET_P,  7 + POOL_OFFSET_P,  7 + POOL_OFFSET_P,  7 + POOL_OFFSET_P,  7 + POOL_OFFSET_P,  
   1 + POOL_OFFSET_P
};
CONSTANT int kOhEndWithOffsetMax = 112 + POOL_OFFSET_P;

//pool output image width
CONSTANT int kOwEndWithOffset[NUM_CONVOLUTIONS] = {
  112 + POOL_OFFSET_Q,
  56 + POOL_OFFSET_Q,
  56 + POOL_OFFSET_Q, 
  28 + POOL_OFFSET_Q, 28 + POOL_OFFSET_Q, 28 + POOL_OFFSET_Q, 28 + POOL_OFFSET_Q, 28 + POOL_OFFSET_Q, 28 + POOL_OFFSET_Q, 28 + POOL_OFFSET_Q, 
  28 + POOL_OFFSET_Q, 28 + POOL_OFFSET_Q, 28 + POOL_OFFSET_Q, 28 + POOL_OFFSET_Q, 28 + POOL_OFFSET_Q, 28 + POOL_OFFSET_Q, 28 + POOL_OFFSET_Q, 
  14 + POOL_OFFSET_Q, 14 + POOL_OFFSET_Q, 14 + POOL_OFFSET_Q, 14 + POOL_OFFSET_Q, 14 + POOL_OFFSET_Q, 14 + POOL_OFFSET_Q, 14 + POOL_OFFSET_Q,
  14 + POOL_OFFSET_Q, 14 + POOL_OFFSET_Q, 14 + POOL_OFFSET_Q, 14 + POOL_OFFSET_Q, 14 + POOL_OFFSET_Q, 14 + POOL_OFFSET_Q, 14 + POOL_OFFSET_Q,
  14 + POOL_OFFSET_Q, 14 + POOL_OFFSET_Q, 14 + POOL_OFFSET_Q, 14 + POOL_OFFSET_Q, 14 + POOL_OFFSET_Q, 14 + POOL_OFFSET_Q, 14 + POOL_OFFSET_Q,
  14 + POOL_OFFSET_Q, 14 + POOL_OFFSET_Q, 14 + POOL_OFFSET_Q, 14 + POOL_OFFSET_Q, 14 + POOL_OFFSET_Q, 14 + POOL_OFFSET_Q, 14 + POOL_OFFSET_Q,
  14 + POOL_OFFSET_Q, 14 + POOL_OFFSET_Q, 14 + POOL_OFFSET_Q, 14 + POOL_OFFSET_Q, 14 + POOL_OFFSET_Q, 14 + POOL_OFFSET_Q, 14 + POOL_OFFSET_Q,
   7 + POOL_OFFSET_Q,  7 + POOL_OFFSET_Q,  7 + POOL_OFFSET_Q,  7 + POOL_OFFSET_Q,  7 + POOL_OFFSET_Q,  7 + POOL_OFFSET_Q,  7 + POOL_OFFSET_Q,
   7 + POOL_OFFSET_Q,  7 + POOL_OFFSET_Q,  7 + POOL_OFFSET_Q,  7 + POOL_OFFSET_Q,  7 + POOL_OFFSET_Q,  7 + POOL_OFFSET_Q,  7 + POOL_OFFSET_Q,
   1 + POOL_OFFSET_Q
};
CONSTANT int kOwEndWithOffsetMax = 112 + POOL_OFFSET_Q;

CONSTANT int kFWvecEnd[NUM_CONVOLUTIONS] = {
  CEIL(3, FW_VECTOR),
  CEIL(1, FW_VECTOR), 
  CEIL(3, FW_VECTOR), 
  CEIL(1, FW_VECTOR), CEIL(1, FW_VECTOR), CEIL(3, FW_VECTOR), CEIL(1, FW_VECTOR), CEIL(5, FW_VECTOR), CEIL(1, FW_VECTOR), CEIL(1, FW_VECTOR), 
  CEIL(1, FW_VECTOR), CEIL(1, FW_VECTOR), CEIL(3, FW_VECTOR), CEIL(1, FW_VECTOR), CEIL(5, FW_VECTOR), CEIL(1, FW_VECTOR), CEIL(1, FW_VECTOR), 
  CEIL(1, FW_VECTOR), CEIL(1, FW_VECTOR), CEIL(3, FW_VECTOR), CEIL(1, FW_VECTOR), CEIL(5, FW_VECTOR), CEIL(1, FW_VECTOR), CEIL(1, FW_VECTOR), 
  CEIL(1, FW_VECTOR), CEIL(1, FW_VECTOR), CEIL(3, FW_VECTOR), CEIL(1, FW_VECTOR), CEIL(5, FW_VECTOR), CEIL(1, FW_VECTOR), CEIL(1, FW_VECTOR), 
  CEIL(1, FW_VECTOR), CEIL(1, FW_VECTOR), CEIL(3, FW_VECTOR), CEIL(1, FW_VECTOR), CEIL(5, FW_VECTOR), CEIL(1, FW_VECTOR), CEIL(1, FW_VECTOR), 
  CEIL(1, FW_VECTOR), CEIL(1, FW_VECTOR), CEIL(3, FW_VECTOR), CEIL(1, FW_VECTOR), CEIL(5, FW_VECTOR), CEIL(1, FW_VECTOR), CEIL(1, FW_VECTOR), 
  CEIL(1, FW_VECTOR), CEIL(1, FW_VECTOR), CEIL(3, FW_VECTOR), CEIL(1, FW_VECTOR), CEIL(5, FW_VECTOR), CEIL(1, FW_VECTOR), CEIL(1, FW_VECTOR), 
  CEIL(1, FW_VECTOR), CEIL(1, FW_VECTOR), CEIL(3, FW_VECTOR), CEIL(1, FW_VECTOR), CEIL(5, FW_VECTOR), CEIL(1, FW_VECTOR), CEIL(1, FW_VECTOR), 
  CEIL(1, FW_VECTOR), CEIL(1, FW_VECTOR), CEIL(3, FW_VECTOR), CEIL(1, FW_VECTOR), CEIL(5, FW_VECTOR), CEIL(1, FW_VECTOR), CEIL(1, FW_VECTOR), 
  CEIL(1, FW_VECTOR)  // fc1000
};
CONSTANT int kFWvecEndMax = CEIL(5, FW_VECTOR);

CONSTANT int kCvecEnd[NUM_CONVOLUTIONS] = {
  CEIL(27,   C_VECTOR),
  CEIL(64,   C_VECTOR),
  CEIL(64,   C_VECTOR),  
  CEIL(192,  C_VECTOR),  CEIL(192, C_VECTOR), CEIL(96,  C_VECTOR),  CEIL(192, C_VECTOR),  CEIL(16,  C_VECTOR), CEIL(1, C_VECTOR),  CEIL(192, C_VECTOR),  
  CEIL(256,  C_VECTOR),  CEIL(256, C_VECTOR), CEIL(128, C_VECTOR),  CEIL(256, C_VECTOR),  CEIL(32,  C_VECTOR), CEIL(1, C_VECTOR),  CEIL(256, C_VECTOR),  
  CEIL(480,  C_VECTOR),  CEIL(480, C_VECTOR), CEIL(96,  C_VECTOR),  CEIL(480, C_VECTOR),  CEIL(16,  C_VECTOR), CEIL(1, C_VECTOR),  CEIL(480, C_VECTOR),  
  CEIL(512,  C_VECTOR),  CEIL(512, C_VECTOR), CEIL(112, C_VECTOR),  CEIL(512, C_VECTOR),  CEIL(24,  C_VECTOR), CEIL(1, C_VECTOR),  CEIL(512, C_VECTOR),  
  CEIL(512,  C_VECTOR),  CEIL(512, C_VECTOR), CEIL(128, C_VECTOR),  CEIL(512, C_VECTOR),  CEIL(24,  C_VECTOR), CEIL(1, C_VECTOR),  CEIL(512, C_VECTOR),  
  CEIL(512,  C_VECTOR),  CEIL(512, C_VECTOR), CEIL(144, C_VECTOR),  CEIL(512, C_VECTOR),  CEIL(32,  C_VECTOR), CEIL(1, C_VECTOR),  CEIL(512, C_VECTOR),  
  CEIL(528,  C_VECTOR),  CEIL(528, C_VECTOR), CEIL(160, C_VECTOR),  CEIL(528, C_VECTOR),  CEIL(32,  C_VECTOR), CEIL(1, C_VECTOR),  CEIL(528, C_VECTOR),  
  CEIL(832,  C_VECTOR),  CEIL(832, C_VECTOR), CEIL(160, C_VECTOR),  CEIL(832, C_VECTOR),  CEIL(32,  C_VECTOR), CEIL(1, C_VECTOR),  CEIL(832, C_VECTOR),  
  CEIL(832,  C_VECTOR),  CEIL(832, C_VECTOR), CEIL(192, C_VECTOR),  CEIL(832, C_VECTOR),  CEIL(48,  C_VECTOR), CEIL(1, C_VECTOR),  CEIL(832, C_VECTOR),  
  CEIL(1024, C_VECTOR)  // fc1000
};
CONSTANT int kCvecEndMax = CEIL(1024, C_VECTOR);

CONSTANT int kFilterCvecEnd[NUM_CONVOLUTIONS] = {
 CEIL(27,  C_VECTOR),
 CEIL(64,  C_VECTOR*FW_VECTOR),
 CEIL(64,  C_VECTOR),  
 CEIL(192, C_VECTOR*FW_VECTOR), CEIL(192, C_VECTOR*FW_VECTOR), CEIL(96,  C_VECTOR), CEIL(192, C_VECTOR*FW_VECTOR),  CEIL(16, C_VECTOR), CEIL(1, C_VECTOR*FW_VECTOR),  CEIL(192, C_VECTOR*FW_VECTOR),  
 CEIL(256, C_VECTOR*FW_VECTOR), CEIL(256, C_VECTOR*FW_VECTOR), CEIL(128, C_VECTOR), CEIL(256, C_VECTOR*FW_VECTOR),  CEIL(32, C_VECTOR), CEIL(1, C_VECTOR*FW_VECTOR),  CEIL(256, C_VECTOR*FW_VECTOR),  
 CEIL(480, C_VECTOR*FW_VECTOR), CEIL(480, C_VECTOR*FW_VECTOR), CEIL(96,  C_VECTOR), CEIL(480, C_VECTOR*FW_VECTOR),  CEIL(16, C_VECTOR), CEIL(1, C_VECTOR*FW_VECTOR),  CEIL(480, C_VECTOR*FW_VECTOR),  
 CEIL(512, C_VECTOR*FW_VECTOR), CEIL(512, C_VECTOR*FW_VECTOR), CEIL(112, C_VECTOR), CEIL(512, C_VECTOR*FW_VECTOR),  CEIL(24, C_VECTOR), CEIL(1, C_VECTOR*FW_VECTOR),  CEIL(512, C_VECTOR*FW_VECTOR),  
 CEIL(512, C_VECTOR*FW_VECTOR), CEIL(512, C_VECTOR*FW_VECTOR), CEIL(128, C_VECTOR), CEIL(512, C_VECTOR*FW_VECTOR),  CEIL(24, C_VECTOR), CEIL(1, C_VECTOR*FW_VECTOR),  CEIL(512, C_VECTOR*FW_VECTOR),  
 CEIL(512, C_VECTOR*FW_VECTOR), CEIL(512, C_VECTOR*FW_VECTOR), CEIL(144, C_VECTOR), CEIL(512, C_VECTOR*FW_VECTOR),  CEIL(32, C_VECTOR), CEIL(1, C_VECTOR*FW_VECTOR),  CEIL(512, C_VECTOR*FW_VECTOR),  
 CEIL(528, C_VECTOR*FW_VECTOR), CEIL(528, C_VECTOR*FW_VECTOR), CEIL(160, C_VECTOR), CEIL(528, C_VECTOR*FW_VECTOR),  CEIL(32, C_VECTOR), CEIL(1, C_VECTOR*FW_VECTOR),  CEIL(528, C_VECTOR*FW_VECTOR),  
 CEIL(832, C_VECTOR*FW_VECTOR), CEIL(832, C_VECTOR*FW_VECTOR), CEIL(160, C_VECTOR), CEIL(832, C_VECTOR*FW_VECTOR),  CEIL(32, C_VECTOR), CEIL(1, C_VECTOR*FW_VECTOR),  CEIL(832, C_VECTOR*FW_VECTOR),  
 CEIL(832, C_VECTOR*FW_VECTOR), CEIL(832, C_VECTOR*FW_VECTOR), CEIL(192, C_VECTOR), CEIL(832, C_VECTOR*FW_VECTOR),  CEIL(48, C_VECTOR), CEIL(1, C_VECTOR*FW_VECTOR),  CEIL(832, C_VECTOR*FW_VECTOR),  
 CEIL(1024, C_VECTOR * FW_VECTOR)  // fc1000
};
CONSTANT int kFilterCvecEndMax = CEIL(1024, C_VECTOR*FW_VECTOR);

// input
CONSTANT int END_WW_MAX_INPUT_READER = CEIL(114, FW_VECTOR);

CONSTANT int kNvecEnd[NUM_CONVOLUTIONS] = {
  CEIL(64,   N_VECTOR),
  CEIL(64,  N_VECTOR),
  CEIL(192,   N_VECTOR), 
  CEIL( 64,  N_VECTOR),  CEIL( 96,  N_VECTOR), CEIL(128,  N_VECTOR), CEIL(16,  N_VECTOR), CEIL( 32,  N_VECTOR), CEIL(192,   N_VECTOR), CEIL(32,   N_VECTOR), 
  CEIL(128,  N_VECTOR),  CEIL(128,  N_VECTOR), CEIL(192,  N_VECTOR), CEIL(32,  N_VECTOR), CEIL( 96,  N_VECTOR), CEIL(256,   N_VECTOR), CEIL(64,   N_VECTOR), 
  CEIL(192,  N_VECTOR),  CEIL( 96,  N_VECTOR), CEIL(208,  N_VECTOR), CEIL(16,  N_VECTOR), CEIL( 48,  N_VECTOR), CEIL(480,   N_VECTOR), CEIL(64,   N_VECTOR), 
  CEIL(160,  N_VECTOR),  CEIL(112,  N_VECTOR), CEIL(224,  N_VECTOR), CEIL(24,  N_VECTOR), CEIL( 64,  N_VECTOR), CEIL(512,   N_VECTOR), CEIL(64,   N_VECTOR), 
  CEIL(128,  N_VECTOR),  CEIL(128,  N_VECTOR), CEIL(256,  N_VECTOR), CEIL(24,  N_VECTOR), CEIL( 64,  N_VECTOR), CEIL(512,   N_VECTOR), CEIL(64,   N_VECTOR), 
  CEIL(112,  N_VECTOR),  CEIL(144,  N_VECTOR), CEIL(288,  N_VECTOR), CEIL(32,  N_VECTOR), CEIL( 64,  N_VECTOR), CEIL(512,   N_VECTOR), CEIL(64,   N_VECTOR), 
  CEIL(256,  N_VECTOR),  CEIL(160,  N_VECTOR), CEIL(320,  N_VECTOR), CEIL(32,  N_VECTOR), CEIL(128,  N_VECTOR), CEIL(528,   N_VECTOR), CEIL(128,  N_VECTOR), 
  CEIL(256,  N_VECTOR),  CEIL(160,  N_VECTOR), CEIL(320,  N_VECTOR), CEIL(32,  N_VECTOR), CEIL(128,  N_VECTOR), CEIL(832,   N_VECTOR), CEIL(128,  N_VECTOR), 
  CEIL(384,  N_VECTOR),  CEIL(192,  N_VECTOR), CEIL(384,  N_VECTOR), CEIL(48,  N_VECTOR), CEIL(128,  N_VECTOR), CEIL(832,   N_VECTOR), CEIL(128,  N_VECTOR), 
  CEIL(1000, N_VECTOR)  // fc1000
};
CONSTANT int kNvecEndMax = CEIL(1000, N_VECTOR);

CONSTANT int kNEndWithOffset[NUM_CONVOLUTIONS] = {
   64,  
   64,
  192,   
   64,   96,  128,  16,  32,  192,  32,  
  128,  128,  192,  32,  96,  256,  64,  
  192,   96,  208,  16,  48,  480,  64,  
  160,  112,  224,  24,  64,  512,  64,  
  128,  128,  256,  24,  64,  512,  64,  
  112,  144,  288,  32,  64,  512,  64,  
  256,  160,  320,  32, 128,  528,  128,  
  256,  160,  320,  32, 128,  832,  128,  
  384,  192,  384,  48, 128,  832,  128,
  1000
};
CONSTANT int kNEndWithOffsetMax = 1000;

// used in pool
CONSTANT int kNStart[NUM_CONVOLUTIONS] = {
    0,
    0, 
    0, 
    0,   0,    64,   0,  192,    0,   224, 
    0,   0,   128,   0,  320,    0,   416, 
    0,   0,   192,   0,  400,    0,   448, 
    0,   0,   160,   0,  384,    0,   448, 
    0,   0,   128,   0,  384,    0,   448, 
    0,   0,   112,   0,  400,    0,   464, 
    0,   0,   256,   0,  576,    0,   704, 
    0,   0,   256,   0,  576,    0,   704, 
    0,   0,   384,   0,  768,    0,   896, 
    0  // fc1000
};

// used in pool
CONSTANT int kNEnd[NUM_CONVOLUTIONS] = {
   64, 
   64,
  192,   
   64,   96,  192,  16,  224,  192,   256,  
  128,  128,  320,  32,  416,  256,   480,  
  192,   96,  400,  16,  448,  480,   512,  
  160,  112,  384,  24,  448,  512,   512,  
  128,  128,  384,  24,  448,  512,   512,  
  112,  144,  400,  32,  464,  512,   528,  
  256,  160,  576,  32,  704,  528,   832,  
  256,  160,  576,  32,  704,  832,   832,  
  384,  192,  768,  48,  896,  832,  1024,
  1000   // fc1000
};

CONSTANT bool kBiasEnable[NUM_CONVOLUTIONS] = {
  1,
  1,
  1,
  1, 1, 1, 1, 1, 0, 1,
  1, 1, 1, 1, 1, 0, 1,
  1, 1, 1, 1, 1, 0, 1,
  1, 1, 1, 1, 1, 0, 1,
  1, 1, 1, 1, 1, 0, 1,
  1, 1, 1, 1, 1, 0, 1,
  1, 1, 1, 1, 1, 0, 1,
  1, 1, 1, 1, 1, 0, 1,
  1, 1, 1, 1, 1, 0, 1,
  1
};

CONSTANT bool kBnEnable[NUM_CONVOLUTIONS] = {
  1,
  0,
  1,
  0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0,
  0
};

CONSTANT int kIpoolEnable[NUM_CONVOLUTIONS] =  {
  0,
  0,
  0,
  0, 0, 0, 0, 0, 1, 0,
  0, 0, 0, 0, 0, 1, 0,
  0, 0, 0, 0, 0, 1, 0,
  0, 0, 0, 0, 0, 1, 0,
  0, 0, 0, 0, 0, 1, 0,
  0, 0, 0, 0, 0, 1, 0,
  0, 0, 0, 0, 0, 1, 0,
  0, 0, 0, 0, 0, 1, 0,
  0, 0, 0, 0, 0, 1, 0,
  0
};

CONSTANT int kPoolPad[NUM_CONVOLUTIONS] =  {
  0,
  0,
  0,
  0, 0, 0, 0, 0, 1, 0,
  0, 0, 0, 0, 0, 1, 0,
  0, 0, 0, 0, 0, 1, 0,
  0, 0, 0, 0, 0, 1, 0,
  0, 0, 0, 0, 0, 1, 0,
  0, 0, 0, 0, 0, 1, 0,
  0, 0, 0, 0, 0, 1, 0,
  0, 0, 0, 0, 0, 1, 0,
  0, 0, 0, 0, 0, 1, 0,
  0
};

//
// only for host code
//

CONSTANT int kInputLayer[NUM_CONVOLUTIONS] = {
   0, 
   1,
   2,
   3,   3,   5,   3,   7,   3,   9,
  68,  68,  12,  68,  14,  68,  16,
  69,  69,  19,  69,  21,  69,  23,
  70,  70,  26,  70,  28,  70,  30,
  71,  71,  33,  71,  35,  71,  37,
  72,  72,  40,  72,  42,  72,  44,
  73,  73,  47,  73,  49,  73,  51,
  74,  74,  54,  74,  56,  74,  58,
  75,  75,  61,  75,  63,  75,  65,
  76
};

CONSTANT bool kBranchTail[NUM_CONVOLUTIONS] ={
  0,
  0, 
  0, 
  1, 0, 1, 0, 1, 0, 1, 
  1, 0, 1, 0, 1, 0, 1, 
  1, 0, 1, 0, 1, 0, 1, 
  1, 0, 1, 0, 1, 0, 1, 
  1, 0, 1, 0, 1, 0, 1, 
  1, 0, 1, 0, 1, 0, 1, 
  1, 0, 1, 0, 1, 0, 1, 
  1, 0, 1, 0, 1, 0, 1, 
  1, 0, 1, 0, 1, 0, 1,
  0
};

CONSTANT int kConcatLayer[NUM_CONVOLUTIONS] ={
  0,
  0, 
  0, 
  0, 0, 0, 0, 0, 0, 0, 
  1, 0, 1, 0, 1, 0, 1, 
  2, 0, 2, 0, 2, 0, 2, 
  3, 0, 3, 0, 3, 0, 3, 
  4, 0, 4, 0, 4, 0, 4, 
  5, 0, 5, 0, 5, 0, 5, 
  6, 0, 6, 0, 6, 0, 6, 
  7, 0, 7, 0, 7, 0, 7, 
  8, 0, 8, 0, 8, 0, 8, 
  0 
};

CONSTANT int kSequencerIdleCycle[NUM_CONVOLUTIONS] = {
  0,
  0,
  0,
  0,    0,    0,    0,    0,    0,    0,
  0,    0,    0,    0,    0,    0,    0,
  0,    0,    0,    0,    0,    0,    0,
  0,    0,    0,    0,    0,    0,    0,
  0,    0,    0,    0,    0,    0,    0,
  0,    0,    0,    0,    0,    0,    0,
  0,    0,    0,    0,    0,    0,    0,
  0,    0,    0,    0,    0,    0,    0,
  0,    0,    0,    0,    0,    0,    0,
  0
};

//
// static cycles
//

#ifdef STATIC_CYCLE
CONSTANT int feature_writer_cycles[NUM_CONVOLUTIONS] = {
  1792,
  1792,
  1344,
  448 ,
  672 ,
  896 ,
  112 ,
  224 ,
  1344,
  224 ,
	224 ,
	896 ,
	336 ,
	224 ,
	168 ,
	1792,
	112 ,
	336 ,
	168 ,
	364 ,
	28  ,
	84  ,
	840 ,
	112 ,
	280 ,
	196 ,
	392 ,
	56  ,
	112 ,
	896 ,
	112 ,
	224 ,
	224 ,
	448 ,
	56  ,
	112 ,
	896 ,
	112 ,
	196 ,
	252 ,
	504 ,
	56  ,
	112 ,
	896 ,
	112 ,
	112 ,
	280 ,
	140 ,
	56  ,
	56  ,
	924 ,
	56  ,
	112 ,
	70  ,
	140 ,
	14  ,
	56  ,
	364 ,
	56  ,
	168 ,
	84  ,
	168 ,
	21  ,
	56  ,
	364 ,
	56  ,
	63  
};

CONSTANT int filter_reader_conv_cycles[NUM_CONVOLUTIONS] = {
  384  ,
  128  ,
  2304 ,
  256  ,
  384  ,
  2304 ,
  64   ,
  320  ,
  0    ,
  128  ,
	768  ,
	768  ,
	4608 ,
	192  ,
	1920 ,
	0    ,
	384  ,
	1920 ,
	960  ,
	3744 ,
	160  ,
	480  ,
	0    ,
	640  ,
	1760 ,
	1232 ,
	4704 ,
	352  ,
	1280 ,
	0    ,
	704  ,
	1408 ,
	1408 ,
	6144 ,
	352  ,
	1280 ,
	0    ,
	704  ,
	1232 ,
	1584 ,
	7776 ,
	352  ,
	1280 ,
	0    ,
	704  ,
	2816 ,
	1760 ,
	9600 ,
	352  ,
	2560 ,
	0    ,
	1408 ,
	4608 ,
	2880 ,
	9600 ,
	576  ,
	2560 ,
	0    ,
	2304 ,
	6912 ,
	3456 ,
	13824,
	864  ,
	3840 ,
	0    ,
	2304 ,
	22176 
};

CONSTANT int conv_cycles[NUM_CONVOLUTIONS] = {
  61824,
  7168 ,
  96768,
  5376 ,
  8064 ,
  24192,
  1344 ,
  3360 ,
  1344 ,
  2688 ,
	14336,
	14336,
	48384,
	3584 ,
	20160,
	1792 ,
	7168 ,
	10080,
	5040 ,
	9828 ,
	840  ,
	1260 ,
	972  ,
	3360 ,
	8960 ,
	6272 ,
	12348,
	1792 ,
	3360 ,
	1044 ,
	3584 ,
	7168 ,
	7168 ,
	16128,
	1792 ,
	3360 ,
	1044 ,
	3584 ,
	6272 ,
	8064 ,
	20412,
	1792 ,
	3360 ,
	1044 ,
	3584 ,
	14784,
	9240 ,
	25200,
	1848 ,
	6720 ,
	1072 ,
	7392 ,
	5824 ,
	3756 ,
	9540 ,
	728  ,
	2520 ,
	645  ,
	2912 ,
	8736 ,
	4580 ,
	13752,
	1208 ,
	3780 ,
	645  ,
	2912 ,
	21888 
};

CONSTANT int pool_cycles[NUM_CONVOLUTIONS] = {
  10488,
  2088 ,
  8352 ,
  600  ,
  900  ,
  1440 ,
  150  ,
  360  ,
  1800 ,
  300  ,
	1200 ,
	1200 ,
	2160 ,
	300  ,
	1080 ,
	2400 ,
	600  ,
	576  ,
	288  ,
	832  ,
	48   ,
	192  ,
	1440 ,
	192  ,
	480  ,
	336  ,
	896  ,
	96   ,
	256  ,
	1536 ,
	192  ,
	384  ,
	384  ,
	1024 ,
	96   ,
	256  ,
	1536 ,
	192  ,
	336  ,
	432  ,
	1152 ,
	96   ,
	256  ,
	1536 ,
	192  ,
	768  ,
	480  ,
	1280 ,
	96   ,
	512  ,
	1584 ,
	384  ,
	288  ,
	180  ,
	360  ,
	36   ,
	144  ,
	936  ,
	144  ,
	432  ,
	216  ,
	432  ,
	54   ,
	144  ,
	936  ,
	144  ,
	189
};

#define FEATURE_WRITER_CYCLE(i) feature_writer_cycles[i]
#define FILTER_READER_CONV_CYCLE(i) filter_reader_conv_cycles[i] 
#define CONV_CYCLE(i) conv_cycles[i]
#define POOL_CYCLE(i) pool_cycles[i]

#define CONV_TOTAL_CYCLE 625082 
#define INPUT_READER_CYCLE 3876 
#define FILTER_PRELOAD_CYCLE 96 
#define FILTER_READER_CONV_TOTAL_CYCLE 151472
#define CONV_TOTAL_WRITE_CACHE 38164 
#define POOL_TOTAL_CYCLE 60389 
#define FEATURE_WRITER_TOTAL_CYCLE 24192 
#define END_POOL_TOTAL_CYCLE 448 

#endif

#ifndef STATIC_CYCLE

#define FEATURE_WRITER_CYCLE(i) FindFeatureWriterCycles(i)
#define FILTER_READER_CONV_CYCLE(i) FindFilterReaderConvCycles(i)
#define CONV_CYCLE(i) FindConvCycles(i)
#define POOL_CYCLE(i) FindPoolCycles(i)

#define CONV_TOTAL_CYCLE FindConvTotalCycles()
#define INPUT_READER_CYCLE FindInputReaderCycles()
#define FILTER_PRELOAD_CYCLE FindFilterPreloadCycles()
#define FILTER_READER_CONV_TOTAL_CYCLE FindFilterReaderConvTotalCycles()
#define CONV_TOTAL_WRITE_CACHE FindConvTotalWriteCache() 
#define POOL_TOTAL_CYCLE FindPoolTotalCycles() 
#define FEATURE_WRITER_TOTAL_CYCLE FindFeatureWriterTotalCycles() 
#define END_POOL_TOTAL_CYCLE FindEndPoolTotalCycles() 

#endif

#endif // __GOOGLENET__
