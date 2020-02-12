#include <stdio.h>

#include "flowprojection_cuda_kernel.cuh"


#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>


#define min(a,b) ((a<b)?(a):(b))
#define max(a,b) ((a>b)?(a):(b))

#define DEBUG (0)
#ifndef BLOCKDIMX
#define BLOCKDIMX (32)
#endif
#ifndef BLOCKDIMY
#define BLOCKDIMY (16)
#endif
using at::Half;




//forward path of our layer
template <typename scalar_t>
__global__ void FlowProjection_gpu_forward_kernelfunc(
		const int nElement,
		const int w,
		const int h,
		const int channel,
		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int count_b_stride, const int count_c_stride, const int count_h_stride, const int count_w_stride,

		const scalar_t* __restrict__    input1,
		scalar_t*  count,
		scalar_t*  output
		)
{

	//blockIdx.z : batch index from 0~B-1
	//blockIdx.y : height patch index from ceil(h/16)
	//blockIdx.x : width patch index from ceil(w/32)

	//threadidx.x: width index 0~31
	//threadIdx.y: height index 0~15
	//threadIdx.z: Not used

	//only use one dimensioon of the grid and block
	const int w_i = blockIdx.x * blockDim.x + threadIdx.x;
	const int h_i = blockIdx.y * blockDim.y + threadIdx.y;
	const bool withinXbounds = w_i < w;
	const bool withinYbounds = h_i < h;

	const int batch_i = blockIdx.z;
	const int off = batch_i * input1_b_stride;

	//    __syncthreads();
//	const float fillvalue =0.0f;

	if( withinXbounds && withinYbounds) {
        float fx = input1[ off + 0 * input1_c_stride + h_i * input1_h_stride + w_i ];
        float fy = input1[ off + 1 * input1_c_stride + h_i * input1_h_stride + w_i ];

        float x2 = (float) (w_i) + fx;
        float y2 = (float) (h_i) + fy;
        if(x2>=0.0f && y2 >= 0.0f &&x2 <= (float) ( w-1) && y2 <= (float) (h -1 ) ){
            int ix2_L = (int) (x2);
            int iy2_T = (int) (y2);
            int ix2_R = min(ix2_L + 1, w - 1);
            int iy2_B = min(iy2_T + 1, h - 1);

            atomicAdd(&output[off + 0 * input1_c_stride + iy2_T * input1_h_stride + ix2_L ] ,-fx);
            atomicAdd(&output[off + 0 * input1_c_stride + iy2_T * input1_h_stride + ix2_R ],-fx);
            atomicAdd(&output[off + 0 * input1_c_stride + iy2_B * input1_h_stride + ix2_L ] ,-fx);
            atomicAdd(&output[off + 0 * input1_c_stride + iy2_B * input1_h_stride + ix2_R ],-fx);

            atomicAdd(&output[off + 1 * input1_c_stride + iy2_T * input1_h_stride + ix2_L] , -fy);
            atomicAdd(&output[off + 1 * input1_c_stride + iy2_T * input1_h_stride + ix2_R]  , -fy);
            atomicAdd(&output[off + 1 * input1_c_stride + iy2_B * input1_h_stride + ix2_L]  , -fy);
            atomicAdd(&output[off + 1 * input1_c_stride + iy2_B * input1_h_stride + ix2_R]  , -fy);

            atomicAdd(& count[batch_i * count_b_stride + 0 + iy2_T * count_h_stride + ix2_L], 1);
            atomicAdd(& count[batch_i * count_b_stride + 0 + iy2_T * count_h_stride + ix2_R] , 1);
            atomicAdd(& count[batch_i * count_b_stride + 0 + iy2_B * count_h_stride + ix2_L] , 1);
            atomicAdd(& count[batch_i * count_b_stride + 0 + iy2_B * count_h_stride + ix2_R] , 1);
        }
	}
	return ;

}
template <typename scalar_t>
__global__ void FlowProjectionAveraging_kernelfunc(
		const int nElement,
		const int w,
		const int h,
		const int channel,
		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int count_b_stride, const int count_c_stride, const int count_h_stride, const int count_w_stride,

		const scalar_t* __restrict__      input1,
		scalar_t* count,
		scalar_t* output
		)
{

	//blockIdx.z : batch index from 0~B-1
	//blockIdx.y : height patch index from ceil(h/16)
	//blockIdx.x : width patch index from ceil(w/32)

	//threadidx.x: width index 0~31
	//threadIdx.y: height index 0~15
	//threadIdx.z: Not used

	//only use one dimensioon of the grid and block
	const int w_i = blockIdx.x * blockDim.x + threadIdx.x;
	const int h_i = blockIdx.y * blockDim.y + threadIdx.y;
	const bool withinXbounds = w_i < w;
	const bool withinYbounds = h_i < h;

	const int batch_i = blockIdx.z;
	const int off = batch_i * input1_b_stride;

	//    __syncthreads();
//	const float fillvalue =0.0f;

	if( withinXbounds && withinYbounds) {
	    float temp =count[batch_i * count_b_stride + 0 + h_i * count_h_stride + w_i] ;
        if(temp > 0.0f){
            output[off + 0 * input1_c_stride + h_i * input1_h_stride + w_i ] /= temp;
            output[off + 1 * input1_c_stride + h_i * input1_h_stride + w_i ] /= temp;
        }
	}
	return ;

}

template <typename scalar_t>
__global__ void FlowFillhole_kernelfunc(
		const int nElement,
		const int w,
		const int h,
		const int channel,
		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int count_b_stride, const int count_c_stride, const int count_h_stride, const int count_w_stride,

		const scalar_t* __restrict__ input1,
		scalar_t*	count,
		scalar_t*	output
		)
{

	//blockIdx.z : batch index from 0~B-1
	//blockIdx.y : height patch index from ceil(h/16)
	//blockIdx.x : width patch index from ceil(w/32)

	//threadidx.x: width index 0~31
	//threadIdx.y: height index 0~15
	//threadIdx.z: Not used

	//only use one dimensioon of the grid and block
	const int w_i = blockIdx.x * blockDim.x + threadIdx.x;
	const int h_i = blockIdx.y * blockDim.y + threadIdx.y;
	const bool withinXbounds = w_i < w;
	const bool withinYbounds = h_i < h;

	const int batch_i = blockIdx.z;
	const int off = batch_i * input1_b_stride;

	//    __syncthreads();
//	const float fillvalue =0.0f;

	if( withinXbounds && withinYbounds) {
	    float temp = count[batch_i * count_b_stride + 0 + h_i * count_h_stride + w_i] ;
        if(temp <= 0.0f){
            //search along the four directions,0/90/180/270, until finding at least one
            int left_offset = w_i;            float left_temp = 0.0f;
            while(left_temp == 0.0f && left_offset - 1 >= 0){
                left_offset = left_offset - 1;
                left_temp = count[batch_i * count_b_stride + 0 + h_i * count_h_stride + left_offset] ;
            }

            int right_offset = w_i ;            float right_temp = 0.0f;
            while(right_temp ==0.0f && right_offset + 1 <= w - 1 ){
                right_offset  = right_offset + 1 ;
                right_temp =  count[batch_i * count_b_stride + 0 + h_i * count_h_stride + right_offset] ;
            }

            int up_offset = h_i ;            float up_temp = 0.0f;
            while(up_temp == 0.0f && up_offset - 1 >=0){
                up_offset = up_offset - 1;
                up_temp =  count[batch_i * count_b_stride + 0 + up_offset * count_h_stride + w_i ] ;
            }

            int down_offset = h_i;            float down_temp = 0.0f;
            while(down_temp == 0.0f && down_offset + 1 <= h - 1 ){
                down_offset = down_offset + 1;
                down_temp =  count[batch_i * count_b_stride + 0 + down_offset * count_h_stride + w_i] ;
            }

            if(left_temp + right_temp + up_temp + down_temp <=0.0f){
                //printf("Can't fill hole, find no neighbor vectors availabel\n");
                return;
            }

            left_temp = (left_temp > 0.0f)?1:0;
            right_temp = (right_temp > 0.0f)?1:0;
            up_temp = (up_temp > 0.0f)?1:0;
            down_temp = (down_temp > 0.0f)?1:0;

            output[off + 0 * input1_c_stride + h_i * input1_h_stride + w_i ] = (
                left_temp *  output[off + 0 * input1_c_stride + h_i * input1_h_stride + left_offset] +
                right_temp *  output[off + 0 * input1_c_stride + h_i * input1_h_stride + right_offset]+
                up_temp *  output[off + 0 * input1_c_stride + up_offset * input1_h_stride + w_i] +
                down_temp *  output[off + 0 * input1_c_stride + down_offset * input1_h_stride + w_i]
            )/(
                left_temp + right_temp + up_temp + down_temp
            ) ;


            output[off + 1 * input1_c_stride + h_i * input1_h_stride + w_i ] =(
                left_temp *  output[off + 1 * input1_c_stride + h_i * input1_h_stride + left_offset] +
                right_temp *  output[off + 1 * input1_c_stride + h_i * input1_h_stride + right_offset]+
                up_temp *  output[off + 1 * input1_c_stride + up_offset * input1_h_stride + w_i] +
                down_temp *  output[off + 1 * input1_c_stride + down_offset * input1_h_stride + w_i]
            )/(
                left_temp + right_temp + up_temp + down_temp
            ) ;
        }
	}
	return ;

}
template <typename scalar_t>
__global__ void FlowProjection_gpu_backward_kernelfunc(
		const int nElement,  	const int w, 	const int h, const int channel,
		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int count_b_stride, const int count_c_stride, const int count_h_stride, const int count_w_stride,

		const scalar_t* __restrict__        input1,
		const scalar_t* __restrict__       count,
		const scalar_t* __restrict__       gradoutput,
		scalar_t*   gradinput1
		)
{
	//blockIdx.z : batch index from 0~B-1
	//blockIdx.y : height patch index from ceil(h/16)
	//blockIdx.x : width patch index from ceil(w/32)

	//threadidx.x: width index 0~31
	//threadIdx.y: height index 0~15
	//threadIdx.z: Not used

	const int w_i = blockIdx.x * blockDim.x + threadIdx.x;
	const int h_i = blockIdx.y * blockDim.y + threadIdx.y;
	const bool withinXbounds = w_i < w;
	const bool withinYbounds = h_i < h;

	const int batch_i = blockIdx.z;
	const int off  = batch_i * input1_b_stride;

	//    __syncthreads();

	if(withinXbounds && withinYbounds){
        float fx = input1[off + 0 * input1_c_stride + h_i * input1_h_stride + w_i] ;
        float fy = input1[off + 1 * input1_c_stride + h_i * input1_h_stride + w_i] ;

        float x2 = (float) ( w_i ) + fx;
        float y2 = (float) ( h_i ) + fy;
        if( x2 >=0.0f && y2 >= 0.0f && x2 <= (float) (w -1) && y2 <= (float) (h-1)){
            int ix2_L = (int)(x2);
            int iy2_T = (int)(y2);
            int ix2_R  = min(ix2_L + 1, w-1);
            int iy2_B  = min(iy2_T + 1, h-1);

            int iu_offset = off + 0 * input1_c_stride + h_i * input1_h_stride + w_i;
            gradinput1[iu_offset] += -  gradoutput[off +  0 * input1_c_stride + iy2_T * input1_h_stride + ix2_L]/
                                        count[batch_i * count_b_stride + 0+ iy2_T * count_h_stride + ix2_L]  ;
            gradinput1[iu_offset] += -    gradoutput[off + 0 * input1_c_stride + iy2_T * input1_h_stride + ix2_R ]/
                                         count[batch_i * count_b_stride +0 + iy2_T * count_h_stride  + ix2_R]          ;
            gradinput1[iu_offset ] += -  gradoutput[off + 0 * input1_c_stride + iy2_B * input1_h_stride + ix2_L]/
                                         count[batch_i * count_b_stride + 0 + iy2_B * count_h_stride + ix2_L]  ;
            gradinput1[iu_offset ]  += -  gradoutput[off + 0 * input1_c_stride + iy2_B * input1_h_stride + ix2_R]/
                                         count[batch_i * count_b_stride + 0+ iy2_B * count_h_stride + ix2_R]   ;

            int iv_offset = off + 1 * input1_c_stride + h_i * input1_h_stride + w_i;
            gradinput1[iv_offset] += -  gradoutput[off + 1 * input1_c_stride + iy2_T * input1_h_stride + ix2_L]/
                                         count[batch_i * count_b_stride + 0 + iy2_T * count_h_stride + ix2_L]  ;
            gradinput1[iv_offset] += - gradoutput[off + 1 * input1_c_stride + iy2_T * input1_h_stride + ix2_R]/
                                         count[batch_i * count_b_stride + 0 + iy2_T * count_h_stride + ix2_R]  ;
            gradinput1[iv_offset] += -  gradoutput[off + 1 * input1_c_stride + iy2_B * input1_h_stride + ix2_L]/
                                    count[batch_i * count_b_stride + 0 + iy2_B * count_h_stride + ix2_L]     ;
            gradinput1[iv_offset] += -  gradoutput[off + 1 * input1_c_stride + iy2_B * input1_h_stride + ix2_R]/
                                    count[batch_i * count_b_stride + 0 + iy2_B * count_h_stride + ix2_R]   ;
        }
	}
	return ;

}


int FlowProjection_gpu_forward_kernel(
		cudaStream_t stream, 		const int nElement,
		const int w, 		const int h, 		const int channel, 		const int batch, const int fillhole,
		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int count_b_stride, const int count_c_stride, const int count_h_stride, const int count_w_stride,

		at::Tensor&  input1,
		at::Tensor&  count,
		at::Tensor&  output
		)
{
    int error = 1 ;


	dim3 grid;
	dim3 block;


	//		blockthread = 128;
	//the threadIdx.x is sheduled first, then threadIdx.y, threadIdx.z
	//the three channels are processsed in one kernel
	block  = dim3(BLOCKDIMX,BLOCKDIMY,1);
	grid = dim3( (w + BLOCKDIMX - 1)/ BLOCKDIMX, (h + BLOCKDIMY - 1) / BLOCKDIMY, batch);
    if(BLOCKDIMX != 32 || BLOCKDIMY != 16||DEBUG)
        printf("BLOCKDIMX revised to %d, BLOCKDIMY revised to %d \n", BLOCKDIMX,BLOCKDIMY);
//    printf("I am here\n");
	//extract the data of CudaTensor and use kernel to calculate.

	AT_DISPATCH_FLOATING_TYPES(input1.type(), "FlowProjection_gpu_forward_kernelfunc", ([&] {
	FlowProjection_gpu_forward_kernelfunc<<<grid,block,0, stream >>>(
			nElement, //to let the nummous
			w,h,channel,
			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			count_b_stride,count_c_stride,count_h_stride,count_w_stride,

			input1.data<scalar_t>(),count.data<scalar_t>(),output.data<scalar_t>()
			);
								}));

    cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("gpuerror in BilinearSampler.updateOutput: %s\n", cudaGetErrorString(err));
		//THError("aborting");
		return error;
	}
//    printf("I am there\n");
	AT_DISPATCH_FLOATING_TYPES(input1.type(), "FlowProjectionAveraging_kernelfunc", ([&] {

    FlowProjectionAveraging_kernelfunc<<<grid,block,0,stream>>>(
    		nElement, //to let the nummous
			w,h,channel,
			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			count_b_stride,count_c_stride,count_h_stride,count_w_stride,

			input1.data<scalar_t>(),count.data<scalar_t>(),output.data<scalar_t>()
    );				
	}));

//    printf("I am kao\n");

	//			THCudaCheck(cudaGetLastError());
    err = cudaGetLastError();

	if (err != cudaSuccess) {
		printf("gpuerror in BilinearSampler.updateOutput: %s\n", cudaGetErrorString(err));
		//THError("aborting");
		return error;
	}
//    printf("I am dd\n");

    if(fillhole){

//        printf("use flow fill hole\n");
    	AT_DISPATCH_FLOATING_TYPES(input1.type(), "FlowFillhole_kernelfunc", ([&] {
    FlowFillhole_kernelfunc<<<grid,block,0,stream>>>(
    		nElement, //to let the nummous
			w,h,channel,
			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			count_b_stride,count_c_stride,count_h_stride,count_w_stride,

			input1.data<scalar_t>(),count.data<scalar_t>(),output.data<scalar_t>()
        );
					}));

    err = cudaGetLastError();

	if (err != cudaSuccess) {
		printf("gpuerror in BilinearSampler.updateOutput: %s\n", cudaGetErrorString(err));
		return error;
	}

    }

	error = 0;
	return error;

}


int FlowProjection_gpu_backward_kernel(
		cudaStream_t stream,
		const int nElement,
		const int w,
		const int h,
		const int channel,
		const int batch,
		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int count_b_stride, const int count_c_stride, const int count_h_stride, const int count_w_stride,

		at::Tensor&  input1,
		at::Tensor&  count,
		at::Tensor&  gradoutput,
		at::Tensor&  gradinput1
		)
{

	int error = 1 ;

	dim3 grid;
	dim3 block;

	//blockthread = 128;
	//the threadIdx.x is sheduled first, then threadIdx.y, threadIdx.z
	//the three channels are processsed in one kernel
	block  = dim3(BLOCKDIMX,BLOCKDIMY,1);
	grid = dim3( (w + BLOCKDIMX - 1)/ BLOCKDIMX, (h + BLOCKDIMY - 1) / BLOCKDIMY, batch);
    if(BLOCKDIMX != 32 || BLOCKDIMY != 16||DEBUG)
        printf("BLOCKDIMX revised to %d, BLOCKDIMY revised to %d \n", BLOCKDIMX,BLOCKDIMY);
    	AT_DISPATCH_FLOATING_TYPES(input1.type(), "FlowProjection_gpu_backward_kernelfunc", ([&] {
	FlowProjection_gpu_backward_kernelfunc <<<grid,block,0, stream>>>(
			nElement, //to let the nummous
			w,h,channel,
			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			count_b_stride,count_c_stride,count_h_stride,count_w_stride,

			input1.data<scalar_t>(),
			count.data<scalar_t>(),
			gradoutput.data<scalar_t>(),
			gradinput1.data<scalar_t>()
			);
		}));

//    printf("gpu I am there\n");

	cudaError_t err = cudaGetLastError();

	if (err != cudaSuccess) {
		printf("gpu error in BilinearSampler.updateGradInput %s\n", cudaGetErrorString(err));
		//THError("aborting");
		return error;
	}
//    printf("gpu I am here\n");

	error = 0;
	return error;


}
