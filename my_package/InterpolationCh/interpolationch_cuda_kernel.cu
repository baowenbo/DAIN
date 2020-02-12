#include <stdio.h>

#include "interpolationch_cuda_kernel.cuh"


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
__global__ void InterpolationChLayer_gpu_forward_kernelfunc(
		const int nElement,
		const int w,
		const int h,
		const int channel,
		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,

		const scalar_t* __restrict__ input1,
		const scalar_t* __restrict__ input2,
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
	const float fillvalue =0.0f;

	if( withinXbounds && withinYbounds) {

		float fx = input2[batch_i * input2_b_stride + 0 * input2_c_stride + h_i * input2_h_stride + w_i  ];
		float fy = input2[batch_i * input2_b_stride + 1 * input2_c_stride + h_i * input2_h_stride + w_i  ];

		float x2 = (float)(w_i) + fx;
		float y2 = (float)(h_i) + fy;

		if(x2 >= 0.0f && y2 >=0.0f && x2 < (float)w && y2 < (float)h){
			int ix2_L = int(x2);
			int iy2_T = int(y2);
			int ix2_R = min(ix2_L + 1, w - 1);
			int iy2_B = min(iy2_T + 1, h - 1);

			float alpha = x2 - ix2_L;
			float beta = y2 - iy2_T;

			for(int c_i = 0 ; c_i < channel ; c_i ++){
				float TL = input1[off + c_i * input1_c_stride + iy2_T * input1_h_stride + ix2_L];
				float TR = input1[off + c_i * input1_c_stride + iy2_T * input1_h_stride + ix2_R];
				float BL = input1[off + c_i * input1_c_stride + iy2_B * input1_h_stride + ix2_L];
				float BR = input1[off + c_i * input1_c_stride + iy2_B * input1_h_stride + ix2_R];
				output[off + c_i * input1_c_stride + h_i * input1_h_stride + w_i] =
					(1- alpha ) *(1-beta) *TL + alpha *(1- beta) * TR + (1-alpha) *beta *BL + alpha *beta * BR;
			}
		} else{
			//the warping data is out of range, we fill it with zeros
			for(int c_i = 0 ;  c_i < channel; c_i ++){
				output[off + c_i * input1_c_stride + h_i * input1_h_stride + w_i] = fillvalue;
			}
		}
	}

	return ;

}

template <typename scalar_t>
__global__ void InterpolationChLayer_gpu_backward_kernelfunc(
		const int nElement,
		const int w,
		const int h,
		const int channel,
		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,

		const scalar_t* __restrict__  input1,
		const scalar_t* __restrict__  input2,
		const scalar_t* __restrict__  gradoutput,
		scalar_t*  gradinput1,
		scalar_t*  gradinput2
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

		float fx= input2[batch_i * input2_b_stride + 0 * input2_c_stride + h_i * input2_h_stride + w_i ];
		float fy = input2[batch_i * input2_b_stride + 1* input2_c_stride + h_i * input2_h_stride + w_i];

		float x2 = float(w_i) + fx;
		float y2 = float(h_i) + fy;

		if(x2 >= 0.0f  && y2 >= 0.0f && x2 < (float)w && y2 < (float)h){
			int ix2_L = int(x2);
			int iy2_T = int(y2);

			int ix2_R  = min(ix2_L+ 1, w - 1);
			int iy2_B = min(iy2_T + 1, h - 1);

			float alpha = x2 - ix2_L;
			float beta = y2 - iy2_T;

			for (int c_i = 0 ; c_i < channel; c_i++){
				float gradoutput_value = gradoutput[off + c_i * input1_c_stride + h_i * input1_h_stride + w_i];

				atomicAdd( & gradinput1[off + c_i * input1_c_stride + iy2_T * input1_h_stride + ix2_L], gradoutput_value * ( 1- alpha) * (1- beta));
				atomicAdd( & gradinput1[off + c_i * input1_c_stride + iy2_T * input1_h_stride + ix2_R], gradoutput_value * alpha * (1-beta));
				atomicAdd( & gradinput1[off + c_i * input1_c_stride + iy2_B * input1_h_stride + ix2_L], gradoutput_value * (1-alpha ) * beta);
				atomicAdd( & gradinput1[off + c_i * input1_c_stride + iy2_B * input1_h_stride + ix2_R], gradoutput_value * alpha * beta);

			}

			float gamma  = iy2_B - y2;

			float bot_diff = 0.0f;
			for(int c_i =0 ; c_i< channel; c_i ++ ){
				float temp = 0;
				temp += gamma * (input1[off + c_i * input1_c_stride + iy2_T * input1_h_stride +ix2_R] -
						input1[off + c_i* input1_c_stride+ iy2_T * input1_h_stride + ix2_L]);
				temp += (1 - gamma) *( input1[off + c_i * input1_c_stride + iy2_B * input1_h_stride + ix2_R] -
						input1[off + c_i * input1_c_stride + iy2_B * input1_h_stride + ix2_L]);

				float warped_diff_value = gradoutput[off+ c_i * input1_c_stride+ h_i* input1_h_stride + w_i];
				bot_diff += warped_diff_value * temp  ;


			}
			//the gradients of the x direction/ horizontal direction
			gradinput2[batch_i * input2_b_stride + 0 * input2_c_stride + h_i * input2_h_stride + w_i] = bot_diff;

			gamma = ix2_R- x2;
			bot_diff = 0.0f;
			for(int c_i = 0 ; c_i < channel;c_i ++ ){
				float temp = 0.0f;
				temp += gamma    * (input1[off + c_i * input1_c_stride + iy2_B * input1_h_stride + ix2_L] -
						input1[off + c_i * input1_c_stride + iy2_T * input1_h_stride + ix2_L]);

				temp += (1-gamma) *( input1[off + c_i * input1_c_stride+ iy2_B* input1_h_stride+ix2_R] -
						input1[off+ c_i* input1_c_stride+ iy2_T * input1_h_stride +ix2_R]);

				float warped_diff_value = gradoutput[off + c_i * input1_c_stride + h_i * input1_h_stride + w_i];
				bot_diff += warped_diff_value * temp;


			}
			gradinput2[batch_i * input2_b_stride + 1 * input2_c_stride + h_i * input2_h_stride + w_i]= bot_diff;

		}


	}
	return ;

}



int InterpolationChLayer_gpu_forward_kernel(
		cudaStream_t stream,
		const int nElement,
		const int w,
		const int h,
		const int channel,
		const int batch,
		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,

		at::Tensor&  input1,
		at::Tensor&  input2,
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
	//extract the data of CudaTensor and use kernel to calculate.
		AT_DISPATCH_FLOATING_TYPES(input1.type(), "InterpolationChLayer_gpu_forward_kernelfunc", ([&] {
	InterpolationChLayer_gpu_forward_kernelfunc<<<grid,block,0, stream >>>(
			nElement, //to let the nummous
			w,h,channel,
			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,

			input1.data<scalar_t>(),input2.data<scalar_t>(),output.data<scalar_t>()
			);
 					}));

	//			THCudaCheck(cudaGetLastError());
	cudaError_t err = cudaGetLastError();

	if (err != cudaSuccess) {
		printf("gpuerror in BilinearSampler.updateOutput: %s\n", cudaGetErrorString(err));
		//THError("aborting");
		return error;
	}

	error = 0;
	return error;

}

int InterpolationChLayer_gpu_backward_kernel(
		cudaStream_t stream,
		const int nElement,
		const int w,
		const int h,
		const int channel,
		const int batch,
		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,

		at::Tensor&  input1,
		at::Tensor&  input2,
		at::Tensor&  gradoutput,
		at::Tensor&  gradinput1,
		at::Tensor&  gradinput2
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
			AT_DISPATCH_FLOATING_TYPES(input1.type(), "InterpolationChLayer_gpu_backward_kernelfunc", ([&] {
InterpolationChLayer_gpu_backward_kernelfunc <<<grid,block,0, stream>>>(
			nElement, //to let the nummous
			w,h,channel,
			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,

			input1.data<scalar_t>(),
			input2.data<scalar_t>(),
			gradoutput.data<scalar_t>(),
			gradinput1.data<scalar_t>(),
			gradinput2.data<scalar_t>()
			);
 					}));

	cudaError_t err = cudaGetLastError();

	if (err != cudaSuccess) {
		printf("gpu error in BilinearSampler.updateGradInput %s\n", cudaGetErrorString(err));
		//THError("aborting");
		return error;
	}

	error = 0;
	return error;

}
