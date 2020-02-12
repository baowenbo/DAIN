#include <stdio.h>

#include "separableconv_cuda_kernel.cuh"


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
__global__ void SeparableConvLayer_gpu_forward_kernelfunc(
		const int nElement,
		const int w, 		const int h, 		const int channel, const int filter_size,

		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,
		const int output_b_stride, const int output_c_stride, const int output_h_stride, const int output_w_stride,

		const scalar_t* __restrict__  input1,    		const scalar_t* __restrict__  input2,    	const scalar_t* __restrict__  input3, 	scalar_t*  output

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
	const bool withinXbounds = w_i < w - filter_size + 1;
	const bool withinYbounds = h_i < h - filter_size + 1;

	const int batch_i = blockIdx.z;


	//    __syncthreads();
//	const float fillvalue =0.0f;

	if( withinXbounds && withinYbounds) {

		for ( int c_i = 0 ; c_i < channel ; c_i ++){

			float out = 0.0f;
			for (int intFilterY = 0; intFilterY < filter_size; intFilterY += 1) {
			for (int intFilterX = 0; intFilterX < filter_size; intFilterX += 1) {
				float temp1 = input1[batch_i * input1_b_stride + c_i * input1_c_stride + (h_i + intFilterY )* input1_h_stride + (w_i + intFilterX)];
				float temp2 = input2[batch_i * input2_b_stride + intFilterY * input2_c_stride + h_i * input2_h_stride + w_i ];
				float temp3 = input3[batch_i * input3_b_stride + intFilterX * input3_c_stride + h_i * input3_h_stride + w_i ];
				out += temp1* temp2 * temp3;
			}
			}
			output[batch_i * output_b_stride + c_i* output_c_stride + h_i * output_h_stride + w_i ] = out;
		}
	}
	return ;

}
 

template <typename scalar_t>
__global__ void SeparableConvLayer_gpu_backward_kernelfunc(
		const int nElement, 	   const int w, 		const int h, 		const int channel, const int filter_size,
		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,
		const int output_b_stride, const int output_c_stride, const int output_h_stride, const int output_w_stride,

		const scalar_t* __restrict__  input1,        		const scalar_t* __restrict__  input2,		const scalar_t* __restrict__  input3,
		const scalar_t* __restrict__  gradoutput,    		scalar_t*  gradinput1,  		scalar_t*  gradinput2,  		scalar_t* gradinput3
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
	const bool withinXbounds = w_i < w - filter_size + 1;
	const bool withinYbounds = h_i < h - filter_size + 1;

	const int batch_i = blockIdx.z;

	if(withinXbounds && withinYbounds){

		for (int c_i = 0 ; c_i < channel ; c_i ++){
				for (int   intFilterY = 0; intFilterY < filter_size; intFilterY += 1) {
				for ( int  intFilterX = 0; intFilterX < filter_size; intFilterX += 1) {
					float temp1 = input1[batch_i * input1_b_stride + c_i * input1_c_stride + (h_i + intFilterY )* input1_h_stride + (w_i + intFilterX)];
					float temp2 = input2[batch_i * input2_b_stride + intFilterY * input2_c_stride + h_i * input2_h_stride + w_i ];
					float temp3 = input3[batch_i * input3_b_stride + intFilterX * input3_c_stride + h_i * input3_h_stride + w_i ];

					float gradout = gradoutput[batch_i * output_b_stride + c_i* output_c_stride + h_i * output_h_stride + w_i ];

					atomicAdd(&gradinput1[batch_i * input1_b_stride + c_i * input1_c_stride + (h_i + intFilterY )* input1_h_stride + (w_i + intFilterX)],
						gradout * temp2 * temp3);
					atomicAdd(&gradinput2[batch_i * input2_b_stride + intFilterY * input2_c_stride  +  h_i * input2_h_stride + w_i ],
						gradout * temp1 * temp3);
					atomicAdd(&gradinput3 [batch_i * input3_b_stride + intFilterX * input3_c_stride + h_i * input3_h_stride + w_i ] ,
						gradout * temp1 * temp2);
				}
				}
		}

	}
	return ;

}



int SeparableConvLayer_gpu_forward_kernel(
		cudaStream_t stream,
		const int nElement,
		const int w, 		const int h, 		const int channel, 		const int batch,const int filter_size,

		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,
		const int output_b_stride, const int output_c_stride, const int output_h_stride, const int output_w_stride,

		at::Tensor&  input1,    		at::Tensor&  input2,    	at::Tensor&  input3, 	at::Tensor&  output

		)
{
	int error = 1 ;

	dim3 grid;
	dim3 block;


	//		blockthread = 128;
	//the threadIdx.x is sheduled first, then threadIdx.y, threadIdx.z
	//the three channels are processsed in one kernel
	block  = dim3(BLOCKDIMX,BLOCKDIMY,1);
	grid = dim3( (w  - filter_size + 1 + BLOCKDIMX - 1)/ BLOCKDIMX, (h  - filter_size + 1 + BLOCKDIMY - 1) / BLOCKDIMY, batch);
    if(BLOCKDIMX != 32 || BLOCKDIMY != 16||DEBUG)
        printf("BLOCKDIMX revised to %d, BLOCKDIMY revised to %d \n", BLOCKDIMX,BLOCKDIMY);
	//extract the data of CudaTensor and use kernel to calculate.
				AT_DISPATCH_FLOATING_TYPES(input1.type(), "DepthFlowProjection_gpu_backward", ([&] {
SeparableConvLayer_gpu_forward_kernelfunc<<<grid,block,0, stream >>>(
			nElement, //to let the nummous
			w,h,channel, filter_size,
			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,
			input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride,
			output_b_stride,output_c_stride,output_h_stride,output_w_stride,

			input1.data<scalar_t>(),input2.data<scalar_t>(),input3.data<scalar_t>(), output.data<scalar_t>()
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


int SeparableConvLayer_gpu_backward_kernel(
		cudaStream_t stream,
		const int nElement,
		const int w,    		const int h,    		const int channel,  		const int batch, const int filter_size,
		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,
		const int output_b_stride, const int output_c_stride, const int output_h_stride, const int output_w_stride,

		at::Tensor&  input1,        		at::Tensor&  input2,		at::Tensor&  input3,

		at::Tensor&  gradoutput,    		at::Tensor&  gradinput1,  		at::Tensor&  gradinput2,  		at::Tensor&  gradinput3
		)
{

	int error = 1 ;

	dim3 grid;
	dim3 block;


	//blockthread = 128;
	//the threadIdx.x is sheduled first, then threadIdx.y, threadIdx.z
	//the three channels are processsed in one kernel
	block  = dim3(BLOCKDIMX,BLOCKDIMY,1);
	grid = dim3( (w - filter_size + 1 + BLOCKDIMX - 1)/ BLOCKDIMX, (h  - filter_size + 1+ BLOCKDIMY - 1) / BLOCKDIMY, batch);
    if(BLOCKDIMX != 32 || BLOCKDIMY != 16||DEBUG)
        printf("BLOCKDIMX revised to %d, BLOCKDIMY revised to %d \n", BLOCKDIMX,BLOCKDIMY);

//    cudaMemset((void*)gradinput1, 0, input1_b_stride * batch * sizeof(float));
//    cudaMemset((void*)gradinput2, 0, input2_b_stride * batch * sizeof(float));
//    cudaMemset((void*)gradinput3, 0, input3_b_stride * batch * sizeof(float));

				AT_DISPATCH_FLOATING_TYPES(input1.type(), "DepthFlowProjection_gpu_backward", ([&] {
SeparableConvLayer_gpu_backward_kernelfunc <<<grid,block,0, stream>>>(
			nElement, //to let the nummous
			w,h,channel, filter_size,
			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,
			input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride,
			output_b_stride,output_c_stride,output_h_stride,output_w_stride,


			input1.data<scalar_t>(), 			input2.data<scalar_t>(),         input3.data<scalar_t>(),  			gradoutput.data<scalar_t>(),
			gradinput1.data<scalar_t>(), 			gradinput2.data<scalar_t>(),     gradinput3.data<scalar_t>()
			);
 					}));

	cudaError_t err = cudaGetLastError();

	if (err != cudaSuccess) {
		printf("gpuerror in BilinearSampler.updateGradInput %s\n", cudaGetErrorString(err));
		//THError("aborting");
		return error;
	}

	error = 0;
	return error;

}