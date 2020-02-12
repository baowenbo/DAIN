#include <stdio.h>

#include "separableconvflow_cuda_kernel.cuh"


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
__global__ void SeparableConvFlowLayer_gpu_forward_kernelfunc(
		const int nElement,
		const int w, 		const int h, 		const int channel, const int filter_size,

		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,
		//const int output_b_stride, const int output_c_stride, const int output_h_stride, const int output_w_stride,
		const int flow_output_b_stride, const int flow_output_c_stride, const int flow_output_h_stride, const int flow_output_w_stride,

		const scalar_t* __restrict__   input1,    		const scalar_t* __restrict__   input2,    	const scalar_t* __restrict__   input3, 	 scalar_t*  flow_output

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
 
		float flow_y = 0.0f;
		float sum_weights = 0.0f;
		for (  int intFilterY = 0; intFilterY < filter_size; intFilterY += 1) {
			float temp2 = input2[batch_i * input2_b_stride + intFilterY * input2_c_stride + h_i * input2_h_stride + w_i ];
			flow_y += (float)(intFilterY) * temp2 ;
			sum_weights += 			temp2;
		}
		//sum_weights = fabs(sum_weights);
		flow_y = flow_y / sum_weights - ((float)(filter_size)-1.0)/2.0;
		flow_output[batch_i * flow_output_b_stride + 1 * flow_output_c_stride+ h_i* flow_output_h_stride + w_i] = 
					fabs(sum_weights) > 0.0f ?  flow_y : -2000;

		float flow_x = 0.0f;
		float sum_weights_x = 0.0f;
		for (   int intFilterX = 0; intFilterX < filter_size; intFilterX += 1) {
			float temp3 = input3[batch_i * input3_b_stride + intFilterX * input3_c_stride + h_i * input3_h_stride + w_i ];
			flow_x += (float)(intFilterX)  * temp3;
			sum_weights_x += 		 temp3;
		}
		//sum_weights_x = fabs(sum_weights_x);
		flow_x = flow_x / sum_weights_x - ((float)(filter_size)-1.0)/2.0;
		// what if the sum_weight is less than zeros.
		flow_output[batch_i * flow_output_b_stride + 0 * flow_output_c_stride + h_i* flow_output_h_stride + w_i] =
					fabs(sum_weights_x) >0.0f ? flow_x : -2000;
	}
	return ;

}


template <typename scalar_t>
__global__ void SeparableConvFlowLayer_gpu_backward_kernelfunc(
		const int nElement, 	   const int w, 		const int h, 		const int channel, const int filter_size,
		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,
		//const int output_b_stride, const int output_c_stride, const int output_h_stride, const int output_w_stride,
		const int flow_output_b_stride, const int flow_output_c_stride, const int flow_output_h_stride, const int flow_output_w_stride,

		const scalar_t* __restrict__      input1,        		const scalar_t* __restrict__    input2,		const scalar_t* __restrict__      input3,
		const scalar_t* __restrict__      gradflow_output,    		scalar_t*  gradinput1,  		scalar_t*  gradinput2,  		scalar_t*  gradinput3
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
		float flow_y = 0.0f;
		float sum_weights = 0.0f;
		
		for ( int  intFilterY = 0; intFilterY < filter_size; intFilterY += 1) {
			float temp2 = input2[batch_i * input2_b_stride + intFilterY * input2_c_stride + h_i * input2_h_stride + w_i ];
			flow_y += (float)(intFilterY) * temp2 ;
			sum_weights += 			temp2;
		}
		//flow_y = flow_y / sum_weights - ((float)(filter_size)-1.0)/2.0;
		//flow_output_data[batch_i * flow_output_b_stride + 1 * flow_output_c_stride+ h_i* flow_output_h_stride + w_i] = 
		//		sum_weights >0.0f ?  flow_y : -2000;
		//float sign = sum_weights >0.0f ? 1.0f : -1.0f;
		//sum_weights = fabs(sum_weights);
		if(fabs(sum_weights) >0.0f ){
			float gradflow_y = gradflow_output[batch_i * flow_output_b_stride + 1* flow_output_c_stride + 
								h_i * flow_output_h_stride + w_i ] ;					
			float offset = flow_y / ( sum_weights * sum_weights);
			for (int  intFilterY = 0; intFilterY < filter_size; intFilterY += 1) {
				gradinput2[batch_i * input2_b_stride + intFilterY * input2_c_stride  +  h_i * input2_h_stride + w_i ] =
							gradflow_y *  ((float)(intFilterY) / sum_weights -  offset);
			}
		}
		
		
		
		float flow_x = 0.0f;
		float sum_weights_x = 0.0f;
		for ( int  intFilterX = 0; intFilterX < filter_size; intFilterX += 1) {
			float temp3 = input3[batch_i * input3_b_stride + intFilterX * input3_c_stride + h_i * input3_h_stride + w_i ];
			flow_x += (float)(intFilterX)  * temp3;
			sum_weights_x += 		 temp3;
		}
		//flow_x = flow_x / sum_weights_x - ((float)(filter_size)-1.0)/2.0;
		//flow_output_data[batch_i * flow_output_b_stride + 0 * flow_output_c_stride + h_i* flow_output_h_stride + w_i] =
		//			sum_weights_x >0 ? flow_x : -2000;
		//float sign_x = sum_weights_x >0.0f ? 1.0f : -1.0f;
		//sum_weights_x = fabs(sum_weights_x);	
		if(fabs(sum_weights_x) > 0.0f ){
			 float gradflow_x = gradflow_output[batch_i * flow_output_b_stride + 0 * flow_output_c_stride + 
									h_i * flow_output_h_stride + w_i];
			float offset  = flow_x / (sum_weights_x * sum_weights_x);
			for ( int intFilterX = 0; intFilterX < filter_size; intFilterX += 1) {
				gradinput3[batch_i * input3_b_stride + intFilterX * input3_c_stride + h_i * input3_h_stride + w_i ] +=
						gradflow_x * ((float)(intFilterX) /sum_weights_x - offset);
			}
		}
	}
	return ;

}


int SeparableConvFlowLayer_gpu_forward_kernel(
		cudaStream_t stream,
		const int nElement,
		const int w, 		const int h, 		const int channel, 		const int batch,const int filter_size,

		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,
		//const int output_b_stride, const int output_c_stride, const int output_h_stride, const int output_w_stride,
		const int flow_output_b_stride, const int flow_output_c_stride, const int flow_output_h_stride, const int flow_output_w_stride,

		at::Tensor&  input1,    		at::Tensor&  input2,    	at::Tensor&  input3,   at::Tensor&  flow_output

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
SeparableConvFlowLayer_gpu_forward_kernelfunc<<<grid,block,0, stream >>>(
			nElement, //to let the nummous
			w,h,channel, filter_size,
			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,
			input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride,
			//output_b_stride,output_c_stride,output_h_stride,output_w_stride,
			flow_output_b_stride,flow_output_c_stride,flow_output_h_stride,flow_output_w_stride,

			input1.data<scalar_t>(),input2.data<scalar_t>(),input3.data<scalar_t>(), flow_output.data<scalar_t>()
			);
 					}));

	//			THCudaCheck(cudaGetLastError());
	cudaError_t err = cudaGetLastError();

	if (err != cudaSuccess) {
		printf("gpuerror in SeparableConvFlowLayer_gpu_forward_kernel: %s\n", cudaGetErrorString(err));
		//THError("aborting");
		return error;
	}

	error = 0;
	return error;

}


int SeparableConvFlowLayer_gpu_backward_kernel(
		cudaStream_t stream,
		const int nElement,
		const int w,    		const int h,    		const int channel,  		const int batch, const int filter_size,
		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,
		//const int output_b_stride, const int output_c_stride, const int output_h_stride, const int output_w_stride,
		const int flow_output_b_stride, const int flow_output_c_stride, const int flow_output_h_stride, const int flow_output_w_stride,

		at::Tensor&  input1,        		at::Tensor&  input2,		at::Tensor&  input3,

		at::Tensor&  gradflow_output,    		at::Tensor&  gradinput1,  		at::Tensor&  gradinput2,  		at::Tensor&  gradinput3
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

	SeparableConvFlowLayer_gpu_backward_kernelfunc <<<grid,block,0, stream>>>(
			nElement, //to let the nummous
			w,h,channel, filter_size,
			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,
			input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride,
			//output_b_stride,output_c_stride,output_h_stride,output_w_stride,
			flow_output_b_stride,flow_output_c_stride,flow_output_h_stride,flow_output_w_stride,


			input1.data<scalar_t>(), 			input2.data<scalar_t>(),         input3.data<scalar_t>(),  			gradflow_output.data<scalar_t>(),
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




 
