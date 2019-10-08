#include <torch/torch.h>
#include <ATen/ATen.h>
#include <stdio.h>
#include <iostream>
#include <ATen/cuda/CUDAContext.h> //works for 1.0.0

#include "separableconv_cuda_kernel.cuh"


int SeparableConvLayer_gpu_forward(
		at::Tensor&  input1,
		at::Tensor&  input2,
		at::Tensor&  input3,
		at::Tensor&  output

		)
		{
	int error = 1 ;

	int channel = input1.size( 1);
	if(channel!=3) return error;
	int batch = input1.size(0);
	if(input2.size( 0) != batch) return error;
	if(input2.size(1) != input3.size(1)) return error; //change by zhenghe, am I right?

	int h = input1.size(2);
	int w = input1.size(3);
	if(input2.size(2) != h - input2.size(1) + 1) return error;// to add some checkpoint
	if(input2.size(3) != w - input2.size(1) + 1) return error;


	int input1_b_stride = input1.stride(0);
	int input1_c_stride = input1.stride(1);
	int input1_h_stride = input1.stride(2);
	int input1_w_stride = input1.stride(3);

	int input2_b_stride = input2.stride(0);
	int input2_c_stride = input2.stride(1);
	int input2_h_stride = input2.stride(2);
	int input2_w_stride = input2.stride(3);

    int input3_b_stride = input3.stride(0);
	int input3_c_stride = input3.stride(1);
	int input3_h_stride = input3.stride(2);
	int input3_w_stride = input3.stride(3);

    int output_b_stride = output.stride(0);
	int output_c_stride = output.stride(1);
	int output_h_stride = output.stride(2);
	int output_w_stride = output.stride(3);
//    printf("filter tensor shape: %d,%d,%d,%d\n", input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride);


	//TODO: do we need to assert the w_stride to be 1
    if(input1_w_stride !=1) return error;
	if(input2_w_stride !=1) return error;
    if(input3_w_stride !=1) return error;
    if(output_w_stride !=1) return error;

	if(input2_b_stride != input3_b_stride) return error;
	if(input2_c_stride != input3_c_stride) return error;


	int	nElement = 0;//UNUSED  THCudaTensor_nElement(state, output);


	error = SeparableConvLayer_gpu_forward_kernel(
//			at::globalContext().getCurrentCUDAStream(), //works for 0.4.1
           at::cuda::getCurrentCUDAStream(), //works for 1.0.0
			nElement,w,h,channel,batch,  input2.size(1),

			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,
			input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride,
			output_b_stride,output_c_stride,output_h_stride,output_w_stride,



			input1,
			input2,
			input3,
			output);
	  if (error) {AT_ERROR("CUDA call failed");}

	return error;

		}
int SeparableConvLayer_gpu_backward(
		at::Tensor&  input1,
		at::Tensor&  input2,
		at::Tensor&  input3,
		at::Tensor&  gradoutput,
		at::Tensor&  gradinput1,
		at::Tensor&  gradinput2,
		at::Tensor&  gradinput3
		)
		{


    int error = 1 ;
	int channel = input1.size( 1);
	if(channel!=3) return error;
	int batch = input1.size(0);
	if(input2.size( 0) != batch) return error;
	if(input2.size(1) != input2.size(1)) return error;

	int h = input1.size(2);
	int w = input1.size(3);
	if(input2.size(2) != h - input2.size(1) + 1) return error;// to add some checkpoint
	if(input2.size(3) != w - input2.size(1) + 1) return error;


	int input1_b_stride = input1.stride(0);
	int input1_c_stride = input1.stride(1);
	int input1_h_stride = input1.stride(2);
	int input1_w_stride = input1.stride(3);

	int input2_b_stride = input2.stride(0);
	int input2_c_stride = input2.stride(1);
	int input2_h_stride = input2.stride(2);
	int input2_w_stride = input2.stride(3);

    int input3_b_stride = input3.stride(0);
	int input3_c_stride = input3.stride(1);
	int input3_h_stride = input3.stride(2);
	int input3_w_stride = input3.stride(3);

    int output_b_stride = gradoutput.stride(0);
	int output_c_stride = gradoutput.stride(1);
	int output_h_stride = gradoutput.stride(2);
	int output_w_stride = gradoutput.stride(3);

//    printf("filter tensor shape: %d,%d,%d,%d\n", input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride);


	//TODO: do we need to assert the w_stride to be 1
	if(input1_w_stride !=1) return error;
	if(input2_w_stride !=1) return error;
    if(input3_w_stride !=1) return error;
    if(output_w_stride !=1) return error;

    if(input1_b_stride != gradinput1.stride(0)) return error;
	if(input2_b_stride != gradinput2.stride(0)) return error;
	if(input1_c_stride != gradinput1.stride(1)) return error;
	if(input2_c_stride != gradinput2.stride(1)) return error;
	if(input3_c_stride != gradinput3.stride(1)) return error;

//    printf("GPU backward: %d,%d,%d,%d\n", input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride);

	int	nElement = 0;//UNUSED  THCudaTensor_nElement(state, gradoutput);

	error  = SeparableConvLayer_gpu_backward_kernel(
//			at::globalContext().getCurrentCUDAStream(), //works for 0.4.1
           at::cuda::getCurrentCUDAStream(), //works for 1.0.0
			nElement, //to let the nummous
			w,h,channel,batch,  input2.size(1),

			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,
			input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride,
			output_b_stride,output_c_stride,output_h_stride,output_w_stride,

			input1,
			input2,
			input3,
			gradoutput,
			gradinput1,
			gradinput2,
			gradinput3
			);
	  if (error) {AT_ERROR("CUDA call failed");}

	return error;
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("SeparableConvLayer_gpu_forward", &SeparableConvLayer_gpu_forward, "SeparableConv forward (CUDA)");
  m.def("SeparableConvLayer_gpu_backward", &SeparableConvLayer_gpu_backward, "SeparableConv backward (CUDA)");
}
