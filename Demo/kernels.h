#ifndef _KERNELS_H
#define _KERNELS_H

#include "curand.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "cuda.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "DataType.h"
#include "opencv2/opencv.hpp"

//cudaError_t InitConstParams(AlogrithConfigFloat *host_algorith_const, int * host_aggr_const);

cudaError_t InitConstParams(AlogrithConfigFloat *host_algorith_const, int3 * host_aggr_const3, int * host_aggr_const);

cudaError_t RandomInitialDisparityAndItsPlane(float* disparity, float3* disparityplane,
										size_t min_disparity, size_t disparity_range,
										size_t width, size_t height, 
										dim3 blockpergrid, dim3 threadsperblock);

cudaError_t Bgr2Gray(float *gray_image, uchar3 *color_image,
	size_t width, size_t height,
	dim3 blockpergrid, dim3 threadsperblock);

cudaError_t Gray2Sobel(float3* grad,float* gray_image,
				size_t width, size_t height,
				dim3 blockpergrid, dim3 threadsperblock);


cudaError_t CaculateCostAggregationInit(float *costaggr, float3* disptyplane,
	uchar3 *left_color_image, uchar3 *right_color_image,
	float3 *left_grad_image, float3 *right_grad_image,
	float alpha, float tau_color, float tau_grad,
	float min_disp, float max_disp, float cost_punish, float gama,
	size_t patch_size, size_t width, size_t height,
	dim3 blockpergrid, dim3 threadsperblock);

cudaError_t CaculateCostAggregationInitConst(float *costaggr, float3* disptyplane,
	uchar3 *left_color_image, uchar3 *right_color_image,
	float3 *left_grad_image, float3 *right_grad_image,
	dim3 blockpergrid, dim3 threadsperblock);

#endif // !_KERNELS_H

