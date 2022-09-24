#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include <stdio.h>
#include "DataType.h"

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

template <typename T>
void check(T result, char const* const func, const char* const file, int const line);



__global__ void CuInitDisparity(ImageInfo* image_info, AlogrithConfig* algori_config,
	float* device_disparity_left, float* device_disparity_right,
	float* rand_disparity_left_data, float* rand_disparity_right_data);


__global__ void CuPlaneToDisparity(float *disp, float3 *plane, size_t width, size_t height);

__global__ void CuCaculateCost(float *cost, float* disparity,
	uchar3*left_color_image, uchar3* right_color_image,
	float3* left_grad_image, float3* right_grad_image,
	float alpha, float tau_color, float tau_grad, size_t width, size_t height);



__global__ void CuInitDisparity(ImageInfo* image_info, AlogrithConfig* algori_config,
								float* device_disparity_left, float* device_disparity_right,
								float* rand_disparity_left_data, float* rand_disparity_right_data);

__global__ void CuCaculateCostAggregation(float3* disp_plane, float min_disp, float max_disp, float cost_punish,
	size_t patch_size, size_t width, size_t height);

__global__ void  SpatialPropagation();

__global__ void cuColorToGray(float *gray_image, uchar3 *color_image, size_t width, size_t height);

__global__ void cuSobelGrad(unsigned char* gray_image, short3* grad, size_t width, size_t height);