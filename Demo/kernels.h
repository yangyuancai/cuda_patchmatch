#ifndef _KERNELS_H

	#include "curand.h"
	#include "cuda_runtime.h"
	#include "device_launch_parameters.h"
	#include <stdio.h>
	#include "DataType.h"

void RandomInitialDisparityAndItsPlane(float* disparity, float3* disparityplane,
										size_t min_disparity, size_t disparity_range,
										size_t width, size_t height, 
										dim3 blockpergrid, dim3 threadsperblock);



	void Bgr2Gray(float *gray_image, uchar3 *color_image,
		size_t width, size_t height,
		dim3 blockpergrid, dim3 threadsperblock);

	void Gray2Sobel(float3* grad,float* gray_image,
					size_t width, size_t height,
					dim3 blockpergrid, dim3 threadsperblock);




#endif // !_KERNELS_H

