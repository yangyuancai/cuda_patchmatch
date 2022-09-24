#ifndef _KERNELS_H

	#include "curand.h"
	#include "cuda_runtime.h"
	#include "device_launch_parameters.h"
	#include <stdio.h>
	#include "DataType.h"

	curandStatus_t RandomInitialFloat(float* arr, size_t N);

	curandStatus_t RandomInitialFloat3(float3* arr, size_t N);

	void Bgr2Gray(float *gray_image, uchar3 *color_image,
		size_t width, size_t height,
		dim3 blockpergrid, dim3 threadsperblock);

	void Gray2Sobel(float3* grad,float* gray_image,
					size_t width, size_t height,
					dim3 blockpergrid, dim3 threadsperblock);




#endif // !_KERNELS_H

