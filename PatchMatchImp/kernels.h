#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "DataType.h"

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

template <typename T>
void check(T result, char const* const func, const char* const file, int const line);



void Bgr2Gray(float *gray_image, uchar3 *color_image,
			  size_t width, size_t height,
			  dim3 blockpergrid, dim3 threadsperblock);

void Gray2Sobel(unsigned char* gray_image, short3* grad,
	size_t width, size_t height,
	dim3 blockpergrid, dim3 threadsperblock);

