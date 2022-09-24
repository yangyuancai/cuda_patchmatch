#include "Bgr2Gray.h"

__global__ void cuColorToGray(float *gray_image,uchar3 *color_image, size_t width, size_t height)
{
    size_t col = blockDim.x * blockIdx.x + threadIdx.x;
    size_t row = blockDim.y * blockIdx.y + threadIdx.y;
    size_t id = row * width + col;

    if (col >= 0 && row >= 0 &&
		col < width - 1 && row < height - 1)
    {
       gray_image[id] = 0.114f * (float)color_image[id].x + 0.587f * (float)color_image[id].y + 0.299f * (float)color_image[id].z;
    }

}


__global__ void cuSobelGrad( float3* grad,float* gray_image,
	size_t width, size_t height)
{
	size_t col = blockDim.x * blockIdx.x + threadIdx.x;
	size_t row = blockDim.y * blockIdx.y + threadIdx.y;
	size_t id = row * width + col;

	if (col > 0 && row > 0 && col < width - 1 && row < height - 1)
	{
		grad[id].x = gray_image[id - width + 1] + 2 * gray_image[id + 1] + gray_image[id + width + 1] -
			gray_image[id - width - 1] - 2 * gray_image[id - 1] - gray_image[id + width - 1];

		grad[id].y = gray_image[id - width - 1] + 2 * gray_image[id - width] + gray_image[id - width + 1] -
			gray_image[id + width - 1] - 2 * gray_image[id + width] - gray_image[id + width + 1];
		grad[id].z = sqrt(float(grad[id].x * grad[id].x + grad[id].y * grad[id].y));

	}
}

void Bgr2Gray(float *gray_image, uchar3 *color_image,
	size_t width, size_t height,
	dim3 blockpergrid, dim3 threadsperblock)
{
	cuColorToGray << <blockpergrid, threadsperblock >> > (gray_image, color_image, width, height);
}

void Gray2Sobel(float3* grad_image,float* gray_image, 
	size_t width, size_t height,
	dim3 blockpergrid, dim3 threadsperblock)
{
	cuSobelGrad << < blockpergrid, threadsperblock >> > (grad_image,gray_image, width, height);

}

template <typename T>
void check(T result, char const* const func, const char* const file, int const line) {
	if (result) {
		fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
			static_cast<unsigned int>(result), cudaGetErrorName(result), func);
		exit(EXIT_FAILURE);
	}
}





