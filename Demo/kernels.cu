#include "kernels.h"
#include "opencv2/opencv.hpp"


__global__ void cuRandomInitialDisparityAndItsPlane(float* disparity, float3* disparityplane,
								size_t min_disparity,size_t disparity_range,
								size_t width, size_t height)
{
	size_t col = blockDim.x * blockIdx.x + threadIdx.x;
	size_t row = blockDim.y * blockIdx.y + threadIdx.y;
	size_t id = row * width + col;
	if (col >= 0 && row >= 0 &&
		col < width && row < height)
	{
		disparity[id] = min_disparity + disparity[id] * disparity_range;
		float rand_x = disparityplane[id].x - 0.5f;
		float rand_y = disparityplane[id].y - 0.5f;
		float rand_z = disparityplane[id].z - 0.5f;
		rand_z = (abs(rand_z) < 1e-3) ? 0.001f : rand_z;
		disparityplane[id].x = -rand_x / rand_z;//a
		disparityplane[id].y = -rand_y / rand_z;//b
		//d = a*col+b*row+c;		c = d-a*col-b*row
		disparityplane[id].z = disparity[id] - disparityplane[id].x*float(col) - disparityplane[id].y * float(row);
	}
}

__global__ void cuColorToGray(float *gray_image,uchar3 *color_image, size_t width, size_t height)
{
    size_t col = blockDim.x * blockIdx.x + threadIdx.x;
    size_t row = blockDim.y * blockIdx.y + threadIdx.y;
    size_t id = row * width + col;

    if (col >= 0 && row >= 0 &&
		col < width && row < height)
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
	cuSobelGrad << < blockpergrid, threadsperblock >> > (grad_image, gray_image, width, height);

}

void RandomInitialDisparityAndItsPlane(float* disparity, float3* disparityplane,size_t min_disparity, size_t disparity_range,
						size_t width, size_t height, dim3 blockpergrid, dim3 threadsperblock)
{
	cuRandomInitialDisparityAndItsPlane << <blockpergrid, threadsperblock >> > (disparity,disparityplane,
															min_disparity,disparity_range,
															width,height);
}





