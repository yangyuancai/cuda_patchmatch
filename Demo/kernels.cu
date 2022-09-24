#include "kernels.h"
#include "opencv2/opencv.hpp"
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
	cuSobelGrad << < blockpergrid, threadsperblock >> > (grad_image,gray_image,  width, height);

}

curandStatus_t RandomInitialFloat(float* arr, size_t N)
{
	curandGenerator_t generator;
	curandStatus_t rand_status;
	cudaError_t runtime_status;
	rand_status = curandCreateGenerator(&generator, CURAND_RNG_QUASI_SCRAMBLED_SOBOL32);
	if (CURAND_STATUS_SUCCESS != rand_status)
		return rand_status;

	rand_status = curandGenerateNormal(generator, arr,N, 0.0, 1.0);
	if (CURAND_STATUS_SUCCESS != rand_status)
		return rand_status;

	//float *hostdispleft = NULL;
	//runtime_status = cudaMallocHost((void**)&hostdispleft,N);
	//runtime_status = cudaMemcpy(hostdispleft, arr, N, cudaMemcpyDeviceToHost);
	//cv::Mat hostdispleft_mat = cv::Mat(375, 450, CV_32FC1, hostdispleft);
	//cudaFree(hostdispleft);

	return CURAND_STATUS_SUCCESS;
}

curandStatus_t RandomInitialFloat3(float3* arr, size_t N)
{
	curandGenerator_t generator;
	curandStatus_t rand_status;
	cudaError_t runtime_status;

	rand_status = curandCreateGenerator(&generator, CURAND_RNG_QUASI_SCRAMBLED_SOBOL32);
	if (CURAND_STATUS_SUCCESS != rand_status)
		return rand_status;
	rand_status = curandGenerateNormal(generator, (float*)arr, N, 0.0, 1.0);
	if (CURAND_STATUS_SUCCESS != rand_status)
		return rand_status;

	//float *hostdispleft = NULL;
	//runtime_status = cudaMallocHost((void**)&hostdispleft, N);
	//runtime_status = cudaMemcpy(hostdispleft, arr, N, cudaMemcpyDeviceToHost);
	//cv::Mat hostdispleft_mat = cv::Mat(375, 450, CV_32FC3, hostdispleft);
	//cudaFree(hostdispleft);

	return CURAND_STATUS_SUCCESS;
}





