#include <stdio.h>
#include "headfile.h"
#include "kernels.h"

#define DEBUG 1

#ifdef DEBUG
	#include "opencv2/opencv.hpp"
#endif // DEBUG


#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T result, char const* const func, const char* const file, int const line) {
	if (result) {
		fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
			static_cast<unsigned int>(result), cudaGetErrorName(result), func);
		exit(EXIT_FAILURE);
	}
}

int main()
{
	cudaError_t cuda_runtime_status = cudaSuccess;
	curandStatus_t cuda_rand_status = CURAND_STATUS_SUCCESS;
	ImageInfo bgrInfo(450u, 375u, Uchar3Img);
	ImageInfo grayInfo(450u, 375u, UcharImg);

	AlogrithConfig alogConfig;

	curandGenerator_t generator;
	curandStatus_t rand_status;
	cudaError_t runtime_status;
	rand_status = curandCreateGenerator(&generator, CURAND_RNG_QUASI_SCRAMBLED_SOBOL32);
	if (CURAND_STATUS_SUCCESS != rand_status)
		return rand_status;

	// memory allocate
	uchar *hostbgrleft = NULL;
	uchar *hostbgrright = NULL;

	float* hostgrayleft = NULL;
	float* hostgrayright = NULL;
	float3* hostgradleft = NULL;
	float3* hostgradright = NULL;

	float *hostdispleft = NULL;
	float *hostdispright = NULL;
	float3 *hostdispplaneleft = NULL;
	float3 *hostdispplaneright = NULL;

	uchar3* devbgrleft = NULL;
	uchar3* devbgrright = NULL;
	float* devgrayleft = NULL;
	float* devgrayright = NULL;
	float3* devgradleft = NULL;
	float3* devgradright = NULL;

	float *devcostleft = NULL;
	float *devcostright = NULL;
	float *devdispleft = NULL;
	float *devdispright = NULL;
	float3 *devdispplaneleft = NULL;
	float3 *devdispplaneright = NULL;

	cuda_runtime_status = cudaMallocHost((void**)&hostbgrleft, sizeof(uchar) * grayInfo.imgsize * 3);
	checkCudaErrors(cuda_runtime_status);
	cuda_runtime_status = cudaMallocHost((void**)&hostbgrright, sizeof(uchar) * grayInfo.imgsize * 3);
	checkCudaErrors(cuda_runtime_status);


	cuda_runtime_status = cudaMallocHost((void**)&hostgrayleft, sizeof(float)*grayInfo.imgsize);
	checkCudaErrors(cuda_runtime_status);
	cuda_runtime_status = cudaMallocHost((void**)&hostgrayright, sizeof(float) * grayInfo.imgsize);
	checkCudaErrors(cuda_runtime_status);
	cuda_runtime_status = cudaMallocHost((void**)&hostgradleft, sizeof(float3) * grayInfo.imgsize);
	checkCudaErrors(cuda_runtime_status);
	cuda_runtime_status = cudaMallocHost((void**)&hostgradright, sizeof(float3) * grayInfo.imgsize);
	checkCudaErrors(cuda_runtime_status);

	cuda_runtime_status = cudaMallocHost((void**)&hostdispleft, sizeof(float) * grayInfo.imgsize);
	checkCudaErrors(cuda_runtime_status);
	cuda_runtime_status = cudaMallocHost((void**)&hostdispright, sizeof(float) * grayInfo.imgsize);
	checkCudaErrors(cuda_runtime_status);
	cuda_runtime_status = cudaMallocHost((void**)&hostdispplaneleft, sizeof(float3) * grayInfo.imgsize);
	checkCudaErrors(cuda_runtime_status);
	cuda_runtime_status = cudaMallocHost((void**)&hostdispplaneright, sizeof(float3) * grayInfo.imgsize);
	checkCudaErrors(cuda_runtime_status);


	cuda_runtime_status = cudaMalloc((void**)&devgrayleft, sizeof(float) * grayInfo.imgsize);
	checkCudaErrors(cuda_runtime_status);
	cuda_runtime_status = cudaMalloc((void**)&devgrayright, sizeof(float) * grayInfo.imgsize);
	checkCudaErrors(cuda_runtime_status);
	cuda_runtime_status = cudaMalloc((void**)&devbgrleft, sizeof(float) * grayInfo.imgsize);
	checkCudaErrors(cuda_runtime_status);
	cuda_runtime_status = cudaMalloc((void**)&devbgrright, sizeof(float) * grayInfo.imgsize);
	checkCudaErrors(cuda_runtime_status);
	cuda_runtime_status = cudaMalloc((void**)&devgradleft, sizeof(float3) * grayInfo.imgsize);
	checkCudaErrors(cuda_runtime_status);
	cuda_runtime_status = cudaMalloc((void**)&devgradright, sizeof(float3) * grayInfo.imgsize);
	checkCudaErrors(cuda_runtime_status);
	cuda_runtime_status = cudaMalloc((void**)&devcostleft, sizeof(float) * grayInfo.imgsize);
	checkCudaErrors(cuda_runtime_status);
	cuda_runtime_status = cudaMalloc((void**)&devcostright, sizeof(float) * grayInfo.imgsize);
	checkCudaErrors(cuda_runtime_status);
	cuda_runtime_status = cudaMalloc((void**)&devdispleft, sizeof(float) * grayInfo.imgsize);
	checkCudaErrors(cuda_runtime_status);
	cuda_runtime_status = cudaMalloc((void**)&devdispright, sizeof(float) * grayInfo.imgsize);
	checkCudaErrors(cuda_runtime_status);
	cuda_runtime_status = cudaMalloc((void**)&devdispplaneleft, sizeof(float3) * grayInfo.imgsize);
	checkCudaErrors(cuda_runtime_status);
	cuda_runtime_status = cudaMalloc((void**)&devdispplaneright, sizeof(float3) * grayInfo.imgsize);
	checkCudaErrors(cuda_runtime_status);

	cuda_runtime_status = cudaMemset(devdispleft, 0, sizeof(float) * grayInfo.imgsize);
	checkCudaErrors(cuda_runtime_status);
	cuda_runtime_status = cudaMemset(devdispright, 0, sizeof(float) * grayInfo.imgsize);
	checkCudaErrors(cuda_runtime_status);
	cuda_runtime_status = cudaMemset(devdispplaneleft, 0, sizeof(float3) * grayInfo.imgsize);
	checkCudaErrors(cuda_runtime_status);
	cuda_runtime_status = cudaMemset(devdispplaneright, 0, sizeof(float3) * grayInfo.imgsize);
	checkCudaErrors(cuda_runtime_status);
	std::cout << sizeof(float3) << std::endl;


	// init device disparity plane
	rand_status = curandGenerateUniform(generator, devdispleft, grayInfo.imgsize);
	checkCudaErrors(cudaError_t(cuda_rand_status));
	rand_status = curandGenerateUniform(generator, devdispright, grayInfo.imgsize);
	checkCudaErrors(cudaError_t(cuda_rand_status));	
	rand_status = curandGenerateUniform(generator, (float*)(void*)devdispplaneleft, grayInfo.imgsize*3);
	checkCudaErrors(cudaError_t(cuda_rand_status));
	rand_status = curandGenerateUniform(generator, (float*)(void*)devdispplaneright,grayInfo.imgsize*3);
	checkCudaErrors(cudaError_t(cuda_rand_status));

	#ifdef DEBUG

	cuda_runtime_status = cudaMemcpy(hostdispleft, devdispleft, sizeof(float) * grayInfo.imgsize, cudaMemcpyDeviceToHost);
	checkCudaErrors(cuda_runtime_status);
	cuda_runtime_status = cudaMemcpy(hostdispright, devdispright, sizeof(float) * grayInfo.imgsize, cudaMemcpyDeviceToHost);
	checkCudaErrors(cuda_runtime_status);
	cuda_runtime_status = cudaMemcpy(hostdispplaneleft, devdispplaneleft, sizeof(float3) * grayInfo.imgsize, cudaMemcpyDeviceToHost);
	checkCudaErrors(cuda_runtime_status);
	cuda_runtime_status = cudaMemcpy(hostdispplaneright, devdispplaneright, sizeof(float3) * grayInfo.imgsize, cudaMemcpyDeviceToHost);
	checkCudaErrors(cuda_runtime_status);
	cudaDeviceSynchronize();

	cv::Mat hostdispleft_mat = cv::Mat(grayInfo.height, grayInfo.width, CV_32FC1, hostdispleft);
	cv::Mat hostdispright_mat = cv::Mat(grayInfo.height, grayInfo.width, CV_32FC1, hostdispright);
	cv::Mat hostdispplaneleft_mat = cv::Mat(grayInfo.height, grayInfo.width, CV_32FC3, hostdispplaneleft);
	cv::Mat hostdispplaneright_mat = cv::Mat(grayInfo.height, grayInfo.width, CV_32FC3, hostdispplaneright);
	#endif // DEBUG

	dim3 threadsperblock(32u, 32u);
	dim3 blockpergrid(15u, 12u);
	RandomInitialDisparityAndItsPlane(devdispleft, devdispplaneleft,
										alogConfig.min_disparity, alogConfig.disparity_range,
										grayInfo.width, grayInfo.height, blockpergrid, threadsperblock);
	RandomInitialDisparityAndItsPlane(devdispright, devdispplaneright,
		alogConfig.min_disparity, alogConfig.disparity_range,
		grayInfo.width, grayInfo.height, blockpergrid, threadsperblock);

	cudaDeviceSynchronize();
#ifdef DEBUG
	cuda_runtime_status = cudaMemcpy(hostdispleft, devdispleft, sizeof(float) * grayInfo.imgsize, cudaMemcpyDeviceToHost);
	checkCudaErrors(cuda_runtime_status);
	cuda_runtime_status = cudaMemcpy(hostdispright, devdispright, sizeof(float) * grayInfo.imgsize, cudaMemcpyDeviceToHost);
	checkCudaErrors(cuda_runtime_status);
	cuda_runtime_status = cudaMemcpy(hostdispplaneleft, devdispplaneleft, sizeof(float3) * grayInfo.imgsize, cudaMemcpyDeviceToHost);
	checkCudaErrors(cuda_runtime_status);
	cuda_runtime_status = cudaMemcpy(hostdispplaneright, devdispplaneright, sizeof(float3) * grayInfo.imgsize, cudaMemcpyDeviceToHost);
	checkCudaErrors(cuda_runtime_status);
	cudaDeviceSynchronize();
#endif // DEBUG


//------------------------------------major code--------------------------------------//
	//load image from host memory
	FILE *fptr = NULL;
	fptr = fopen("../../data/Cone/left.raw", "rb");
	fread(hostbgrleft, sizeof(uchar), bgrInfo.width * bgrInfo.height * 3, fptr);
	fclose(fptr);
	fptr = fopen("../../data/Cone/right.raw", "rb");
	fread(hostbgrright, sizeof(uchar), bgrInfo.width * bgrInfo.height * 3, fptr);
	fclose(fptr);

	#ifdef DEBUG
		cv::Mat host_bgr_left_mat = cv::Mat(grayInfo.height, grayInfo.width, CV_8UC3, hostbgrleft);
		cv::Mat host_bgr_right_mat = cv::Mat(grayInfo.height, grayInfo.width, CV_8UC3, hostbgrright);
	#endif //DEBUG


	//Copy image to device
	cuda_runtime_status = cudaMemcpy(devbgrleft, hostbgrleft, sizeof(char3) * grayInfo.imgsize, cudaMemcpyHostToDevice);
	checkCudaErrors(cuda_runtime_status);
	cuda_runtime_status = cudaMemcpy(devbgrright, hostbgrright, sizeof(char3) * grayInfo.imgsize, cudaMemcpyHostToDevice);
	checkCudaErrors(cuda_runtime_status);
	
	//convert bgr image to gray and grad image

	Bgr2Gray(devgrayleft, devbgrleft,grayInfo.width, grayInfo.height, blockpergrid, threadsperblock);
	Bgr2Gray(devgrayright, devbgrright,grayInfo.width, grayInfo.height, blockpergrid, threadsperblock);
	Gray2Sobel(devgradleft,devgrayleft,  grayInfo.width, grayInfo.height, blockpergrid, threadsperblock);
	Gray2Sobel(devgradright,devgrayright, grayInfo.width, grayInfo.height, blockpergrid, threadsperblock);
	

	//load image to host
#ifdef DEBUG
	cuda_runtime_status = cudaMemcpy(hostgrayleft, devgrayleft, sizeof(float) * grayInfo.imgsize, cudaMemcpyDeviceToHost);
	checkCudaErrors(cuda_runtime_status);
	cuda_runtime_status = cudaMemcpy(hostgrayright, devgrayright, sizeof(float) * grayInfo.imgsize, cudaMemcpyDeviceToHost);
	checkCudaErrors(cuda_runtime_status);
	cuda_runtime_status = cudaMemcpy(hostgradleft, devgradleft, sizeof(float3) * grayInfo.imgsize, cudaMemcpyDeviceToHost);
	checkCudaErrors(cuda_runtime_status);
	cuda_runtime_status = cudaMemcpy(hostgradright, devgradright, sizeof(float3) * grayInfo.imgsize, cudaMemcpyDeviceToHost);
	checkCudaErrors(cuda_runtime_status);

	cudaDeviceSynchronize();

	cv::Mat gray_left_mat = cv::Mat(grayInfo.height, grayInfo.width, CV_32FC1, hostgrayleft);
	cv::Mat grad_left_mat = cv::Mat(grayInfo.height, grayInfo.width, CV_32FC3, hostgradleft);
	cv::Mat gray_right_mat = cv::Mat(grayInfo.height, grayInfo.width, CV_32FC1, hostgrayright);
	cv::Mat grad_right_mat = cv::Mat(grayInfo.height, grayInfo.width, CV_32FC3, hostgradright);
#endif //DEBUG


	for (int k = 0; k < alogConfig.num_iters; ++k)
	{
		DoPropagation();
	}



	cudaFree(hostbgrleft);
	cudaFree(hostbgrright);
	cudaFree(hostgrayleft);
	cudaFree(hostgrayright);
	cudaFree(hostgradleft);
	cudaFree(hostgradright);
	cudaFree(hostdispleft);
	cudaFree(hostdispright);
	cudaFree(hostdispplaneleft);
	cudaFree(hostdispplaneright);
			  
	cudaFree(devbgrleft);
	cudaFree(devbgrright);
	cudaFree(devgrayleft);
	cudaFree(devgrayright);
	cudaFree(devgradleft);
	cudaFree(devgradright);
	cudaFree(devcostleft);
	cudaFree(devcostright);
	cudaFree(devdispleft);
	cudaFree(devdispright);
	cudaFree(devdispplaneleft);
	cudaFree(devdispplaneright);

	return 0;
}














//cv::Mat hostbgrleft = cv::imread("../../data/Cone/left.png", 1);
//cv::Mat hostbgrright = cv::imread("../../data/Cone/right.png", 1);
//FILE *fptr = NULL;
//fptr = fopen("../../data/Cone/left.raw", "wb");
//fwrite(hostbgrleft.data,sizeof(uchar),450*375*3, fptr);
//fclose(fptr);
//FILE *fptr2 = NULL;
//fptr2 = fopen("../../data/Cone/right.raw", "wb");
//fwrite(hostbgrright.data, sizeof(uchar), 450 * 375*3, fptr2);
//fclose(fptr);