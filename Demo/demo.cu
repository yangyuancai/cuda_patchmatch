#include <stdio.h>
#include "headfile.h"
#include "kernels.h"


#define _patch_size 16;
#define _patch_length 1089;//(2*16+1)*(2*16+1)
#define _image_height 375;
#define _image_width 450;

__constant__ AlogrithConfigFloat dev_algorith_const;
__constant__ int dev_aggr_const[1024];
__constant__ int3 dev_aggr_const3[1024];
__constant__ int dev_patch_size = _patch_size;
__constant__ int kWidthDev = 450;
__constant__ int kHeightDev = 375;
__constant__ unsigned int kWidthAddDev = 451;
__constant__ unsigned int kHeightAddDev = 376;
__constant__ unsigned int kWidthSubDev = 449;
__constant__ unsigned int kHeightSubDev = 374;

__constant__ float kSoblexDev[9] = { -1,0,1,-2,0,3,-1,0,1 };
__constant__ float kSobleyDev[9] = { 1,2,1,0,0,0,-1,-2,-1 };



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
	cudaError_t runtime_status = cudaSuccess;
	int patch_size = _patch_size;
	int patch_length = _patch_length;
	int id = 0;
	int *host_aggr_offset = NULL;
	cudaMallocHost((void**)&host_aggr_offset, sizeof(int) * patch_length);
	int3 *host_aggr3_offset = NULL;
	cudaMallocHost((void**)&host_aggr3_offset, sizeof(int3) * patch_length);

	for (int row = -patch_size; row < patch_size; ++row)
	{
		for (int col = -patch_size; col < patch_size; ++col)
		{
			host_aggr_offset[id] = row * _image_width + col;
			host_aggr3_offset[id] = make_int3(col, row, host_aggr_offset[id]);
			++id;
		}
	}
	AlogrithConfigFloat host_algorith;
	host_algorith.alpha = 0.9;
	host_algorith.cost_punish = 120;
	host_algorith.disparity_range = 64;
	host_algorith.gamma = 0.1;
	host_algorith.is_check_lr = 1;
	host_algorith.is_fill_holes = 1;
	host_algorith.is_fource_fpw = 1;
	host_algorith.is_integer_disp = 1;
	host_algorith.lrcheck_thres = 10;
	host_algorith.max_disparity = 64;
	host_algorith.min_disparity = 0;
	host_algorith.tau_color = 10;
	host_algorith.tau_grad = 2;



	runtime_status = InitConstParams(&host_algorith, host_aggr3_offset, host_aggr_offset);
	checkCudaErrors(runtime_status);

	cudaDeviceProp devprop;
	runtime_status = cudaGetDeviceProperties(&devprop, 0);
	checkCudaErrors(runtime_status);
	size_t maxblockperSM = devprop.maxBlocksPerMultiProcessor;
	size_t maxthreadperBlock = devprop.maxThreadsPerBlock;
	size_t maxthreadperSM = devprop.maxThreadsPerMultiProcessor;
	size_t maxregisterperSM = devprop.regsPerMultiprocessor;
	size_t maxregisterperblock = devprop.regsPerBlock;
	size_t num_SM = devprop.multiProcessorCount;

	curandStatus_t cuda_rand_status = CURAND_STATUS_SUCCESS;
	ImageInfo bgrInfo(450u, 375u, Uchar3Img);
	ImageInfo grayInfo(450u, 375u, UcharImg);

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
	float *hostcostaggrleft = NULL;
	float *hostcostaggrright = NULL;


	uchar3* devbgrleft = NULL;
	uchar3* devbgrright = NULL;
	float* devgrayleft = NULL;
	float* devgrayright = NULL;
	float3* devgradleft = NULL;
	float3* devgradright = NULL;


	float *devdispleft = NULL;
	float *devdispright = NULL;
	float3 *devdispplaneleft = NULL;
	float3 *devdispplaneright = NULL;
	float *devcostaggrleft = NULL;
	float *devcostaggrright = NULL;

	//host
	runtime_status = cudaMallocHost((void**)&hostbgrleft, sizeof(uchar) * grayInfo.imgsize * 3);
	checkCudaErrors(runtime_status);
	runtime_status = cudaMallocHost((void**)&hostbgrright, sizeof(uchar) * grayInfo.imgsize * 3);
	checkCudaErrors(runtime_status);
	runtime_status = cudaMallocHost((void**)&hostgrayleft, sizeof(float)*grayInfo.imgsize);
	checkCudaErrors(runtime_status);
	runtime_status = cudaMallocHost((void**)&hostgrayright, sizeof(float) * grayInfo.imgsize);
	checkCudaErrors(runtime_status);
	runtime_status = cudaMallocHost((void**)&hostgradleft, sizeof(float3) * grayInfo.imgsize);
	checkCudaErrors(runtime_status);
	runtime_status = cudaMallocHost((void**)&hostgradright, sizeof(float3) * grayInfo.imgsize);
	checkCudaErrors(runtime_status);
	runtime_status = cudaMallocHost((void**)&hostdispleft, sizeof(float) * grayInfo.imgsize);
	checkCudaErrors(runtime_status);
	runtime_status = cudaMallocHost((void**)&hostdispright, sizeof(float) * grayInfo.imgsize);
	checkCudaErrors(runtime_status);
	runtime_status = cudaMallocHost((void**)&hostdispplaneleft, sizeof(float3) * grayInfo.imgsize);
	checkCudaErrors(runtime_status);
	runtime_status = cudaMallocHost((void**)&hostdispplaneright, sizeof(float3) * grayInfo.imgsize);
	checkCudaErrors(runtime_status);

	runtime_status = cudaMallocHost((void**)&hostcostaggrleft, sizeof(float) * grayInfo.imgsize);
	checkCudaErrors(runtime_status);
	runtime_status = cudaMallocHost((void**)&hostcostaggrright, sizeof(float) * grayInfo.imgsize);
	checkCudaErrors(runtime_status);

	//gray image and grad image
	runtime_status = cudaMalloc((void**)&devgrayleft, sizeof(float) * grayInfo.imgsize);
	checkCudaErrors(runtime_status);
	runtime_status = cudaMalloc((void**)&devgrayright, sizeof(float) * grayInfo.imgsize);
	checkCudaErrors(runtime_status);
	runtime_status = cudaMalloc((void**)&devbgrleft, sizeof(float) * grayInfo.imgsize);
	checkCudaErrors(runtime_status);
	runtime_status = cudaMalloc((void**)&devbgrright, sizeof(float) * grayInfo.imgsize);
	checkCudaErrors(runtime_status);
	runtime_status = cudaMalloc((void**)&devgradleft, sizeof(float3) * grayInfo.imgsize);
	checkCudaErrors(runtime_status);
	runtime_status = cudaMalloc((void**)&devgradright, sizeof(float3) * grayInfo.imgsize);
	checkCudaErrors(runtime_status);

	//disparity image and disparity plane image and cost aggr image
	runtime_status = cudaMalloc((void**)&devdispleft, sizeof(float) * grayInfo.imgsize);
	checkCudaErrors(runtime_status);
	runtime_status = cudaMalloc((void**)&devdispright, sizeof(float) * grayInfo.imgsize);
	checkCudaErrors(runtime_status);
	runtime_status = cudaMalloc((void**)&devdispplaneleft, sizeof(float3) * grayInfo.imgsize);
	checkCudaErrors(runtime_status);
	runtime_status = cudaMalloc((void**)&devdispplaneright, sizeof(float3) * grayInfo.imgsize);
	checkCudaErrors(runtime_status);
	runtime_status = cudaMalloc((void**)&devcostaggrleft, sizeof(float) * grayInfo.imgsize);
	checkCudaErrors(runtime_status);
	runtime_status = cudaMalloc((void**)&devcostaggrright, sizeof(float) * grayInfo.imgsize);
	checkCudaErrors(runtime_status);

	//
	runtime_status = cudaMemset(devdispleft, 0, sizeof(float) * grayInfo.imgsize);
	checkCudaErrors(runtime_status);
	runtime_status = cudaMemset(devdispright, 0, sizeof(float) * grayInfo.imgsize);
	checkCudaErrors(runtime_status);
	runtime_status = cudaMemset(devdispplaneleft, 0, sizeof(float3) * grayInfo.imgsize);
	checkCudaErrors(runtime_status);
	runtime_status = cudaMemset(devdispplaneright, 0, sizeof(float3) * grayInfo.imgsize);
	checkCudaErrors(runtime_status);
	runtime_status = cudaMemset(devcostaggrleft, 0, sizeof(float) * grayInfo.imgsize);
	checkCudaErrors(runtime_status);
	runtime_status = cudaMemset(devcostaggrright, 0, sizeof(float) * grayInfo.imgsize);
	checkCudaErrors(runtime_status);

	std::cout << sizeof(float3) << std::endl;


	// init device disparity plane
	curandGenerator_t generator;
	curandStatus_t rand_status = CURAND_STATUS_SUCCESS;
	rand_status = curandCreateGenerator(&generator, CURAND_RNG_QUASI_SCRAMBLED_SOBOL32);
	if (CURAND_STATUS_SUCCESS != rand_status)
	{
		return rand_status;
	}

	rand_status = curandGenerateUniform(generator, devdispleft, grayInfo.imgsize);
	checkCudaErrors(cudaError_t(cuda_rand_status));
	rand_status = curandGenerateUniform(generator, devdispright, grayInfo.imgsize);
	checkCudaErrors(cudaError_t(cuda_rand_status));	
	rand_status = curandGenerateUniform(generator, (float*)(void*)devdispplaneleft, grayInfo.imgsize*3);
	checkCudaErrors(cudaError_t(cuda_rand_status));
	rand_status = curandGenerateUniform(generator, (float*)(void*)devdispplaneright,grayInfo.imgsize*3);
	checkCudaErrors(cudaError_t(cuda_rand_status));

	#ifdef DEBUG

	runtime_status = cudaMemcpy(hostdispleft, devdispleft, sizeof(float) * grayInfo.imgsize, cudaMemcpyDeviceToHost);
	checkCudaErrors(runtime_status);
	runtime_status = cudaMemcpy(hostdispright, devdispright, sizeof(float) * grayInfo.imgsize, cudaMemcpyDeviceToHost);
	checkCudaErrors(runtime_status);
	runtime_status = cudaMemcpy(hostdispplaneleft, devdispplaneleft, sizeof(float3) * grayInfo.imgsize, cudaMemcpyDeviceToHost);
	checkCudaErrors(runtime_status);
	runtime_status = cudaMemcpy(hostdispplaneright, devdispplaneright, sizeof(float3) * grayInfo.imgsize, cudaMemcpyDeviceToHost);
	checkCudaErrors(runtime_status);
	cudaDeviceSynchronize();

	cv::Mat hostdispleft_mat = cv::Mat(grayInfo.height, grayInfo.width, CV_32FC1, hostdispleft);
	cv::Mat hostdispright_mat = cv::Mat(grayInfo.height, grayInfo.width, CV_32FC1, hostdispright);
	cv::Mat hostdispplaneleft_mat = cv::Mat(grayInfo.height, grayInfo.width, CV_32FC3, hostdispplaneleft);
	cv::Mat hostdispplaneright_mat = cv::Mat(grayInfo.height, grayInfo.width, CV_32FC3, hostdispplaneright);
	#endif // DEBUG

	dim3 threadsperblock(32u, 16u);
	dim3 blockpergrid(15u, 24u);
	runtime_status = RandomInitialDisparityAndItsPlane(devdispleft, devdispplaneleft,
										host_algorith.min_disparity, host_algorith.disparity_range,
										grayInfo.width, grayInfo.height, blockpergrid, threadsperblock);
	checkCudaErrors(runtime_status);
	runtime_status = RandomInitialDisparityAndItsPlane(devdispright, devdispplaneright,
										host_algorith.min_disparity, host_algorith.disparity_range,
										grayInfo.width, grayInfo.height, blockpergrid, threadsperblock);
	checkCudaErrors(runtime_status);

	cudaDeviceSynchronize();
#ifdef DEBUG
	runtime_status = cudaMemcpy(hostdispleft, devdispleft, sizeof(float) * grayInfo.imgsize, cudaMemcpyDeviceToHost);
	checkCudaErrors(runtime_status);
	runtime_status = cudaMemcpy(hostdispright, devdispright, sizeof(float) * grayInfo.imgsize, cudaMemcpyDeviceToHost);
	checkCudaErrors(runtime_status);
	runtime_status = cudaMemcpy(hostdispplaneleft, devdispplaneleft, sizeof(float3) * grayInfo.imgsize, cudaMemcpyDeviceToHost);
	checkCudaErrors(runtime_status);
	runtime_status = cudaMemcpy(hostdispplaneright, devdispplaneright, sizeof(float3) * grayInfo.imgsize, cudaMemcpyDeviceToHost);
	checkCudaErrors(runtime_status);
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
	runtime_status = cudaMemcpy(devbgrleft, hostbgrleft, sizeof(char3) * grayInfo.imgsize, cudaMemcpyHostToDevice);
	checkCudaErrors(runtime_status);
	runtime_status = cudaMemcpy(devbgrright, hostbgrright, sizeof(char3) * grayInfo.imgsize, cudaMemcpyHostToDevice);
	checkCudaErrors(runtime_status);
	cudaDeviceSynchronize();
	//convert bgr image to gray and grad image
	runtime_status = Bgr2Gray(devgrayleft, devbgrleft,grayInfo.width, grayInfo.height, blockpergrid, threadsperblock);
	checkCudaErrors(runtime_status);
	runtime_status = Bgr2Gray(devgrayright, devbgrright,grayInfo.width, grayInfo.height, blockpergrid, threadsperblock);
	checkCudaErrors(runtime_status);
	runtime_status = Gray2Sobel(devgradleft,devgrayleft,  grayInfo.width, grayInfo.height, blockpergrid, threadsperblock);
	checkCudaErrors(runtime_status);
	runtime_status = Gray2Sobel(devgradright,devgrayright, grayInfo.width, grayInfo.height, blockpergrid, threadsperblock);
	checkCudaErrors(runtime_status);
	cudaDeviceSynchronize();
	//load image to host
#ifdef DEBUG
	runtime_status = cudaMemcpy(hostgrayleft, devgrayleft, sizeof(float) * grayInfo.imgsize, cudaMemcpyDeviceToHost);
	checkCudaErrors(runtime_status);
	runtime_status = cudaMemcpy(hostgrayright, devgrayright, sizeof(float) * grayInfo.imgsize, cudaMemcpyDeviceToHost);
	checkCudaErrors(runtime_status);
	runtime_status = cudaMemcpy(hostgradleft, devgradleft, sizeof(float3) * grayInfo.imgsize, cudaMemcpyDeviceToHost);
	checkCudaErrors(runtime_status);
	runtime_status = cudaMemcpy(hostgradright, devgradright, sizeof(float3) * grayInfo.imgsize, cudaMemcpyDeviceToHost);
	checkCudaErrors(runtime_status);
	cudaDeviceSynchronize();

	cv::Mat gray_left_mat = cv::Mat(grayInfo.height, grayInfo.width, CV_32FC1, hostgrayleft);
	cv::Mat grad_left_mat = cv::Mat(grayInfo.height, grayInfo.width, CV_32FC3, hostgradleft);
	cv::Mat gray_right_mat = cv::Mat(grayInfo.height, grayInfo.width, CV_32FC1, hostgrayright);
	cv::Mat grad_right_mat = cv::Mat(grayInfo.height, grayInfo.width, CV_32FC3, hostgradright);
#endif //DEBUG


	runtime_status = CaculateCostAggregationInitConst(devcostaggrleft, devdispplaneleft,
								devbgrleft, devbgrright,
								devgradleft, devgradright,
								blockpergrid, threadsperblock);

	checkCudaErrors(runtime_status);

	runtime_status = CaculateCostAggregationInitConst(devcostaggrright, devdispplaneright,
								devbgrright, devbgrleft,
								devgradright, devgradleft,
								blockpergrid, threadsperblock);

	checkCudaErrors(runtime_status);
	cudaDeviceSynchronize();

#ifdef DEBUG

	runtime_status = cudaMemcpy(hostcostaggrleft, devcostaggrleft, sizeof(float) * grayInfo.imgsize, cudaMemcpyDeviceToHost);
	checkCudaErrors(runtime_status);
	runtime_status = cudaMemcpy(hostcostaggrright, devcostaggrright, sizeof(float) * grayInfo.imgsize, cudaMemcpyDeviceToHost);
	checkCudaErrors(runtime_status);
	cudaDeviceSynchronize();

	cv::Mat hostcostaggrleft_mat = cv::Mat(grayInfo.height, grayInfo.width, CV_32FC1, hostcostaggrleft);
	cv::Mat hostcostaggrright_mat = cv::Mat(grayInfo.height, grayInfo.width, CV_32FC1, hostcostaggrright);

#endif // DEBUG


	runtime_status = cudaFreeHost(hostbgrleft);
	checkCudaErrors(runtime_status);
	runtime_status = cudaFreeHost(hostbgrright);
	checkCudaErrors(runtime_status);
	runtime_status = cudaFreeHost(hostgrayleft);
	checkCudaErrors(runtime_status);
	runtime_status = cudaFreeHost(hostgrayright);
	checkCudaErrors(runtime_status);
	runtime_status = cudaFreeHost(hostgradleft);
	checkCudaErrors(runtime_status);
	runtime_status = cudaFreeHost(hostgradright);
	checkCudaErrors(runtime_status);
	runtime_status = cudaFreeHost(hostdispleft);
	checkCudaErrors(runtime_status);
	runtime_status = cudaFreeHost(hostdispright);
	checkCudaErrors(runtime_status);
	runtime_status = cudaFreeHost(hostdispplaneleft);
	checkCudaErrors(runtime_status);
	runtime_status = cudaFreeHost(hostdispplaneright);
	checkCudaErrors(runtime_status);
	runtime_status = cudaFreeHost(hostcostaggrleft);
	checkCudaErrors(runtime_status);
	runtime_status = cudaFreeHost(hostcostaggrright);
	checkCudaErrors(runtime_status);
	runtime_status = cudaFreeHost(host_aggr_offset);
	checkCudaErrors(runtime_status);
	runtime_status = cudaFreeHost(host_aggr3_offset);
	checkCudaErrors(runtime_status);


	runtime_status = cudaFree(devbgrleft);
	checkCudaErrors(runtime_status);
	runtime_status = cudaFree(devbgrright);
	checkCudaErrors(runtime_status);
	runtime_status = cudaFree(devgrayleft);
	checkCudaErrors(runtime_status);
	runtime_status = cudaFree(devgrayright);
	checkCudaErrors(runtime_status);
	runtime_status = cudaFree(devgradleft);
	checkCudaErrors(runtime_status);
	runtime_status = cudaFree(devgradright);
	checkCudaErrors(runtime_status);

	runtime_status = cudaFree(devdispleft);
	checkCudaErrors(runtime_status);
	runtime_status = cudaFree(devdispright);
	checkCudaErrors(runtime_status);
	runtime_status = cudaFree(devdispplaneleft);
	checkCudaErrors(runtime_status);
	runtime_status = cudaFree(devdispplaneright);
	checkCudaErrors(runtime_status);


	runtime_status = cudaFree(devcostaggrleft);
	checkCudaErrors(runtime_status);
	runtime_status = cudaFree(devcostaggrright);
	checkCudaErrors(runtime_status);


	return 0;
}
