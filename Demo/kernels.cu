#include "kernels.h"


extern __constant__ AlogrithConfigFloat dev_algorith_const;
extern __constant__ int dev_aggr_const[1024];
extern __constant__ int3 dev_aggr_const3[1024];
extern __constant__ int dev_patch_size;
extern __constant__ int kWidthDev;
extern __constant__ int kHeightDev;
extern __constant__ unsigned int kWidthAddDev;
extern __constant__ unsigned int kHeightAddDev;
extern __constant__ unsigned int kWidthSubDev;
extern __constant__ unsigned int kHeightSubDev;

extern __constant__ float kSoblexDev[9];
extern __constant__ float kSobleyDev[9];
#define DEBUG 1


cudaError_t InitConstParams(AlogrithConfigFloat *host_algorith_const,int3 * host_aggr_const3,int * host_aggr_const)
{
	cudaError_t err = cudaSuccess;
	std::cout << sizeof(AlogrithConfigFloat) << std::endl;
	std::cout << sizeof(float) << std::endl;
	err = cudaMemcpyToSymbol(dev_algorith_const, host_algorith_const, sizeof(AlogrithConfigFloat));

	if (err != cudaSuccess)
	{
		printf("CUDA Error: %s\n", cudaGetErrorString(err));
		printf("CUDA error at %s:%d code=%d(%s)\n", __FILE__, __LINE__,
			static_cast<unsigned int>(err), cudaGetErrorName(err));
		return err;
	}

#ifdef DEBUG

	AlogrithConfigFloat *dev_algorith_address = NULL;
	cudaGetSymbolAddress((void**)&dev_algorith_address, dev_algorith_const);
	AlogrithConfigFloat algorith_check;
	err = cudaMemcpyFromSymbol(&algorith_check, dev_algorith_const, sizeof(AlogrithConfigFloat));
	if (err != cudaSuccess)
	{
		printf("CUDA Error: %s\n", cudaGetErrorString(err));
		return err;
	}
#endif // DEBUG

	err = cudaMemcpyToSymbol(dev_aggr_const, host_aggr_const, 1024 * sizeof(int));
	if (err != cudaSuccess)
	{
		printf("CUDA Error: %s\n", cudaGetErrorString(err));
		return err;
	}

#ifdef DEBUG
	int *dev_aggr_address = NULL;
	cudaGetSymbolAddress((void**)&dev_aggr_address, dev_aggr_const);
	int dev_aggr_check[1024];
	err = cudaMemcpyFromSymbol(dev_aggr_check, dev_aggr_const, 1024 * sizeof(int));
	if (err != cudaSuccess)
	{
		printf("CUDA Error: %s\n", cudaGetErrorString(err));
		return err;
	}
#endif // DEBUG
	err = cudaMemcpyToSymbol(dev_aggr_const3, host_aggr_const3, 1024 * sizeof(int3));
	if (err != cudaSuccess)
	{
		printf("CUDA Error: %s\n", cudaGetErrorString(err));
		return err;
	}
#ifdef DEBUG
	int3 dev_aggr_check3[1024];
	err = cudaMemcpyFromSymbol(dev_aggr_check3, dev_aggr_const3, 1024 * sizeof(int));
	if (err != cudaSuccess)
	{
		printf("CUDA Error: %s\n", cudaGetErrorString(err));
		return err;
	}
#endif // DEBUG

	return err;
}


/*
 初始化视差和视差平面
*/
__global__ void cuRandomInitialDisparityAndItsPlane(float* disparity, float3* disparityplane,
													size_t min_disparity,size_t disparity_range,
													size_t width, size_t height)
{
	size_t col = blockDim.x * blockIdx.x + threadIdx.x;
	size_t row = blockDim.y * blockIdx.y + threadIdx.y;
	size_t id = row * width + col;
	if (col >= 0u && row >= 0u &&
		col < width && row < height)
	{
		disparity[id] = min_disparity + disparity[id] * disparity_range;
		float rand_x = disparityplane[id].x - 0.5f;
		float rand_y = disparityplane[id].y - 0.5f;
		float rand_z = disparityplane[id].z - 0.5f;
		rand_z = (abs(rand_z) < 0.001f) ? 0.001f : rand_z;
		disparityplane[id].x = -rand_x / rand_z;//a
		disparityplane[id].y = -rand_y / rand_z;//b
		//d = a*col+b*row+c;		c = d-a*col-b*row
		disparityplane[id].z = disparity[id] - disparityplane[id].x*float(col) - disparityplane[id].y * float(row);
	}
}

cudaError_t RandomInitialDisparityAndItsPlane(float* disparity, float3* disparityplane,
	size_t min_disparity, size_t disparity_range,
	size_t width, size_t height,
	dim3 blockpergrid, dim3 threadsperblock)
{
	cuRandomInitialDisparityAndItsPlane << <blockpergrid, threadsperblock >> > (disparity, disparityplane,
		min_disparity, disparity_range, width, height);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		printf("CUDA Error: %s\n", cudaGetErrorString(err));
	}
	return err;
}

/*
 灰度图计算
*/
__global__ void cuColorToGray(float *gray_image,uchar3 *color_image)
{
    size_t col = blockDim.x * blockIdx.x + threadIdx.x;
    size_t row = blockDim.y * blockIdx.y + threadIdx.y;

    if (col >= 0u && row >= 0u && col < kWidthDev && row < kHeightDev)
    {
		size_t id = row * kWidthDev + col;
		gray_image[id] = 0.114f * (float)color_image[id].x;
		gray_image[id] += 0.587f * (float)color_image[id].y;
		gray_image[id] += 0.299f * (float)color_image[id].z;
    }

}

cudaError_t Bgr2Gray(float *gray_image, uchar3 *color_image,
	size_t width, size_t height,
	dim3 blockpergrid, dim3 threadsperblock)
{
	cuColorToGray << <blockpergrid, threadsperblock >> > (gray_image, color_image);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		printf("CUDA Error: %s\n", cudaGetErrorString(err));
	}
	return err;
}

/*
 梯度计算
*/

__global__ void cuSobelGrad( float3* grad,float* gray_image)
{
	unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int id = row * kWidthDev + col;
	/*
	边界部分需要补充
	*/
	if (col > 0u && row > 0u && col < kWidthSubDev && row < kHeightSubDev)
	{
		grad[id].x = gray_image[id - kWidthSubDev] + 2.f * gray_image[id + 1u] + gray_image[id + kWidthAddDev] -
					 gray_image[id - kWidthAddDev] - 2.f * gray_image[id - 1u] - gray_image[id + kWidthSubDev];

		grad[id].y = gray_image[id - kWidthAddDev] + 2.f * gray_image[id - kWidthDev] + gray_image[id - kWidthSubDev] -
					 gray_image[id + kWidthSubDev] - 2.f * gray_image[id + kWidthDev] - gray_image[id + kWidthAddDev];
		
		
		grad[id].z = sqrtf(float(grad[id].x * grad[id].x + grad[id].y * grad[id].y));

	}
}

cudaError_t Gray2Sobel(float3* grad_image, float* gray_image,
	size_t width, size_t height,
	dim3 blockpergrid, dim3 threadsperblock)
{
	cuSobelGrad << < blockpergrid, threadsperblock >> > (grad_image, gray_image);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		printf("CUDA Error: %s\n", cudaGetErrorString(err));
	}
	return err;
}

/*
单点视差聚合代价计算
*/
__global__ void CuCostAggrSingle(float *costaggr,float3 *plane,
								unsigned int center_x,unsigned int center_y,
								unsigned int image_width, 
								uchar3 *left_color_image, uchar3 *right_color_image,
								float3 *left_grad_image, float3 *right_grad_image,
								float alpha, float tau_color, float tau_grad, float gama)
{
	__shared__ float cost_single[1024];
	__shared__ float cost_sum;
	unsigned int center_id = center_y * image_width + center_x;
	unsigned int thread_id = blockDim.x*threadIdx.y + threadIdx.x;
	unsigned int id_l = center_id + ((threadIdx.y - 16)* image_width + (threadIdx.x - 16)); //offset;
	unsigned int id_r = id_l - (plane[center_id].x * (center_x + threadIdx.x-16) +
								plane[center_id].y * (center_y + threadIdx.y-16) + plane[center_id].z);

	float cost = 0.f;
	cost = abs(left_color_image[id_l].x - right_color_image[id_r].x);
	cost += abs(left_color_image[id_l].y - right_color_image[id_r].y);
	cost += abs(left_color_image[id_l].z - right_color_image[id_r].z);
	float weight = exp(-cost * gama);
	cost_single[thread_id] = fmin(tau_color,cost);
	cost = abs(left_grad_image[id_l].x - right_grad_image[id_r].x);
	cost += abs(left_grad_image[id_l].y - right_grad_image[id_r].y);
	cost += abs(left_grad_image[id_l].z - right_grad_image[id_r].z);
	cost = fmin(cost, tau_grad);
	cost_single[thread_id] = weight * (alpha * cost_single[thread_id] + (1.f - alpha) * cost);

	if (0 == thread_id)
	{
		cost_sum = 0;
	}
	__syncthreads();
	atomicAdd(&cost_sum, cost_single[thread_id]);
	__syncthreads();
	costaggr[center_id] = cost_sum;
}


/*
 全图视差聚合代价计算
*/
__global__ void CuCaculateCostAggregationInit(float *costaggr, float3* disptyplane,
	uchar3 *left_color_image, uchar3 *right_color_image,
	float3 *left_grad_image, float3 *right_grad_image,
	float alpha, float tau_color, float tau_grad,
	float min_disp, float max_disp, float cost_punish, float gama,
	size_t patch_size, size_t width, size_t height)
{
	/*
	patch_size = r;total_PatchSize  = 2*patch_size
	*/
	unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;

	if (col >= patch_size && col < width - patch_size &&
		row >= patch_size && row < height - patch_size)
	{
		dim3 block(32, 32, 1);
		CuCostAggrSingle << <1, block >> > (costaggr, disptyplane, col, row, width,
											left_color_image, right_color_image,
											left_grad_image, right_grad_image,
											alpha, tau_color, tau_grad, gama);

	}

}


cudaError_t CaculateCostAggregationInit(float *costaggr, float3* disptyplane,
	uchar3 *left_color_image, uchar3 *right_color_image,
	float3 *left_grad_image, float3 *right_grad_image,
	float alpha, float tau_color, float tau_grad,
	float min_disp, float max_disp, float cost_punish, float gama,
	size_t patch_size, size_t width, size_t height,
	dim3 blockpergrid, dim3 threadsperblock)
{
	CuCaculateCostAggregationInit << < blockpergrid, threadsperblock >> > (costaggr, disptyplane,
		left_color_image, right_color_image,
		left_grad_image, right_grad_image,
		alpha, tau_color, tau_grad,
		min_disp, max_disp, cost_punish, gama,
		patch_size, width, height);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		printf("CUDA Error: %s\n", cudaGetErrorString(err));
	}
	return err;
}

/*
梯度视差代价计算
*/
//__global__ void CuCaculateCostGradAggrInit(float *costgradaggr, float3* disptyplane,
//	float3 *left_grad_image, float3 *right_grad_image,
//	float tau_grad, float min_disp, float max_disp,
//	size_t patch_size, size_t width, size_t height)
//{
//	/*
//	total_PatchSize  = 2*patch_size+1
//	*/
//	size_t col = blockDim.x * blockIdx.x + threadIdx.x;
//	size_t row = blockDim.y * blockIdx.y + threadIdx.y;
//	size_t id = row * width + col;
//
//	if (col >= patch_size && col < width - patch_size &&
//		row >= patch_size && row < height - patch_size)
//	{
//		row = row - patch_size;
//		col = col - patch_size;
//
//		size_t id_l = 0u;
//		size_t disp = 0u;
//		costgradaggr[id] = 0.f;
//		for (; row < height; ++row);
//		{
//			for (; col <= width; ++col);
//			{
//				id_l = row * width + col;
//				disp = size_t(disptyplane[id].x*col + disptyplane[id].y*row + disptyplane[id].z);/*使用纹理内存，可以不用转换数据类型*/
//				if ((col - disp) >= 0 && (col - disp <= width) && disp >= min_disp && disp <= max_disp)
//				{
//					costgradaggr[id] = abs(left_grad_image[id_l].x - right_grad_image[id_l - disp].x);
//					costgradaggr[id] += abs(left_grad_image[id_l].y - right_grad_image[id_l - disp].y);
//					costgradaggr[id] += abs(left_grad_image[id_l].z - right_grad_image[id_l - disp].z);
//					costgradaggr[id] = fmin(costgradaggr[id], tau_grad);
//				}
//			}
//		}
//
//
//	}
//
//}
//
//cudaError_t CaculateCostGradAggrInit(float *costgradaggr, float3* disptyplane,
//	float3 *left_grad_image, float3 *right_grad_image,
//	float tau_grad, float min_disp, float max_disp,
//	size_t patch_size, size_t width, size_t height,
//	dim3 blockpergrid, dim3 threadsperblock)
//{
//	cudaError_t status = cudaSuccess;
//
//	CuCaculateCostGradAggrInit << < blockpergrid, threadsperblock >> > (costgradaggr, disptyplane,
//		left_grad_image, right_grad_image,
//		tau_grad, min_disp, max_disp,
//		patch_size, width, height);
//	status = cudaGetLastError();
//	return status;
//}



//--------------------------------------const--------------------------------------//
__global__ void CuCostAggrSingleConst(float *costaggr, float3 *plane,
										unsigned int center_x, unsigned int center_y,
										uchar3 *left_color_image, uchar3 *right_color_image,
										float3 *left_grad_image, float3 *right_grad_image)
{

	__shared__ float cost_single[1024];
	__shared__ float cost_sum;
	int center_id = center_y * kWidthDev + center_x;
	int thread_id = threadIdx.x;

	int id_l = center_id + dev_aggr_const3[thread_id].z;
	int id_r = id_l - (plane[center_id].x * (center_x + dev_aggr_const3[thread_id].x) +
		plane[center_id].y * (center_y + dev_aggr_const3[thread_id].y) +
		plane[center_id].z);
	float cost = 0.f;
	cost = abs(left_color_image[id_l].x - right_color_image[id_r].x);
	cost += abs(left_color_image[id_l].y - right_color_image[id_r].y);
	cost += abs(left_color_image[id_l].z - right_color_image[id_r].z);
	float weight = exp(-cost * dev_algorith_const.gamma);
	cost_single[thread_id] = fmin(dev_algorith_const.tau_color, cost);
	cost = abs(left_grad_image[id_l].x - right_grad_image[id_r].x);
	cost += abs(left_grad_image[id_l].y - right_grad_image[id_r].y);
	cost += abs(left_grad_image[id_l].z - right_grad_image[id_r].z);
	cost = fmin(cost, dev_algorith_const.tau_grad);
	cost_single[thread_id] = weight * (dev_algorith_const.alpha * cost_single[thread_id] + (1.f - dev_algorith_const.alpha) * cost);
	if (0 == thread_id)
	{
		cost_sum = 0;
	}
	__syncthreads();
	atomicAdd(&cost_sum, cost_single[thread_id]);
	__syncthreads();
	costaggr[center_id] = cost_sum;
}


__global__ void CuCaculateCostAggregationInitConst(float *costaggr, float3* disptyplane,
													uchar3 *left_color_image, uchar3 *right_color_image,
													float3 *left_grad_image, float3 *right_grad_image)
{
	/*
	total_PatchSize  = 2*patch_size
	*/
	unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;

	if (col >= dev_patch_size && col < kWidthDev - dev_patch_size &&
		row >= dev_patch_size && row < kHeightDev - dev_patch_size)
	{
		CuCostAggrSingleConst << <1, 1024 >> > (costaggr, disptyplane, col, row,
											left_color_image, right_color_image,
											left_grad_image, right_grad_image);

	}

}

__global__ void CuCaculateCostAggregationInitAllConst(float *costaggr, float3* disptyplane,
														uchar3 *left_color_image, uchar3 *right_color_image,
														float3 *left_grad_image, float3 *right_grad_image)
{
	/*
	total_PatchSize  = 2*patch_size+1
	*/
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	int row = blockDim.y * blockIdx.y + threadIdx.y;
	int center_id = row * kWidthDev + col;
	float cost_sum = 0;
	float cost_color = 0.f;
	float cost_grad = 0.f;
	int id_l = 0;
	int id_r = 0;
	float weight = 0;
	float disparity = 0.f;

	if (col >= dev_patch_size && col < kWidthDev - dev_patch_size &&
		row >= dev_patch_size && row < kHeightDev - dev_patch_size)
	{

		for (int row_id = -dev_patch_size; row_id <= dev_patch_size; ++row_id)
		{
			for (int col_id = -dev_patch_size; col_id <= dev_patch_size; ++col_id)
			{

				id_l = (row + row_id)* kWidthDev + (col + col_id);
				disparity = (disptyplane[center_id].x * (col + col_id) +
							disptyplane[center_id].y * (row + row_id) +
							disptyplane[center_id].z);

				if (disparity<0 || disparity > dev_algorith_const.max_disparity ||
					(col + col_id - disparity) < 0 || (col + col_id - disparity) >= kWidthDev)
				{
					cost_sum += dev_algorith_const.cost_punish;
					continue;
				}

				id_r = id_l - disparity;
				cost_color = abs(left_color_image[id_l].x - right_color_image[id_r].x);
				cost_color += abs(left_color_image[id_l].y - right_color_image[id_r].y);
				cost_color += abs(left_color_image[id_l].z - right_color_image[id_r].z);
				cost_grad = abs(left_grad_image[id_l].x - right_grad_image[id_r].x);
				cost_grad += abs(left_grad_image[id_l].y - right_grad_image[id_r].y);
				cost_grad += abs(left_grad_image[id_l].z - right_grad_image[id_r].z);
				weight = exp(-cost_color * dev_algorith_const.gamma);
				cost_color = fmin(dev_algorith_const.tau_color, cost_color);
				cost_grad = fmin(cost_grad, dev_algorith_const.tau_grad);
				cost_sum += weight * (dev_algorith_const.alpha * (cost_color - cost_grad) + cost_grad);
			}
		}
		costaggr[center_id] = cost_sum/((2* dev_patch_size+1)*(2 * dev_patch_size + 1));
	}
	else
	{
		costaggr[center_id] = dev_algorith_const.cost_punish;
	}
	__syncthreads();
}

cudaError_t CaculateCostAggregationInitConst(float *costaggr, float3* disptyplane,
											uchar3 *left_color_image, uchar3 *right_color_image,
											float3 *left_grad_image, float3 *right_grad_image,
											dim3 blockpergrid, dim3 threadsperblock)
{

		CuCaculateCostAggregationInitAllConst << < blockpergrid, threadsperblock >> > (costaggr, disptyplane,
																				left_color_image, right_color_image,
																				left_grad_image, right_grad_image);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		printf("CUDA Error: %s\n", cudaGetErrorString(err));
	}
	return err;
}










