#include "PatchMatchImp.h"


template <typename T>
void check(T result, char const* const func, const char* const file, int const line) {
    if (result) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), cudaGetErrorName(result), func);
        exit(EXIT_FAILURE);
    }
}



__global__ void CuInitDisparity(ImageInfo* image_info, AlogrithConfig* algori_config,
    float* device_disparity_left, float* device_disparity_right,
    float* rand_disparity_left_data, float* rand_disparity_right_data)
{
    int thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
    int thread_idy = blockDim.y * blockIdx.y + threadIdx.y;
    int point_id = thread_idx * image_info->width + thread_idy;
    device_disparity_left[point_id] = algori_config->min_disparity + rand_disparity_left_data[point_id] * algori_config->disparity_range;
    device_disparity_right[point_id] = algori_config->min_disparity + rand_disparity_right_data[point_id] * algori_config->disparity_range;

}


__global__ void CuPlaneToDisparity(float *disp,float3 *plane, size_t width, size_t height)
{
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int id = row * width + col;
    if (col > 0 && row > 0 && col < width - 1 && row < height - 1)
    {
        disp[id] = plane[id].x * col + plane[id].y * row + plane[id].z;//d = ax+by+c = a*col+b*row+c
    }
}



/*
基于颜色空间color_image、梯度空间grad_image计算视察disparity下的损失cost
*/
__global__ void CuCaculateCost(float *cost,float* disparity,
                                uchar3*left_color_image, uchar3* right_color_image,
                                float3* left_grad_image, float3* right_grad_image,
                                float alpha, float tau_color,float tau_grad,size_t width, size_t height)
{
    size_t col = blockDim.x * blockIdx.x + threadIdx.x;
    size_t row = blockDim.y * blockIdx.y + threadIdx.y;
    size_t id_left = row * width + col;
    size_t id_right = id_left - disparity[id_left];

    float color_cost = abs(left_color_image[id_left].x - right_color_image[id_right].x);
    color_cost += abs(left_color_image[id_left].y - right_color_image[id_right].y);
    color_cost += abs(left_color_image[id_left].z - right_color_image[id_right].z);
    
    float grad_cost = abs(left_grad_image[id_left].x - right_grad_image[id_right].x);
    grad_cost += abs(left_grad_image[id_left].y - right_grad_image[id_right].y);
    grad_cost += abs(left_grad_image[id_left].z - right_grad_image[id_right].z);

    color_cost = fmin(color_cost, tau_color);
    grad_cost = fmin(grad_cost, tau_grad);
    cost[id_left] = (1 - alpha) * grad_cost + alpha * color_cost;
}


__global__ void CuCaculateCostAggregation(float3* disp_plane,float min_disp, float max_disp,float cost_punish,
                                        size_t patch_size, size_t width, size_t height)
{
    float cost = 0.f;
    size_t col = blockDim.x * blockIdx.x + threadIdx.x;
    size_t row = blockDim.y * blockIdx.y + threadIdx.y;
    size_t col_curr = 0;
    size_t row_curr = 0;
    size_t id_curr = 0;
    //size_t id_curr = 0;
    for (size_t row_offset = -patch_size; row_offset <= patch_size; row_offset++)
    {
        row_curr = row + row_offset;
        for (size_t col_offset = -patch_size; col_offset <= patch_size; col_offset++)
        {
            col_curr = col + col_offset;
            id_curr = col_offset * width + col_curr;
            if (col_curr > 0 && row_curr > 0 && col_curr < width - 1 && row_curr < height - 1)
            {
                float disp_curr = disp_plane[id_curr].x * col_curr + disp_plane[id_curr].y * row_curr + disp_plane[id_curr].z;
                if (disp_curr<min_disp || disp_curr>max_disp)
                {
                    cost += cost_punish;
                }
                else
                {
                    ;
                }
            }

        }
    }


}


__global__ void  SpatialPropagation()
{
    ;
}


__global__ void cuSobelGrad(unsigned char* gray_image, short3* grad, size_t width, size_t height)
{
    size_t col = blockDim.x * blockIdx.x + threadIdx.x;
    size_t row = blockDim.y * blockIdx.y + threadIdx.y;
    size_t id = row * width + col;

    if (col > 0 && row > 0 &&col < width - 1 && row < height - 1)
    {
        grad[id].x =  gray_image[id - width + 1] + 2 * gray_image[id + 1] + gray_image[id + width + 1] -
            gray_image[id - width - 1] - 2 * gray_image[id - 1] - gray_image[id + width - 1];
       
        grad[id].y = gray_image[id - width - 1] + 2 * gray_image[id - width] + gray_image[id - width + 1] -
             gray_image[id + width - 1] - 2 * gray_image[id + width] - gray_image[id + width + 1];
        grad[id].z = sqrtf(float(grad[id].x * grad[id].x + grad[id].y * grad[id].y));

    }
}


__global__ void cuColorToGray(float *gray_image,uchar3 *color_image, size_t width, size_t height)
{
    size_t col = blockDim.x * blockIdx.x + threadIdx.x;
    size_t row = blockDim.y * blockIdx.y + threadIdx.y;
    size_t id = row * width + col;

    if (col > 0 && row > 0 && col < width - 1 && row < height - 1)
    {
       gray_image[id] = 0.114f * (float)color_image[id].x + 0.587f * (float)color_image[id].y + 0.299f * (float)color_image[id].z;
    }

}




