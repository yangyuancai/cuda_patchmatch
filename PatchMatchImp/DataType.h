#ifndef  _DATA_TYPE_H
#define _DATA_TYPE_H

struct ImageInfo
{
	size_t width = 450u;
	size_t height = 375u;
	size_t image_size = width * height;
};

struct AlogrithConfig {
	size_t patch_size;			// patch尺寸，局部窗口为 patch_size*patch_size
	size_t min_disparity;		// 最小视差
	size_t max_disparity;		// 最大视差
	size_t disparity_range = max_disparity - min_disparity;
	size_t num_iters;			// 传播迭代次数

	float gamma;				// gamma 权值因子
	float alpha;				// alpha 相似度平衡因子
	float tau_color;			// tau for color	相似度计算颜色空间的绝对差的下截断阈值
	float tau_grad;				// tau for gradient 相似度计算梯度空间的绝对差下截断阈值

	size_t is_check_lr;			// 是否检查左右一致性
	float lrcheck_thres;		// 左右一致性约束阈值

	size_t is_fill_holes;		// 是否填充视差空洞
	size_t is_fource_fpw;		// 是否强制为Frontal-Parallel Window
	size_t is_integer_disp;		// 是否为整像素视差
};

#endif // ! _DATA_TYPE_H


