#ifndef  _DATA_TYPE_H
#define _DATA_TYPE_H

enum ImgType
{
	UcharImg = 1,
	Uchar3Img = 3,
	FloatImg = 4,
	Float3Img = 12,
	DoubleImg = 8,
	Double3Img = 24,
};

struct ImageInfo
{
	size_t width = 0u;
	size_t height = 0u;
	size_t imgsize = width * height;
	ImgType imgtype = UcharImg;

	ImageInfo(size_t width_, size_t height_,ImgType imgtype_) :
		     width(width_), height(height_),imgtype(imgtype_){
		imgsize = width_ * height_;
	}
};

struct AlogrithConfigInt {

	size_t patch_size = 16u;			
	size_t num_iters = 3u;
};

struct AlogrithConfigFloat {

	float min_disparity;//0.f;
	float max_disparity;//64.f;
	float disparity_range;//64.f;

	float gamma;//0.1f;
	float alpha;//0.9f;

	float tau_color;//10.f;
	float tau_grad;//2.f;
	float cost_punish;//120.f;
	float lrcheck_thres;//10.f;		// 

	float is_check_lr;//1.f;		// check uniform left right disparity 
	float is_fill_holes;//1.f;		// file holes
	float is_fource_fpw;//1.f;		// force Frontal-Parallel Window
	float is_integer_disp;//1.f;	// 

};


#endif // ! _DATA_TYPE_H


