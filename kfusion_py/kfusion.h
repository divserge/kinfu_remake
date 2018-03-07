#pragma once

#include <cstdint>
#include <memory>

#include <kfusion/kinfu.hpp>
#include <kfusion/types.hpp>

using kfusion::KinFuParams;
using kfusion::KinFu;
using kfusion::cuda::Depth;
using kfusion::cuda::setDevice;
using kfusion::cuda::printShortCudaDeviceInfo;

class TsdfWrapper {
  public:
	std::vector<float> tsdf;
	uint32_t dim_x;
	uint32_t dim_y;
	uint32_t dim_z;
};


class KinectFusionWrapper {
  public:
	
	KinectFusionWrapper(int device) : kinfu_ (new KinFu(KinFuParams::default_params())) {
    	setDevice (device);
    	printShortCudaDeviceInfo (device);
	}
	
	bool capture_depth(
		const std::vector<uint16_t>& depth_data,
		uint32_t nrows,
		uint32_t cols
	);

	TsdfWrapper get_tsdf();
  private:
  	std::unique_ptr<KinFu> kinfu_;
    Depth depth_device_;
};