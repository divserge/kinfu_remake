#include "kfusion.h"
#include <iostream>

using kfusion::SampledScopeTime;

bool KinectFusionWrapper::capture_depth(
	const std::vector<uint16_t>& depth_data,
	uint32_t nrows,
	uint32_t ncols
) {
	
	depth_device_.upload(&depth_data[0], 2 * ncols, nrows, ncols);
	
	bool has_image = false;

    has_image = (*kinfu_.get())(depth_device_);

    return has_image;
}

TsdfWrapper KinectFusionWrapper::get_tsdf() {
	
	auto dims = kinfu_->tsdf().getDims();

	TsdfWrapper tsdf = {
		std::vector<float> (dims[0] * dims[1] * dims[2], 0),
		dims[0],
		dims[1],
		dims[2]
	};

	kinfu_->tsdf().data().download(&tsdf.tsdf[0]);
	
	return tsdf;
};