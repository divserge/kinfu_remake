from libcpp.vector cimport vector
from libc.stdint cimport uint32_t, uint16_t
from libcpp cimport bool

cdef extern from "kfusion.h":
	ctypedef struct TsdfWrapper:
		vector[float] tsdf
		uint32_t dim_x
		uint32_t dim_y
		uint32_t dim_z

	cdef cppclass KinectFusionWrapper:
		KinectFusionWrapper(int)
		bool capture_depth(vector[uint16_t], uint32_t, uint32_t)
		TsdfWrapper get_tsdf()


cdef class PyTsdfWrapper:
	cdef TsdfWrapper cpp_tsdf
	def __cinit__(self, wrapper):
		self.cpp_tsdf = TsdfWrapper(
			wrapper['tsdf'],
			wrapper['dim_x'],
			wrapper['dim_y'],
			wrapper['dim_z']
		)
	def get_dims(self):
		return (self.cpp_tsdf.dim_x, self.cpp_tsdf.dim_y, self.cpp_tsdf.dim_z)
	def get_data(self):
		return self.cpp_tsdf.tsdf

cdef class PyKinectFusion:
	cdef KinectFusionWrapper* cpp_kinfu
	def __cinit__(self, device):
		self.cpp_kinfu = new KinectFusionWrapper(device)
	def capture(self, data, nrows, ncols):
		return self.cpp_kinfu.capture_depth(data, nrows, ncols)

	def get_tsdf(self):
		return PyTsdfWrapper(self.cpp_kinfu.get_tsdf())