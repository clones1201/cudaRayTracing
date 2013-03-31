#if !defined __SINGLESPHERE_CUH__
#define __SINGLESPHERE_CUH__

#include "simpleSphere_k.cuh"
#include "Tracer.cuh"

class SingleSphere : public Tracer{
public:
	__device__ SingleSphere(void){}

	__device__ SingleSphere(World * w_ptr){}

	__device__ __host__ 
		RGBAColor trace_ray(const Ray& ray) const;
};

#endif