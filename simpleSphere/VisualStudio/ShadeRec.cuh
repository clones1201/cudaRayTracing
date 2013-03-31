#if !defined __SHADEREC_CUH__
#define __SHADEREC_CUH__

#include "simpleSphere.cuh"

struct ShadeRec{
	bool hit_an_object;
	Point3D local_hit_point;
	Normal normal;
	RGBAColor color;	
};

#endif