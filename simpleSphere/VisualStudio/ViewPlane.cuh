#if !defined __VIEWPLANE_CUH__
#define __VIEWPLANE_CUH__

#include"simpleSphere.cuh"

struct ViewPlane{
public:
	int hres;
	int vres;
	float s;
	float gamma;
	float inv_gamma;

	RGBAColor *buffer;

};

#endif