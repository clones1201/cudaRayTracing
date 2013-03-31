#if !defined __RAY_CUH__
#define __RAY_CUH__

#include "Tools.cuh"

struct Ray{
	Point3D o;
	Vector3D d;
};

#endif