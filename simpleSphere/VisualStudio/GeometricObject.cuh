#ifndef __GEOMETRICOBJECT_CUH__
#define __GEOMETRICOBJECT_CUH__

#include "simpleSphere.cuh"
#include "ShadeRec.cuh"
#include "Ray.cuh"

__device__ static const float kEpsilon = 0.001;

struct Sphere{
	Point3D	center;
	float radius;
};

__device__ __host__
bool sphere_hit(Sphere *sphere,Ray *ray,float tmin, ShadeRec *sr);

#endif