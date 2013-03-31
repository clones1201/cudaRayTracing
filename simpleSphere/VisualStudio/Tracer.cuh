#ifndef __TRACER_CUH__
#define __TRACER_CUH__

#include "simpleSphere.cuh"
#include "GeometricObject.cuh"
#include "Ray.cuh"
#include "Tools.cuh"
#include "World.cuh"

__device__ __host__
void singleSphere_traceRay(World *w,Sphere *s,Ray *ray,RGBAColor *color);

#endif