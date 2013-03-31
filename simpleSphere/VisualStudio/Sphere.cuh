#ifndef __SPHERE_CUH__
#define __SPHERE_CUH__

#include "simpleSphere_k.cuh"
#include "GeometricObject.cuh"
#include "ShadeRec.cuh"
#include "Tools.cuh"
#include "Ray.cuh"

class  Sphere : public GeometricObject{
private:
	Point3D center;
	float radius;

	const float kEpsilon;
public:
	__device__ __host__ Sphere(void);
	__device__ __host__ Sphere(const Point3D &ct,const float &r );
	__device__ __host__ 
	virtual bool hit( const Ray &ray, float &tmin, ShadeRec &sr) const;

	__device__ __host__
		void SetCenter(Point3D c);
	__device__ __host__
		void SetRadius(float r);
};

#endif
