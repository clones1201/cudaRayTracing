#include "defines.cuh"

__device__ static const float kEpsilon = 0.001;

__device__ inline 
bool sphereHit(
	Sphere *sphere,Ray ray, float *tmin ,ShadeRec *sr){
		float t;
		Vector3D temp = ray.o - sphere->center;
		float a = ray.d * ray.d;
		float b = 2.0 * temp * ray.d;
		float c = temp * temp - sphere->radius * sphere->radius;
		float disc = b * b - 4 * a * c;

		if( disc < 0.0 )
			return false;
		else{
			float e = sqrt(disc);
			float denom = 2.0 * a ;
			t = ( - b - e ) / denom;

			if( t > kEpsilon ){
				*tmin = t;
				sr->normal = ( temp + t * ray.d ) / sphere->radius;
				sr->localHitPoint = ray.o + t * ray.d;
				return true;
			}

			t = ( - b + e ) / denom;

			if( t > kEpsilon ){
				*tmin = t;
				sr->normal = ( temp + t * ray.d ) / sphere->radius;
				sr->localHitPoint = ray.o + t * ray.d;
				return true;
			}
		}
		return false;
}

__device__ inline 
bool planeHit(Plane *plane, Ray ray, float *tmin, ShadeRec *sr){
	float t = (plane->point - ray.o) * plane->normal /( ray.d * plane->normal );

	if( t > kEpsilon ){
		*tmin = t;
		sr->normal = plane->normal;
		sr->localHitPoint = ray.o + t * ray.d;

		return true;
	}
	else{
		return false;
	}
}

__host__
void initPlane(Plane **pl, Point3D p,Normal n,RGBAColor c){
	Plane plane;
	plane.normal = n;
	plane.point = p;
	plane.color = c;
	plane.type = GMO_TYPE_PLANE;

	cudaMalloc(pl,sizeof(Plane));
	cudaCheckErrors("plane memory allocate failed");

	cudaMemcpy((void*)*pl,&plane,sizeof(Plane),cudaMemcpyHostToDevice);
	cudaCheckErrors("plane memory copy failed");
}

__host__
void initSphere(Sphere **s, Point3D c, float r,RGBAColor cl){
	Sphere sphere;
	sphere.center = c;
	sphere.radius = r;
	sphere.color = cl;
	sphere.type = GMO_TYPE_SPHERE;

	cudaMalloc(s,sizeof(Sphere));
	cudaCheckErrors("sphere memory allocate failed");

	cudaMemcpy((void*)*s,&sphere,sizeof(Sphere),cudaMemcpyHostToDevice);
	cudaCheckErrors("sphere memory copy failed");
}

__host__
void freePlane(Plane *pl){
	cudaFree(pl);
	cudaCheckErrors("plane memory free failed");
}

__host__
void freeSphere(Sphere *s){
	cudaFree(s);
	cudaCheckErrors("sphere memory free failed");
}
