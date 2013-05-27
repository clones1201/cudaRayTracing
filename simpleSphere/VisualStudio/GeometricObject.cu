#include "defines.cuh"
#include "function_defines.cuh"

__device__ static const float kEpsilon = 0.01;

__device__ inline 
bool SphereHit(
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
				Normalize(sr->normal);
				sr->localHitPoint = ray.o + t * ray.d;
				return true;
			}

			t = ( - b + e ) / denom;

			if( t > kEpsilon ){
				*tmin = t;
				sr->normal = ( temp + t * ray.d ) / sphere->radius;
				Normalize(sr->normal);
				sr->localHitPoint = ray.o + t * ray.d;
				return true;
			}
		}
		return false;
}

__device__ inline 
bool PlaneHit(Plane *plane, Ray ray, float *tmin, ShadeRec *sr){
	float t = (plane->point - ray.o) * plane->normal /( ray.d * plane->normal );

	if( t > kEpsilon ){
		*tmin = t;
		sr->normal = plane->normal;
		Normalize(sr->normal);
		sr->localHitPoint = ray.o + t * ray.d;

		return true;
	}
	else{
		return false;
	}
}

__device__
bool Hit(GeometricObject *obj, Ray ray, float *tmin, ShadeRec *sr){
	switch( obj->type ){
	case GMO_TYPE_SPHERE:
		return SphereHit((Sphere*)obj,ray,tmin,sr);
	case GMO_TYPE_PLANE:
		return PlaneHit((Plane*)obj,ray,tmin,sr);
	default:
		return false;
	}
}

__device__ inline 
bool SphereShadowHit(
	Sphere *sphere,Ray ray, float *tmin){
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
				return true;
			}

			t = ( - b + e ) / denom;

			if( t > kEpsilon ){
				*tmin = t;
				return true;
			}
		}
		return false;
}

__device__ inline
bool PlaneShadowHit(
	Plane *plane,Ray ray, float *tmin){
		float t = (plane->point - ray.o) * plane->normal /( ray.d * plane->normal );

		if( t > kEpsilon ){
			*tmin = t;
			return true;
		}else{
			return false;
		}
}

__device__
bool ShadowHit(GeometricObject *obj, Ray ray, float *tmin){
	switch( obj->type ){
	case GMO_TYPE_SPHERE:
		return SphereShadowHit((Sphere*)obj,ray,tmin);
	case GMO_TYPE_PLANE:
		return PlaneShadowHit((Plane*)obj,ray,tmin);
	default:
		return false;
	}
}

__host__
void initPlane(Plane **pl, Point3D p,Normal n,int mIdx){
	
	(*pl) = (Plane*)malloc(sizeof(Plane));
	(*pl)->normal = n;
	(*pl)->point = p;
//	(*pl)->color = ((Matte*)m)->diffuseBRDF.cd;
	(*pl)->materialIdx = mIdx;
	(*pl)->type = GMO_TYPE_PLANE;

}

__host__
void initSphere(Sphere **s, Point3D c, float r,int mIdx){
	(*s) = (Sphere*)malloc(sizeof(Sphere));
	(*s)->center = c;
//	(*s)->color = ((Matte*)m)->diffuseBRDF.cd;
	(*s)->radius = r;
	(*s)->materialIdx = mIdx;
	(*s)->type = GMO_TYPE_SPHERE;
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
