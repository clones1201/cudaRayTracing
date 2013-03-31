#include "Sphere.cuh"

__device__ __host__
bool Sphere::hit(
	const Ray &ray, float &tmin, ShadeRec &sr )const{
		float t;
		Vector3D temp = ray.o - center;
		float a = ray.d * ray.d;
		float b = 2.0 * temp * ray.d;
		float c = temp * temp - radius * radius;
		float disc = b * b - 4.0 * a * c;

		if( disc < 0.0 )
			return false;
		else{
			float e = sqrt(disc);
			float denom = 2.0 * a ;
			t = ( - b - e ) / denom;

			if( t > kEpsilon ){
				tmin = t;
				sr.normal = ( temp + t * ray.d ) / radius;
				sr.local_hit_point = ray.o + t * ray.d;
				return true;
			}

			t = ( - b + e ) / denom;

			if( t > kEpsilon ){
				tmin = t;
				sr.normal = ( temp + t * ray.d ) / radius;
				sr.local_hit_point = ray.o + t * ray.d;
				return true;
			}
		}
		return false;
}