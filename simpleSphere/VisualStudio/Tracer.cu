#include "defines.cuh"

__device__ inline
void singleSphereTraceRay(World *w,Sphere *sphere,Ray ray, RGBAColor *color){
		ShadeRec sr;
		//cudaMalloc(&sr,sizeof(ShadeRec));
		Sphere s;
		s.center = Point3D(0.0,0.0,300);
		s.radius = 200;

		float t;

		if( sphereHit(&s,ray,&t,&sr))
			*color = (red);
		else
			*color = (black);

		//*color = make_uchar4( (unsigned char)( ray->o.x * ray->o.x + ray->o.y * ray->o.y ) % 255 , 0,0,255);
}

__device__ inline
void multiObjTraceRay(World *w, Ray ray,RGBAColor *color){
	ShadeRec sr;
	hitBareBonesObject(w,ray,&sr);

	if( sr.hitAnObject == true ){
		*color = sr.color;
	}
	else{
		*color = w->backgroundColor;
	}
}