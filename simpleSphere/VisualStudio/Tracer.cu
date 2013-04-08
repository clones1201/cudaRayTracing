#include "defines.cuh"
#include "function_defines.cuh"

__device__
void singleSphereTraceRay(World *w,Sphere *sphere,Ray ray, RGBColor *color){
		ShadeRec sr;
		//cudaMalloc(&sr,sizeof(ShadeRec));
		//Sphere s;
		//s.center = Point3D(0.0,0.0,300);
		//s.radius = 200;

		float t;

		if( Hit((GeometricObject*)sphere,ray,&t,&sr))
			*color = (red);
		else
			*color = (black);

		//*color = make_uchar4( (unsigned char)( ray->o.x * ray->o.x + ray->o.y * ray->o.y ) % 255 , 0,0,255);
}

__device__ 
RGBColor multiObjTraceRay(World *w, Ray ray){
	ShadeRec sr;
	HitBareBonesObject(w,ray,&sr);

	if( sr.hitAnObject == true ){
		return sr.color;
	}
	else{
		return w->backgroundColor;
	}
}

__device__
RGBColor RayCastTraceRay(World *w, Ray ray ,int depth){
	ShadeRec sr;
	sr.w = w;

	HitObject(w,ray,&sr);

	if( sr.hitAnObject ){
		sr.ray = ray;
		return Shade(sr.material,&sr);
	}
	else{
		return w->backgroundColor;
	}
}