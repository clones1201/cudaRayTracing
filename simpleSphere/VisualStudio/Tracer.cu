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

__device__
RGBColor PathTraceRay(World *w, Ray ray, int depth){
	ShadeRec sr;
	sr.w = w;
	
	bool noMore = true;
	Ray temp = ray;
	RGBColor result = black;
	if( depth != 0){
		HitObject(w,temp,&sr);
	
		if( sr.hitAnObject ){
			sr.ray = temp;
			Vector3D wi;
			result = PathShade(sr.material,&sr,&wi) ;
			Ray newRay;
			newRay.o = sr.hitPoint;
			newRay.d = wi;
			temp = newRay;
			noMore = false;
		}
		else{
			result = result + w->backgroundColor;
			noMore = true;
		}
	}
	for( int d = 1 ; d < depth ; d++ ){
		if( ! noMore ){
			
			HitObject(w,temp,&sr);

			if( sr.hitAnObject ){
				sr.ray = temp;
				Vector3D wi;
				result = powc( result , PathShade(sr.material,&sr,&wi));
				Ray newRay;
				newRay.o = sr.hitPoint;
				newRay.d = wi;
				temp = newRay;
			}
			else{
				result = result + w->backgroundColor;
				noMore = true;
			}
		}
	}
	return result;
}

__device__
RGBColor WhittedRayTrace(World *w, Ray ray, int depth){
	ShadeRec sr;
	sr.w = w;
	
	Ray temp = ray;
	RGBColor result = black;
	
	Stack<RGBColor> stack;
	initStack(&stack);
	RGBColor l = black;
	
	for( int d = 0 ; d < depth ; d++ ){
			
		HitObject(w,temp,&sr);

		if( sr.hitAnObject ){
			sr.ray = temp;
			Vector3D wi;
		//	if( sr.material->type == MATERIAL_TYPE_REFLECTIVE ){
				Push(&stack,Shade(sr.material,&sr));
				Push(&stack,PathShade(sr.material,&sr,&wi));
				Ray newRay;
				newRay.o = sr.hitPoint;

				float ndotwi;

				newRay.d = temp.d + 2.0 * (Vector3D(0,0,0) - temp.d ) * sr.normal * sr.normal;
				temp = newRay;
				if( d == depth - 1)
					Push(&stack,black); 
		/*	}else{
				Push(&stack,Shade(sr.material,&sr));
				break;
			}*/
		}
		else{
			Push(&stack,black);
			break;
		}
	}

	
	do{
		if(!isEmpty(&stack))
			l = Pop( &stack );
		if(!isEmpty(&stack))
			Push( &stack,powc(l , Pop(&stack)) + Pop(&stack) );
	}while( !isEmpty(&stack) );

	result = result + l;
	return result;

}