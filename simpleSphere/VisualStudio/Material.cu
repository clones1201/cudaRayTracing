#include "defines.cuh"
#include "function_defines.cuh"

__device__ inline
RGBColor MatteShade(Matte *m, ShadeRec *sr){
	Vector3D wo = Vector3D(0,0,0) - sr->ray.d;

	RGBColor result = powc( Rho((BRDF*)&(m->ambientBRDF),sr,&wo) , L(sr->w->ambient,sr));
	//RGBColor result = red * 0.25 * white;

	for(int j = 0; j < sr->w->numLight ; ++j ){
		Vector3D wi = GetDirection((sr->w->lights[j]),sr);
		float ndotwi = sr->normal * wi;
		if( ndotwi > 0.01 ){
			bool isShadow = false;

			if( sr->w->lights[j]->shadows ){
				Ray shadowRay;
				shadowRay.d = wi;shadowRay.o = sr->hitPoint;
				isShadow = inShadow(sr->w->lights[j],shadowRay,sr);
			}

			if(!isShadow){
				result = result + powc(F((BRDF*)&(m->diffuseBRDF),sr,&wo,&wi) , L(sr->w->lights[j],sr)) * ndotwi;
			}
		}
	}
	return (result);
}

__device__ inline
RGBColor PhongShade(Phong *pg, ShadeRec *sr){
	Vector3D wo = Vector3D(0,0,0) - sr->ray.d;
	RGBColor result = powc( Rho((BRDF*)&(pg->ambientBRDF),sr,&wo) , L(sr->w->ambient,sr) );

	for(int j = 0; j < sr->w->numLight ; ++j ){
		Vector3D wi = GetDirection((sr->w->lights[j]),sr);
		float ndotwi = sr->normal * wi;
		
		if( ndotwi > 0.01 ){
			bool isShadow = false;

			if( sr->w->lights[j]->shadows ){
				Ray shadowRay;
				shadowRay.d = wi;shadowRay.o = sr->hitPoint;
				isShadow = inShadow(sr->w->lights[j],shadowRay,sr);
			}

			if(!isShadow){
				result = result + 
					powc ( (F((BRDF*)&(pg->diffuseBRDF),sr,&wo,&wi) + F((BRDF*)&(pg->specularBRDF),sr,&wo,&wi)),
						L(sr->w->lights[j],sr))  * ndotwi;
			}
		}
	}
	return result;
}

__device__
RGBColor ReflectiveShade( Reflective * ref, ShadeRec *sr){
	return PhongShade( (Phong*) ref, sr);
}

__device__
RGBColor Shade(Material *m, ShadeRec *sr){
	switch( m->type ){
	case MATERIAL_TYPE_MATTE:
		return MatteShade((Matte*)m, sr);
	case MATERIAL_TYPE_PHONG:
		return PhongShade((Phong*)m, sr);
	case MATERIAL_TYPE_REFLECTIVE:
		return ReflectiveShade((Reflective*)m,sr);
	default:
		return (black);
	}
}


__device__ inline
RGBColor MattePathShade(Matte *m, ShadeRec *sr,Vector3D *wi){
	Vector3D wo = Vector3D(0,0,0) - sr->ray.d;
	float pdf;
	RGBColor result = SampleF( (BRDF*)& m->diffuseBRDF ,sr,wi,&wo,&pdf);
	float ndotwi = sr->normal * *wi;
	result = result * ndotwi / pdf;
	return black;
}

__device__ inline
RGBColor PhongPathShade(Phong *pg, ShadeRec *sr,Vector3D *wi){
	Vector3D wo = Vector3D(0,0,0) - sr->ray.d;
	
	float pdf1,pdf2;
	RGBColor result = SampleF( (BRDF*)& pg->specularBRDF ,sr,wi,&wo,&pdf1)
			+ SampleF( (BRDF*)& pg->diffuseBRDF ,sr,wi,&wo,&pdf2 );

	float ndotwi = sr->normal * *wi;
	
	result = result * ndotwi / (pdf1 * pdf2) * (pdf1 + pdf2);
	return result;
}


__device__ inline
RGBColor ReflectivePathShade(Reflective *ref, ShadeRec *sr, Vector3D *wi){
	//RGBColor result = PhongShade( (Phong*)ref, sr);
	Vector3D wo = Vector3D(0,0,0) - sr->ray.d;
	float pdf;
	RGBColor result = SampleF((BRDF*)& ref->reflectiveBRDF , sr, wi, &wo,/* useless*/ &pdf);

	result = result * (sr->normal * *wi);
	return white;
}

__device__
RGBColor PathShade(Material *m, ShadeRec *sr, Vector3D *wi){
	switch( m->type ){
	case MATERIAL_TYPE_MATTE:
		return MattePathShade((Matte*)m, sr,wi);
	case MATERIAL_TYPE_PHONG:
		return PhongPathShade((Phong*)m, sr,wi);
	case MATERIAL_TYPE_REFLECTIVE:
		return ReflectivePathShade( (Reflective*)m, sr, wi);
	default:
		return (black);
	}
}
