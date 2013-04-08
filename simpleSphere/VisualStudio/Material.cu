#include "defines.cuh"
#include "function_defines.cuh"

__device__ inline
RGBColor MatteShade(Matte *m, ShadeRec *sr){
	Vector3D wo = Vector3D(0,0,0) - sr->ray.d;

	RGBColor result = Rho((BRDF*)&(m->ambientBRDF),sr,&wo) * L(sr->w->ambient,sr);
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
				result = result + F((BRDF*)&(m->diffuseBRDF),sr,&wo,&wi) * L(sr->w->lights[j],sr) * ndotwi;
			}
		}
	}
	return (result);
}

__device__ inline
RGBColor PhongShade(Phong *pg, ShadeRec *sr){
	return black;//temp value
}

__device__
RGBColor Shade(Material *m, ShadeRec *sr){
	switch( m->type ){
	case MATERIAL_TYPE_MATTE:
		return MatteShade((Matte*)m, sr);
	case MATERIAL_TYPE_PHONG:
		return PhongShade((Phong*)m, sr);
	default:
		return (black);
	}
}
