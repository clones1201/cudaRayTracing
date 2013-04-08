#include "defines.cuh"
#include "function_defines.cuh"

__device__ inline
RGBColor LambertianF(Lambertian* lbt,ShadeRec *sr, Vector3D *wi, Vector3D *wo){
	return (lbt->cd * lbt->kd);
}

__device__ inline
RGBColor LambertianRho(Lambertian* lbt, ShadeRec *sr, Vector3D *wo){
	return (lbt->cd * lbt->kd);
}

__device__ inline
RGBColor PerfectSpecularF(PerfectSpecular *ps, ShadeRec *sr, Vector3D *wi, Vector3D *wo){
	return (black);  //temp value
}

__device__ inline 
RGBColor PerfectSpecularRho(PerfectSpecular *ps, ShadeRec *sr, Vector3D *wo){
	return (black);//temp value
}

__device__ inline
RGBColor GlossySpecularF(GlossySpecular *gs, ShadeRec *sr, Vector3D *wi, Vector3D *wo){
	return (black);//temp value
}

__device__ inline
RGBColor GlossySpecularRho(GlossySpecular *gs, ShadeRec *sr, Vector3D *wo){
	return (black);//temp value
}

__device__
RGBColor F(BRDF *brdf, ShadeRec *sr, Vector3D *wi, Vector3D *wo){
	switch( brdf->type ){
	case BRDF_TYPE_LAMBERTIAN:
		return (LambertianF((Lambertian*)brdf,sr,wi,wo));
	case BRDF_TYPE_PERFECTSPECULAR:
		return (PerfectSpecularF((PerfectSpecular*)brdf,sr,wi,wo));
	case BRDF_TYPE_GLOSSYSPECULAR:
		return (GlossySpecularF((GlossySpecular*)brdf,sr,wi,wo));
	default:
		return black;
	}
}
 
__device__
RGBColor SampleF(BRDF *brdf, ShadeRec *sr, Vector3D *wi, Vector3D *wo){
	return black;
}

__device__
RGBColor Rho(BRDF *brdf, ShadeRec *sr, Vector3D *wo){
	switch( brdf->type ){
	case BRDF_TYPE_LAMBERTIAN:
		return (LambertianRho((Lambertian*)brdf,sr,wo));
	case BRDF_TYPE_PERFECTSPECULAR:
		return (PerfectSpecularRho((PerfectSpecular*)brdf,sr,wo));
	case BRDF_TYPE_GLOSSYSPECULAR:
		return (GlossySpecularRho((GlossySpecular*)brdf,sr,wo));
	default:
		return black;
	}
}