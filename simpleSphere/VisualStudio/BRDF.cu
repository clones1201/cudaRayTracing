#include "defines.cuh"
#include "function_defines.cuh"

__device__ inline
RGBColor LambertianF(Lambertian* lbt,ShadeRec *sr, Vector3D *wi, Vector3D *wo){
	return (lbt->cd * lbt->kd);
}

__device__ inline
RGBColor LambertianSampleF(Lambertian *lbt, ShadeRec *sr, Vector3D *wi, Vector3D *wo,float *pdf){
	Vector3D w = sr->normal;
	Vector3D v = CrossProduct( Vector3D(0.034,0,0.0071),w);
	Normalize(v);
	Vector3D u = CrossProduct( v,w );

	Point3D sp = getSampleUnitHemiSphere( sr->w->vp->sampler, 1);
	*wi = sp.x * u + sp.y * v + sp.z * w;
	Normalize(*wi);
	*pdf = sr->normal * *wi * invPI;
	return ( lbt->cd * (lbt->kd * invPI)) ;  
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
RGBColor PerfectSpecularSampleF(PerfectSpecular *ps, ShadeRec *sr, Vector3D *wi, Vector3D *wo,float *pdf){
	float ndotwo = sr->normal * (*wo);
	*wi = ( Vector3D(0,0,0) - *wo ) + sr->normal * 2.0 * ndotwo;
	return ( ps->kr * ps->cr / ( sr->normal * *wi) );
}
__device__ inline 
RGBColor PerfectSpecularRho(PerfectSpecular *ps, ShadeRec *sr, Vector3D *wo){
	return (black);//temp value
}

__device__ inline
RGBColor GlossySpecularF(GlossySpecular *gs, ShadeRec *sr, Vector3D *wi, Vector3D *wo){
	RGBColor result;
	float ndotwi = sr->normal * (*wi);
	Vector3D r = ( Vector3D(0,0,0) - *wi ) + sr->normal * 2.0 * ndotwi;
	float rdotwo = r * (*wo);
	
	if( rdotwo > 0.001 ){
		float l = gs->ks * powf(rdotwo,gs->exp);
		result = RGBColor(l,l,l);
	}
	return (result);
}

__device__ inline
RGBColor GlossySpecularSampleF(GlossySpecular *gs, ShadeRec *sr,Vector3D *wi, Vector3D *wo,float *pdf){
	float ndotwo = sr->normal * (*wo);
	Vector3D r =  Vector3D(0,0,0) - *wo + sr->normal * 2.0 *  ndotwo;

	Vector3D w = r;
	Vector3D u = CrossProduct(Vector3D(0.00424,1,0.00764 ),w);
	Normalize(u);
	Vector3D v = CrossProduct( u,w);

	Point3D sp = getSampleUnitHemiSphere( sr->w->vp->sampler , gs->exp);
	*wi = sp.x * u + sp.y * v + sp.z * w;

	if( sr->normal * (*wi) < 0.0 ){
		*wi = -sp.x * u - sp.y * v + sp.z * w;
	}

	float phongLobe = powf( r * (*wi) , gs->exp );
	*pdf = phongLobe  * ( sr->normal * (*wi));

	return (gs->ks * gs->cs * phongLobe);
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
RGBColor SampleF(BRDF *brdf, ShadeRec *sr, Vector3D *wi, Vector3D *wo, float *pdf){
	switch( brdf->type ){
	case BRDF_TYPE_LAMBERTIAN:
		return (LambertianSampleF((Lambertian*)brdf,sr,wi,wo,pdf));
	case BRDF_TYPE_PERFECTSPECULAR:
		return (PerfectSpecularSampleF((PerfectSpecular*)brdf,sr,wi,wo,pdf));
	case BRDF_TYPE_GLOSSYSPECULAR:
		return (GlossySpecularSampleF((GlossySpecular*)brdf,sr,wi,wo,pdf));
	default:
		return black;
	}
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