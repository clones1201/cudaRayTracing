#if !defined __FUNCTION_DEFINES_CUH__
#define __FUNCTION_DEFINES_CUH__

#include "defines.cuh"

/*Tracer function */

extern __device__  
void singleSphereTraceRay(World *w,Sphere *s,Ray ray,RGBColor *color);

extern __device__
RGBColor multiObjTraceRay(World *w, Ray ray);

extern __device__
RGBColor RayCastTraceRay(World *w, Ray ray ,int depth);

/* Geometric Object function */
extern __device__
bool Hit(GeometricObject *obj,Ray ray , float *tmin, ShadeRec *sr);

extern __device__
bool ShadowHit(GeometricObject *obj,Ray ray, float *tmin);

extern __host__
void initPlane(Plane **pl, Point3D p,Normal n,Material *m);

extern __host__
void initSphere(Sphere **s, Point3D c, float r,Material *m);

extern __host__
void freePlane(Plane **pl);

extern __host__
void freeSphere(Sphere **s);

/* Sampler function */

extern __device__ inline
Point2D	getSampleUnitSquare(SamplerType type, int idx,SampleScale scale);

extern __device__ __host__ inline
int getSampleNum(SampleScale scale);

extern __device__ __host__ inline
SampleScale getSampleScale(int num);


/* World function */
extern void BuildWorld(World **h_w, World **d_w, int width,int height);

extern void HitBareBonesObject(World *w, Ray ray,ShadeRec *sr);

extern void HitObject(World *w,Ray ray,ShadeRec *sr);

/* camara function */
__device__ __host__ inline
void ComputeUVW(Camara *c){
	c->w = c->eye - c->lookat;
	c->w = Normalize( c->w );
	c->u = CrossProduct( c->up , c->w );
	c->u = Normalize( c->u );
	c->v = CrossProduct( c->w , c->u );
}

extern void RenderScene(World *w, int width,int height, RGBColor *buffer);

/* BRDF function */
extern __device__ 
RGBColor F(BRDF *brdf, ShadeRec *sr, Vector3D *wi, Vector3D *wo);

extern __device__
RGBColor SampleF(BRDF *brdf, ShadeRec *sr, Vector3D *wi, Vector3D *wo);

extern __device__
RGBColor Rho(BRDF *brdf, ShadeRec *sr, Vector3D *wo);

/* Light function */

extern __device__ 
Vector3D GetDirection(Light *l, ShadeRec *sr);

extern __device__
RGBColor L(Light *l, ShadeRec *sr);

extern __device__
bool inShadow(Light *l,Ray ray, ShadeRec *sr);


/* Material function */
__device__ __host__ inline
Matte* newMatte(float ka,float kd,RGBColor cd){
	Matte *result = (Matte*)malloc(sizeof(Matte));
	result->ambientBRDF.type = BRDF_TYPE_LAMBERTIAN;
	result->diffuseBRDF.type = BRDF_TYPE_LAMBERTIAN;

	result->ambientBRDF.kd = ka;
	result->diffuseBRDF.kd = kd;
	result->ambientBRDF.cd = cd;
	result->diffuseBRDF.cd = cd;
	result->type = MATERIAL_TYPE_MATTE;
	return result;
}

extern __device__
RGBColor Shade(Material *m,ShadeRec *sr);

#endif
