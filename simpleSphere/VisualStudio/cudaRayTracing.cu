#include "cudaRayTracing.cuh"

extern World *h_w;
extern World *d_w;

#include "Ray.cu"
#include "Tools.cu"
#include "Sampler.cu"
#include "ShadeRec.cu"
#include "Tracer.cu"
#include "Camara.cu"
#include "Light.cu"
#include "Material.cu"
#include "BRDF.cu"
#include "ViewPlane.cu"
#include "GeometricObject.cu"
#include "World.cu"

void cudaRayTracingInit(World **h_w,World **d_w,int width,int height){
	//cudaMalloc(&w,sizeof(World));
	BuildWorld(h_w,d_w,width,height);
}

void cudaRayTracing(World *w, int width, int height, uchar3 *buffer){
	RenderScene( w, width, height, buffer);
	cudaDeviceSynchronize();
}