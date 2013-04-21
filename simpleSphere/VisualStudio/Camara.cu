#include "defines.cuh"
#include "function_defines.cuh"

__device__ inline
Vector3D RayDirection(Pinhole *ph,Point2D p){
	Vector3D result = p.x * ph->u + p.y * ph->v - ph->w * ph->viewDistance;
	result = Normalize(result);
	return result;
}

__global__ 
void PinholeRenderScene_k(World *w, uchar3 *buffer){
	
	Ray ray;
	ViewPlane vp = *(w->vp);
	Pinhole pinhole = *((Pinhole*)w->camara);
	//int depth = 0;
	
	Point2D sp;		//sample point in [0,1] x [0,1]
	Point2D pp;		//sample point in a pixel

	vp.s = vp.s / pinhole.zoom;
	ray.o = pinhole.eye;

	int r = blockIdx.x * blockDim.x + threadIdx.x;
	int c = blockIdx.y * blockDim.y + threadIdx.y;
	
	int offset = r * gridDim.x * blockDim.x + c;

	int numSample = getSampleNum(  vp.sampleScale  );

	RGBColor pixelColor = black;
	buffer[offset] = make_uchar3(0,0,0);
	for(int i = 0 ; i < numSample ; ++ i ){
			sp = getSampleUnitSquare( vp.samplerType , i , vp.sampleScale );

			//pp.x = w->vp->s * ( c - 0.5 * w->vp->hres + sp.x );
			//pp.y = w->vp->s * ( r - 0.5 * w->vp->vres + sp.y );
			pp.x = vp.s * ( c - 0.5 * vp.hres + sp.x );
			pp.y = vp.s * ( r - 0.5 * vp.vres + sp.y );

			ray.d = RayDirection(&pinhole,pp);

			pixelColor = pixelColor + RayCastTraceRay(w,ray,0) / numSample;
			//singleSphereTraceRay(w,(Sphere*)*(w->object),ray,&pixelColor);
			//pixelColor =multiObjTraceRay(w,ray);
		}
	buffer[offset] = MapToUchar(pixelColor);

}
/*
__global__
	void render_scene_k(World *w,uchar3 *buffer){

		RGBColor pixelColor = red;
		
		Ray ray;
				
		int r = blockIdx.x * blockDim.x + threadIdx.x;
		int c = blockIdx.y * blockDim.y + threadIdx.y;
		
		int offset = r * gridDim.x * blockDim.x + c;

		buffer[offset] = w->backgroundColor;

		Point2D sp;		//sample point in [0,1] x [0,1]
		Point2D pp;		//sample point in a pixel

		//SamplerType type = SAMPLER_JITTERED;		
		//SampleScale scale = SAMPLE_SCALE_16;
		

		int numSample = getSampleNum(  w->vp->sampleScale  );

		for(int i = 0 ; i < numSample ; ++ i ){
			sp = getSampleUnitSquare( w->vp->samplerType , i , w->vp->sampleScale );

			//pp.x = w->vp->s * ( c - 0.5 * w->vp->hres + sp.x );
			//pp.y = w->vp->s * ( r - 0.5 * w->vp->vres + sp.y );
			pp.x = 1 * ( c - 0.5 * 512 + sp.x );
			pp.y = 1 * ( r - 0.5 * 512 + sp.y );

			ray.o = Point3D(pp.x,pp.y,0);
			ray.d = Vector3D(0,0,1);

			//Sphere*s;
			pixelColor = multiObjTraceRay(w,ray);
			//singleSphereTraceRay(w,(Sphere*)*(w->object),ray,&pixelColor);

			buffer[offset] = buffer[offset] + pixelColor / numSample;
			//buffer[offset] = w->backgroundColor;
		}
}
*/

void RenderScene(World *w,int width,int height,uchar3 *buffer){
	
	dim3 blockPerGrid(width/16,height/16);
	dim3 threadPerBlock(16,16);

	World h_world;
	cudaMemcpy(&h_world, w,sizeof(World),cudaMemcpyDeviceToHost);
	cudaCheckErrors("world copy failed");

	Camara h_camara,*pointer;
	cudaMemcpy( &h_camara,h_world.camara,sizeof(Camara),cudaMemcpyDeviceToHost);
	cudaCheckErrors("camara copy failed");

	switch( h_camara.type ){
	case CAMARA_TYPE_PINHOLE:
		PinholeRenderScene_k<<<blockPerGrid,threadPerBlock>>>(w,buffer);
		break;
	default:
		break;
	}
/* old version  *//* 
	render_scene_k<<<blockPerGrid,threadPerBlock>>>(w,buffer);
	cudaCheckErrors("render_scene_k failed...");
	*/
}
