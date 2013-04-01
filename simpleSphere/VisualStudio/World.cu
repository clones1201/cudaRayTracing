#include "defines.cuh"

#include "stdlib.h"
#include "stdio.h"

void build_world(World **w,int width,int height){
	
	World *h_world;
	h_world = (World*)malloc(sizeof(World));
		
	ViewPlane *h_vp = (ViewPlane*)malloc(sizeof(ViewPlane));
	h_vp->hres = width;
	h_vp->vres = height;
	h_vp->s = 1;	
	h_vp->samplerType = SAMPLER_JITTERED;
	h_vp->sampleScale = SAMPLE_SCALE_4;
	cudaMalloc((void**)&(h_world->vp),sizeof(ViewPlane));
	cudaCheckErrors("viewplane allocate failed");
	cudaMemcpy(h_world->vp,h_vp,sizeof(ViewPlane),cudaMemcpyHostToDevice);
	cudaCheckErrors("viewplane memory copy failed");

	h_world->backgroundColor = black;
	
	/********  Geometric Object  ***********/
	h_world->numObject = 3;
	GeometricObject **obj = (GeometricObject **)malloc(3 * sizeof(GeometricObject*));

	initSphere( ((Sphere**)(obj)),		Point3D(0,0,200),	180,		red);
	initSphere( ((Sphere**)(obj+1)),	Point3D(0,50,200),	150,		green);
	initPlane( ((Plane**)(obj+2)),	Point3D(0,0,200),	Normal(0,1,-1),blue);

	cudaMalloc((void**)&(h_world->object),3 * sizeof(GeometricObject*) );
	cudaCheckErrors("object pointer memory allocate failed");
	
	cudaMemcpy(h_world->object,obj,3 * sizeof(GeometricObject*),cudaMemcpyHostToDevice);
	cudaCheckErrors("object pointer memory copy failed");

	cudaMalloc((void**) w , sizeof(World));
	cudaCheckErrors( "world allocate failed" );
	cudaMemcpy( *w , h_world,sizeof(World),cudaMemcpyHostToDevice);
	cudaCheckErrors("world memory copy failed");

	free(h_world);free(h_vp);
}

__global__
	void render_scene_k(World *w,RGBAColor *buffer){

		RGBAColor pixelColor = red;
		
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
			//multiObjTraceRay(w,ray,&pixelColor);
			singleSphereTraceRay(w,(Sphere*)*(w->object),ray,&pixelColor);

			buffer[offset] = buffer[offset] + pixelColor / numSample;
			//buffer[offset] = w->backgroundColor;
		}
}

void render_scene(World *w,int width,int height,RGBAColor *buffer){
	
	dim3 blockPerGrid(width/16,height/16);
	dim3 threadPerBlock(16,16);
	render_scene_k<<<blockPerGrid,threadPerBlock>>>(w,buffer);
	cudaCheckErrors("render_scene_k failed...");

	cudaDeviceSynchronize();
}

__device__ 
void  hitBareBonesObject(World *w, Ray ray,ShadeRec *sr){
	float t;
	float tmin = hugeValue;
		
	sr->hitAnObject = false;

	for(int i = 0 ; i < w->numObject ; ++i){
		switch( ((GeometricObject*)(w->object[i]))->type ){
		case GMO_TYPE_SPHERE:
			if( sphereHit((Sphere*)w->object[i],ray,&t,sr) && (t < tmin) ){
				sr->hitAnObject= true;
				tmin = t;
				sr->color = w->object[i]->color;
			}
			break;
		case GMO_TYPE_PLANE:
			if( planeHit((Plane*)w->object[i],ray,&t,sr) && (t < tmin) ){
				sr->hitAnObject= true;
				tmin = t;
				sr->color = w->object[i]->color;
			}
			break;
		default:
			sr->color = w->backgroundColor;
			break;
		}
	}
}
