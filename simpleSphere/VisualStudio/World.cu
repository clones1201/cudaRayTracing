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
	h_world->numObject = 4;
	GeometricObject **obj = (GeometricObject **)malloc(h_world->numObject * sizeof(GeometricObject*));

	initSphere( ((Sphere**)(obj)),		Point3D(0,120,280),	120,		red);
	initSphere( ((Sphere**)(obj+1)),	Point3D(0,150,0),	150,		green);
	initSphere( ((Sphere**)(obj+2)),	Point3D(210,100,100),	100,		yellow);
	initPlane( ((Plane**)(obj+3)),	Point3D(0,0,0),	Normal(0,1,0),		blue);

	cudaMalloc((void**)&(h_world->object),h_world->numObject  * sizeof(GeometricObject*) );
	cudaCheckErrors("object pointer memory allocate failed");
	
	cudaMemcpy(h_world->object,obj,h_world->numObject * sizeof(GeometricObject*),cudaMemcpyHostToDevice);
	cudaCheckErrors("object pointer memory copy failed");

	/* camara */
	Pinhole pinhole;
	pinhole.type = CAMARA_TYPE_PINHOLE;
	pinhole.eye = Point3D(300,400,500);
	pinhole.lookat = Point3D(0,100,0);
	pinhole.up = Vector3D(0,1,0);
	pinhole.viewDistance = 50;
	pinhole.zoom = 5;
	ComputeUVW((Camara*)&pinhole);
	cudaMalloc((void**)&(h_world->camara),sizeof(Pinhole));
	cudaCheckErrors("pinhole memory allocate failed");
	cudaMemcpy((h_world->camara),&pinhole,sizeof(Pinhole),cudaMemcpyHostToDevice);
	cudaCheckErrors("pinhole copy failed");

	/*  world */
	cudaMalloc((void**) w , sizeof(World));
	cudaCheckErrors( "world allocate failed" );
	cudaMemcpy( *w , h_world,sizeof(World),cudaMemcpyHostToDevice);
	cudaCheckErrors("world memory copy failed");

	free(h_world);free(h_vp);
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