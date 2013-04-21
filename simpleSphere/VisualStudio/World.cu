#include "defines.cuh"
#include "function_defines.cuh"

#include "stdlib.h"
#include "stdio.h"

void UpdateWorldToDevice(World *h_w, World **d_w){

	/*  The data copy to device should be implemented from ground up. 
	/*	a pointer which is contained by a pointer should be allocated first
	*/

	World *temp = (World*)malloc(sizeof(World));

	cudaMalloc(&(temp->vp),sizeof(ViewPlane));
	cudaCheckErrors("viewplane allocate failed");
	cudaMemcpy((temp->vp),h_w->vp,sizeof(ViewPlane),cudaMemcpyHostToDevice);
	cudaCheckErrors("viewplane copy failed");

	temp->backgroundColor = h_w->backgroundColor;

	/*  GeometricObject                   
	/*  struct GeometricObject has 2 types, sphere and plane, contains a pointer to material  
	/*  if any change, this comment should be updated;
	*/
	temp->numObject = h_w->numObject;

	temp->objects = (GeometricObject**)malloc( temp->numObject * sizeof(GeometricObject*));
	for(int i = 0 ; i < temp->numObject ; ++i ){
		int sizeOfObject;
		switch( h_w->objects[i]->type ){
		case GMO_TYPE_SPHERE:
			sizeOfObject = sizeof(Sphere);
			break;
		case GMO_TYPE_PLANE:
			sizeOfObject = sizeof(Plane);
			break;
		default:
			sizeOfObject = sizeof(GeometricObject);
			break;
		}
		GeometricObject *h_gmo = (GeometricObject*)malloc(sizeOfObject);
		memcpy(h_gmo,h_w->objects[i],sizeOfObject);

		int sizeOfMaterial;
		switch( h_w->objects[i]->material->type ){
		case MATERIAL_TYPE_MATTE:
			sizeOfMaterial = sizeof(Matte);
			break;
		case MATERIAL_TYPE_PHONG:
			sizeOfMaterial = sizeof(Phong);
			break;
		default:
			sizeOfMaterial = sizeof(Material);
			break;
		}
		Material *d_m;
		cudaMalloc(&d_m,sizeOfMaterial);
		cudaCheckErrors("material allocate failed");
		cudaMemcpy(d_m,h_w->objects[i]->material,sizeOfMaterial,cudaMemcpyHostToDevice);
		cudaCheckErrors("material copy failed");

		h_gmo->material = d_m;

		cudaMalloc(temp->objects + i,sizeOfObject);
		cudaCheckErrors("geometricobjects allocate failed");
		cudaMemcpy(temp->objects[i],h_gmo,sizeOfObject,cudaMemcpyHostToDevice);
		cudaCheckErrors("geometricobjects copy failed");

		free(h_gmo);
	}
	GeometricObject **d_obj;
	cudaMalloc(&d_obj,temp->numObject * sizeof(GeometricObject*));
	cudaCheckErrors("geometricobject* allocate failed");
	cudaMemcpy(d_obj,temp->objects,temp->numObject * sizeof(GeometricObject*),cudaMemcpyHostToDevice);
	cudaCheckErrors("geometricobject* copy failed");

	free(temp->objects);
	temp->objects = d_obj;

	/* Camara 
	*/
	int sizeOfCamara;
	switch( h_w->camara->type ){
	case CAMARA_TYPE_PINHOLE:
		sizeOfCamara = sizeof(Pinhole);
		break;
	default:
		sizeOfCamara = sizeof(Camara);
		break;
	}
	cudaMalloc(&(temp->camara),sizeOfCamara);
	cudaCheckErrors("camara allocate failed");
	cudaMemcpy(temp->camara, h_w->camara,sizeOfCamara,cudaMemcpyHostToDevice);
	cudaCheckErrors("camara copy allocate failed");

	/* Light 
	*/
	cudaMalloc(&(temp->ambient),sizeof(Ambient));
	cudaCheckErrors("ambient allocate failed");
	cudaMemcpy(temp->ambient,h_w->ambient, sizeof(Ambient),cudaMemcpyHostToDevice);
	cudaCheckErrors("ambient copy failed");

	temp->numLight = h_w->numLight;

	temp->lights = (Light**)malloc(temp->numLight * sizeof(Light*));
	for( int i = 0 ; i < temp->numLight ; ++i ){
		int sizeOfLight;
		switch( h_w->lights[i]->type ){
		case LIGHT_TYPE_AMBIENT:
			sizeOfLight = sizeof(Ambient);
			break;
		case LIGHT_TYPE_POINTLIGHT:
			sizeOfLight = sizeof(PointLight);
			break;
		case LIGHT_TYPE_DIRECTIONAL:
			sizeOfLight = sizeof(Directional);
			break;
		default:
			sizeOfLight = sizeof(Light);
			break;
		}
		cudaMalloc( temp->lights + i , sizeOfLight );
		cudaCheckErrors(" lights allocate failed ");
		cudaMemcpy( temp->lights[i] , h_w->lights[i] , sizeOfLight , cudaMemcpyHostToDevice);
		cudaCheckErrors(" lights copy failed ");
	}
	Light **lts;
	cudaMalloc( &lts ,temp->numLight * sizeof(Light*) );
	cudaCheckErrors("light* allocate failed");
	cudaMemcpy(lts , temp->lights , temp->numLight * sizeof(Light*),cudaMemcpyHostToDevice);

	free(temp->lights);
	temp->lights = lts;

	cudaMalloc(d_w,sizeof(World));
	cudaCheckErrors("world allocate failed");
	cudaMemcpy(*d_w,temp,sizeof(World),cudaMemcpyHostToDevice);
	cudaCheckErrors("world copy failed");

	free(temp);
}

void BuildWorld(World **h_w, World **d_w, int width,int height){
	
	(*h_w) = (World*)malloc(sizeof(World));
	(*h_w)->vp = (ViewPlane*)malloc(sizeof(ViewPlane));
		
	(*h_w)->vp->hres = width;
	(*h_w)->vp->vres = height;
	(*h_w)->vp->s	 = 1;
	(*h_w)->vp->samplerType = SAMPLER_JITTERED;
	(*h_w)->vp->sampleScale = SAMPLE_SCALE_4;

	(*h_w)->backgroundColor = RGBColor(5,5,30);

	/* GeometricObject
	*/
	(*h_w)->numObject = 5;
	(*h_w)->objects = (GeometricObject **)malloc((*h_w)->numObject * sizeof(GeometricObject*));

	Matte *material1 = newMatte(0.25,0.65,red);	
	Phong *material2 = newPhong(0.25,0.6,green,0.2,20);	
	Phong *material3 = newPhong(0.25,0.6,yellow,0.2,20);	
	Matte *material4 = newMatte(0.25,0.65,white);
	Phong *material5 = newPhong(0.25,0.6,red,0.2,20);

	initSphere( ((Sphere**)((*h_w)->objects)),		Point3D(0,120,280),	120,	(Material*)material5		);
	initSphere( ((Sphere**)((*h_w)->objects+1)),	Point3D(0,150,0),	150,		(Material*)material2		);
	initSphere( ((Sphere**)((*h_w)->objects+2)),	Point3D(210,100,100),	100,		(Material*)material3	);
	initPlane( ((Plane**)((*h_w)->objects+3)),	Point3D(0,0,0),	Normal(0,1,0),		(Material*)material4		);
	initPlane( ((Plane**)((*h_w)->objects+4)),	Point3D(-600,0,-600),	Normal(1,0,1),		(Material*)material4);

/*	(*h_w)->numObject = 100;
	(*h_w)->objects = (GeometricObject **)malloc((*h_w)->numObject * sizeof(GeometricObject*));
	Matte *material = newMatte(0.25,0.65,red);
	Matte *material4 = newMatte(0.25,0.65,RGBColor(5,5,40));

	initPlane(  ((Plane**)((*h_w)->objects )),	Point3D(-600,0,-600),	Normal(1,0,1),		(Material*)material4		);
	for( int i = 1 ; i < (*h_w)->numObject ; ++ i ){
		initSphere( ((Sphere**)((*h_w)->objects + i )), 
			Point3D( 400 * float(rand())/float(RAND_MAX) , 400 * float(rand())/float(RAND_MAX) , 400 * float(rand())/float(RAND_MAX) ),
			35 * float(rand())/float(RAND_MAX) + 5 , (Material*)material); 
	}*/
	
	Pinhole *pinhole = (Pinhole*)malloc(sizeof(Pinhole));
	pinhole->type = CAMARA_TYPE_PINHOLE;
	pinhole->eye = Point3D(500,300,300);
	pinhole->lookat = Point3D(0,100,100);
	pinhole->up = Vector3D(0,1,0);
	pinhole->viewDistance = 400;
	pinhole->zoom = 1;
	ComputeUVW( (Camara*)pinhole );
	(*h_w)->camara = (Camara*)pinhole;

	/* Light
	*/
	Ambient *h_ab = (Ambient*)malloc(sizeof(Ambient));
	h_ab->ls = 0.5;
	h_ab->color = white;
	h_ab->shadows = false;
	h_ab->type = LIGHT_TYPE_AMBIENT;
	(*h_w)->ambient = (Light*)h_ab;

	(*h_w)->numLight = 1;

	(*h_w)->lights = (Light**)malloc( (*h_w)->numLight * sizeof(Light*) );
	
	PointLight *h_pl = (PointLight*)malloc(sizeof(PointLight));
	h_pl->ls = 1;
	h_pl->pos = Point3D(1000,400,1200);
	h_pl->color = white;
	h_pl->shadows = true;
	h_pl->type = LIGHT_TYPE_POINTLIGHT;
	(*h_w)->lights[0] = (Light*)h_pl;
	
	UpdateWorldToDevice(*h_w,d_w);
}

void FreeWorld(World *d_w,World *h_w){


}


__device__ 
void  HitBareBonesObject(World *w, Ray ray,ShadeRec *sr){
	float t;
	float tmin = hugeValue;
		
	sr->hitAnObject = false;

	for(int i = 0 ; i < w->numObject ; ++i){
		if( Hit(w->objects[i],ray,&t,sr) && (t < tmin) ){
			sr->hitAnObject= true;
			tmin = t;
			sr->color = w->objects[i]->color;
		}
	}
}

__device__
void HitObject(World *w, Ray ray, ShadeRec *sr ){
	float t ;
	Normal normal;
	Point3D localHitPoint;
	float tmin = hugeValue;

	for(int i = 0 ; i < w->numObject ; ++i){
		if( Hit( w->objects[i],ray,&t,sr) && (t < tmin )){
			sr->hitAnObject = true;
			tmin = t;
			sr->material = w->objects[i]->material;
			sr->hitPoint = ray.o + t * ray.d;
			normal = sr->normal;
			localHitPoint = sr->localHitPoint;
		}
	}

	if( sr->hitAnObject ){
		//sr->t = tmin;    //ShadeRec does not contain a 't'
		sr->normal = normal;
		sr->localHitPoint = localHitPoint;
	}
}