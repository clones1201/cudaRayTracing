#include "defines.cuh"
#include "function_defines.cuh"

#include "stdlib.h"
#include "stdio.h"

void UpdateWorldToDevice(World *h_w, World **d_w){

	/*  The data copy to device should be implemented from ground up. 
	/*	a pointer which is contained by a pointer should be allocated first
	*/

	World *temp = (World*)malloc(sizeof(World));

	ViewPlane h_vp = *(h_w->vp);
	cudaMalloc(& h_vp.sampler ,sizeof( Sampler ));
	cudaCheckErrors("sampler allocate failed");
	cudaMemcpy( h_vp.sampler, h_w->vp->sampler, sizeof(Sampler) ,cudaMemcpyHostToDevice );
	cudaCheckErrors("sampler copy failed");

	cudaMalloc(&(temp->vp),sizeof(ViewPlane));
	cudaCheckErrors("viewplane allocate failed");
	cudaMemcpy((temp->vp),&h_vp,sizeof(ViewPlane),cudaMemcpyHostToDevice);
	cudaCheckErrors("viewplane copy failed");

	temp->backgroundColor = h_w->backgroundColor;

	/* Material table
	*/
	temp->numMaterial = h_w->numMaterial;
	temp->materials = (Material**)malloc( temp->numMaterial * sizeof( Material* )); 
	for( int i = 0 ; i < h_w->numMaterial ; i ++ ){

		int sizeOfMaterial;
		switch( h_w->materials[i]->type ){
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
		cudaMalloc(temp->materials + i,sizeOfMaterial);
		cudaCheckErrors("material allocate failed");
		cudaMemcpy(temp->materials[i],h_w->materials[i],sizeOfMaterial,cudaMemcpyHostToDevice);
		cudaCheckErrors("material copy failed");		
	}
	Material **d_ms;
	cudaMalloc(&d_ms,temp->numMaterial * sizeof( Material*));
	cudaCheckErrors("material* allocate failed ");
	cudaMemcpy(d_ms,temp->materials,temp->numMaterial * sizeof(Material*),cudaMemcpyHostToDevice);
	cudaCheckErrors("mateiral* copy failed");
	free( temp->materials );
	temp->materials = d_ms;

	/*  GeometricObject                   
	/*  struct GeometricObject has 2 types, sphere and plane, contains a index to material table 
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
		cudaMalloc(temp->objects + i,sizeOfObject);
		cudaCheckErrors("geometricobjects allocate failed");
		cudaMemcpy(temp->objects[i],h_w->objects[i],sizeOfObject,cudaMemcpyHostToDevice);
		cudaCheckErrors("geometricobjects copy failed");
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

	(*h_w)->vp->sampler = (Sampler*)malloc( sizeof(Sampler));
	(*h_w)->vp->sampler->count = 0;
	(*h_w)->vp->sampler->numSamples = 4;
	(*h_w)->vp->sampler->type = SAMPLER_JITTERED;
	GenerateSample( (*h_w)->vp->sampler );

	(*h_w)->backgroundColor = RGBColor(5,5,30);

	/* GeometricObject
	*/
	/*
	(*h_w)->numObject = 9;
	(*h_w)->objects = (GeometricObject **)malloc((*h_w)->numObject * sizeof(GeometricObject*));

	(*h_w)->numMaterial = 13;
	(*h_w)->materials = (Material**)malloc((*h_w)->numMaterial * sizeof(Material*));

	(*h_w)->materials[0] = (Material*)newMatte(0.25,0.65,red);   //Red Matte
	(*h_w)->materials[1] = (Material*)newMatte(0.25,0.65,green); //Green Matte
	(*h_w)->materials[2] = (Material*)newMatte(0.25,0.65,yellow);//Yellow Matte
	(*h_w)->materials[3] = (Material*)newMatte(0.25,0.65,blue);//Blue Matte
	(*h_w)->materials[4] = (Material*)newMatte(0.25,0.65,white);//White Matte
	(*h_w)->materials[5] = (Material*)newMatte(0.25,0.65, RGBColor(0.8,0.8,0.8)); //Gray Matte
	(*h_w)->materials[6] = (Material*)newPhong(0.25,0.6,green,white,0.2,5);	//Green Phong
	(*h_w)->materials[7] = (Material*)newPhong(0.25,0.6,red,white,0.2,5);	//Red Phong
	(*h_w)->materials[8] = (Material*)newPhong(0.25,0.6,yellow,white,0.2,5);//Yellow Phong
	(*h_w)->materials[9] = (Material*)newPhong(0.25,0.6,white,white,0.2,5);//White Phong
	(*h_w)->materials[10] = (Material*)newPhong(0.25,0.6, RGBColor(0.8,0.8,0.8),white,0.2,20); // Gray Phong
	(*h_w)->materials[11] = (Material*)newPhong(0.25,0.6,blue,white,0,1);//Blue Phong
	(*h_w)->materials[12] = (Material*)newReflective(0.25,0.5,RGBColor(1,1,1),RGBColor(1,1,1),0.15,100,0.75,white); //Mirror
	
	initSphere( ((Sphere**)((*h_w)->objects)),		Point3D(0,200,0),	200,			7	);
	initSphere( ((Sphere**)((*h_w)->objects+1)),	Point3D(-220,120,250),	120,		8	);
	initSphere( ((Sphere**)((*h_w)->objects+2)),	Point3D(200,60,80),	60,				6	);
	initPlane( ((Plane**)((*h_w)->objects+3)),	Point3D(0,0,0),	Normal(0,1,0),			4	);
	initPlane( ((Plane**)((*h_w)->objects+4)),	Point3D(-500,0,0),	Normal(1,0,0),		0	);
	initPlane( ((Plane**)((*h_w)->objects+5)),	Point3D(500,0,0),	Normal(-1,0,0),		12  );
	initPlane( ((Plane**)((*h_w)->objects+6)),	Point3D(0,0,-500),	Normal(0,0,1),		12	);
	initPlane( ((Plane**)((*h_w)->objects+7)),	Point3D(0,1000,0),	Normal(0,-1,0),		4	);
	initPlane( ((Plane**)((*h_w)->objects+8)),	Point3D(0,0,1000),	Normal(0,0,-1),		4	);
	*/
	
	(*h_w)->numObject = 101;
	(*h_w)->objects = (GeometricObject **)malloc((*h_w)->numObject * sizeof(GeometricObject*));

	(*h_w)->numMaterial = 3;
	(*h_w)->materials = (Material**)malloc((*h_w)->numMaterial * sizeof(Material*));

	(*h_w)->materials[0] = (Material*)newMatte(0.25,0.65,red);   //Red Matte
	(*h_w)->materials[1] = (Material*)newMatte(0.25,0.65, RGBColor(0.8,0.8,0.8));
	(*h_w)->materials[2] = (Material*)newPhong(0.25,0.6,red,white,0.2,5);	

	initPlane(  ((Plane**)((*h_w)->objects  )),	Point3D(0,0,-600),	Normal(0,0,1),		1);
	for( int i = 1 ; i < (*h_w)->numObject ; ++ i ){
		float x,y,z,r;
		x = 800 * float(rand())/float(RAND_MAX) - 200 ;
		y = 800 * float(rand())/float(RAND_MAX) - 200 ;
		z = 400 * float(rand())/float(RAND_MAX) - 100 ;
		r = 50 * float(rand())/float(RAND_MAX) + 25;
		initSphere( ((Sphere**)((*h_w)->objects + i )),	Point3D(x,y,z) ,r , 2); 
	}
	//initSphere( ((Sphere**)((*h_w)->objects+0)),	Point3D(60,0,80),	60,				0	);
	//initSphere( ((Sphere**)((*h_w)->objects+1)),	Point3D(-60,0,80),	60,				1	);
	//initSphere( ((Sphere**)((*h_w)->objects+2)),	Point3D(0,104,80),	60,				2	);
	
	(*h_w)->kdTree = BuildTree((*h_w)->objects + 1,(*h_w)->numObject - 1);

	Pinhole *pinhole = (Pinhole*)malloc(sizeof(Pinhole));
	pinhole->type = CAMARA_TYPE_PINHOLE;
	pinhole->eye = Point3D(-350,600,900);
	pinhole->lookat = Point3D(200,100,0);
	pinhole->up = Vector3D(0,1,0);
	pinhole->viewDistance = 400;
	pinhole->zoom = 1;
	ComputeUVW( (Camara*)pinhole );
	(*h_w)->camara = (Camara*)pinhole;

	/* Light
	*/
	Ambient *h_ab = (Ambient*)malloc(sizeof(Ambient));
	h_ab->ls = 0.2;
	h_ab->color = white;
	h_ab->shadows = false;
	h_ab->type = LIGHT_TYPE_AMBIENT;
	(*h_w)->ambient = (Light*)h_ab;

	(*h_w)->numLight = 1;

	(*h_w)->lights = (Light**)malloc( (*h_w)->numLight * sizeof(Light*) );
	
	PointLight *h_pl = (PointLight*)malloc(sizeof(PointLight));
	h_pl->ls = 1.1;
	h_pl->pos = Point3D(0,700,450);
	h_pl->color = white;
	h_pl->shadows = true;
	h_pl->type = LIGHT_TYPE_POINTLIGHT;
	(*h_w)->lights[0] = (Light*)h_pl;
/*
	AmbientOccluder *h_ao = (AmbientOccluder*)malloc(sizeof(AmbientOccluder));
	h_ao->ls = 1;
	h_ao->color = white;
	h_ao->shadows = true;
	h_ao->type = LIGHT_TYPE_AMBIENTOCCLUDER;
	(*h_w)->lights[0] = (Light*)h_ao;
	*/
	
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
			sr->material = w->materials[ w->objects[i]->materialIdx ];
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