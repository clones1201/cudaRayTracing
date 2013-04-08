#include "defines.cuh"
#include "function_defines.cuh"

__device__ inline
Vector3D AmbientGetDirection(Ambient *ab, ShadeRec *sr){
	return (Vector3D(0,0,0));
}

__device__ inline
RGBColor AmbientL(Ambient *ab, ShadeRec *sr){
	return (ab->color * ab->ls);
}

__device__ inline
Vector3D DirectionalGetDirection(Directional *dnl, ShadeRec *sr){
	return (dnl->dir);
}

__device__ inline
RGBColor DirectionalL(Directional *dnl, ShadeRec *sr){
	return (dnl->color * dnl->ls);
}

__device__ inline
Vector3D PointlightGetDirection(PointLight *pl, ShadeRec *sr){
	return Normalize( pl->pos - sr->hitPoint );
}

__device__ inline
RGBColor PointlightL(PointLight *pl, ShadeRec *sr){
	return (pl->color * pl->ls);
}

__device__ 
Vector3D GetDirection(Light *l, ShadeRec *sr){
	switch( l->type ){
	case LIGHT_TYPE_AMBIENT:
		return (AmbientGetDirection((Ambient*)l,sr));
	case LIGHT_TYPE_DIRECTIONAL:
		return (DirectionalGetDirection((Directional*)l,sr));
	case LIGHT_TYPE_POINTLIGHT:
		return (PointlightGetDirection((PointLight*)l,sr));
	default:
		return Vector3D(0,0,0);
	}
}

__device__
RGBColor L(Light *l, ShadeRec *sr){
	switch( l->type ){
	case LIGHT_TYPE_AMBIENT:
		return (AmbientL((Ambient*)l,sr));
	case LIGHT_TYPE_DIRECTIONAL:
		return (DirectionalL((Directional*)l,sr));
	case LIGHT_TYPE_POINTLIGHT:
		return (PointlightL((PointLight*)l,sr));
	default:
		return (black);
	}
}

__device__ inline
bool DirectionalInShadow(Directional * dir,Ray ray, ShadeRec *sr){
	return false; //temp value;
}

__device__ inline
bool PointlightInShadow(PointLight* pl,Ray ray , ShadeRec *sr){
	float t;
	float d = Distance( pl->pos , ray.o );

	for( int i = 0 ; i < sr->w->numObject ; i++ ){
		if( ShadowHit( sr->w->objects[i],ray,&t) && (t < d) ){
			return true;
		}
	}
	return false;
}

__device__
bool inShadow(Light *l,Ray ray ,ShadeRec *sr){
	switch( l->type ){
	case LIGHT_TYPE_AMBIENT:
		return false;
	case LIGHT_TYPE_DIRECTIONAL:
		return (DirectionalInShadow((Directional*)l,ray,sr));
	case LIGHT_TYPE_POINTLIGHT:
		return (PointlightInShadow((PointLight*)l,ray,sr));
	default:
		return (false);
	}
}