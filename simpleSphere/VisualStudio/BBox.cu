#include "defines.cuh"
#include "function_defines.cuh"

__host__
BBox SphereBound(Sphere *sphere){
	BBox result;
	result.pMax = Point3D( sphere->center.x + sphere->radius,sphere->center.y + sphere->radius,sphere->center.z + sphere->radius  );
	result.pMin = Point3D( sphere->center.x - sphere->radius,sphere->center.y - sphere->radius,sphere->center.z - sphere->radius  );
	return result;
}

__host__ /* build new box to overlay the object */
BBox Bounds(GeometricObject *object){
	switch( object->type ){
	case GMO_TYPE_SPHERE:
		return SphereBound( (Sphere*)object);
	default:
		return BBox();
	}
}
__host__
float SurfaceArea(BBox box){
	float result;
	float x = box.pMax.x - box.pMin.x;
	float y = box.pMax.y - box.pMin.y;
	float z = box.pMax.z - box.pMin.z;
	result = 2 * ( x * y + y * z + x * z );
	return result;
}

__host__ /* Union , modify a box to overlay a point */
BBox Union(BBox box, Point3D p){
	BBox result;

	result.pMax.x = MAX(box.pMax.x,p.x);
	result.pMax.y = MAX(box.pMax.y,p.y);
	result.pMax.z = MAX(box.pMax.z,p.z);

	
	result.pMin.x = MIN(box.pMin.x,p.x);
	result.pMin.y = MIN(box.pMin.y,p.y);
	result.pMin.z = MIN(box.pMin.z,p.z);
	
	return result;
}

__host__ /* Union , modify a box to overlay another box */
BBox Union(BBox box1, BBox box2){
	BBox result;

	result.pMax.x = MAX(box1.pMax.x,box1.pMax.x);
	result.pMax.y = MAX(box1.pMax.y,box1.pMax.y);
	result.pMax.z = MAX(box1.pMax.z,box1.pMax.z);
		
	result.pMin.x = MIN(box1.pMin.x,box1.pMin.x);
	result.pMin.y = MIN(box1.pMin.y,box1.pMin.y);
	result.pMin.z = MIN(box1.pMin.z,box1.pMin.z);
	
	return result;
}

__device__
bool HitBox(BBox box, Ray ray, float *tmin, float *tmax){
	
	float ox = ray.o.x;float oy = ray.o.y; float oz = ray.o.z;
	float dx = ray.d.x;float dy = ray.d.y; float dz = ray.d.z;

	float tx_min,ty_min,tz_min;
	float tx_max,ty_max,tz_max;

	float a = 1.0 / dx;
	if( a >= 0 ){
		tx_min = ( box.pMin.x - ox ) * a;
		tx_max = ( box.pMax.x - ox ) * a;
	}
	else{
		tx_min = ( box.pMax.x - ox ) * a;
		tx_max = ( box.pMin.x - ox ) * a;
	}

	float b = 1.0 / dx;
	if( b >= 0 ){
		ty_min = ( box.pMin.y - oy ) * b;
		ty_max = ( box.pMax.y - oy ) * b;
	}
	else{
		ty_min = ( box.pMax.y - oy ) * b;
		ty_max = ( box.pMin.y - oy ) * b;
	}
	
	float c = 1.0 / dx;
	if( c >= 0 ){
		tz_min = ( box.pMin.z - oz ) * c;
		tz_max = ( box.pMax.z - oz ) * c;
	}
	else{
		tz_min = ( box.pMax.z - oz ) * c;
		tz_max = ( box.pMin.z - oz ) * c;
	}

	//find largest entering t value
	if( tx_min > ty_min ){
		*tmin = tx_min;
	}else{
		*tmin = ty_min;
	}
	if( tz_min > *tmin )
		*tmin = tz_min;
	//find smallest exiting t value
	if( tx_max < ty_max ){
		*tmax = tx_max;
	}else{
		*tmax = ty_max;
	}
	if( tz_max < *tmax )
		*tmax = tz_max;

	return ( *tmin < *tmax && *tmax > 0.001 );
}