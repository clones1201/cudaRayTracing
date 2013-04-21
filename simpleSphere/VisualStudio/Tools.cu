#include "Tools.cuh"
#include <math_functions.h>

// operator of Vector3D
__device__ __host__ inline 
Vector3D operator+(const Vector3D& param1, const Vector3D& param2){
	Vector3D result;
	result.x = param1.x + param2.x;
	result.y = param1.y + param2.y;
	result.z = param1.z + param2.z;
	return result;
}

__device__ __host__ inline 
Vector3D operator-(const Vector3D& param1, const Vector3D& param2){
	Vector3D result;
	result.x = param1.x - param2.x;
	result.y = param1.y - param2.y;
	result.z = param1.z - param2.z;
	return result;
}

__device__ __host__ inline
Vector3D operator*(const float param1,const Vector3D& param2){
	Vector3D result;
	result.x = param1 * param2.x;
	result.y = param1 * param2.y;
	result.z = param1 * param2.z;
	return result;
}

__device__ __host__ inline 
Vector3D operator*(const Vector3D& param1,const float param2){
	Vector3D result;
	result.x = param1.x * param2;
	result.y = param1.y * param2;
	result.z = param1.z * param2;
	return result;
}
__device__ __host__ inline 
Vector3D operator/(const Vector3D& param1,const float param2){
	Vector3D result;
	result.x = param1.x / param2;
	result.y = param1.y / param2;
	result.z = param1.z / param2;
	return result;
}

//__global__ Vector3D& operator=(const Vector3D &param);
__device__ __host__ inline 
float Exponentiation(const Vector3D &param){
	float result;
	result = param.x * param.x + param.y * param.y + param.z * param.z;
	return result;
}

__device__ __host__	inline
float Norm(const Vector3D &param){
		float result;
		result = sqrt( Exponentiation( param ));
		return result;
}

__device__ __host__	inline 
float operator*(const Vector3D &param1,const Vector3D &param2){
	float result;
	result = param1.x * param2.x + param1.y * param2.y + param1.z * param2.z;
	return result;
}

__device__ __host__	inline 
Vector3D CrossProduct(const Vector3D &param1, const Vector3D &param2){
	Vector3D result;
	result.x = param1.y * param2.z - param1.z * param2.y;
	result.y = param1.z * param2.x - param1.x * param2.z;
	result.z = param1.x * param2.y - param1.y * param2.x;	
	return result;
}

__device__ __host__	inline 
Vector3D Normalize(const Vector3D &param){
	Vector3D result;
	float norm = Norm(param);
	result.x = param.x / norm;
	result.y = param.y / norm;
	result.z = param.z / norm;
	return result;
}


__device__ __host__ inline 
float Distance(const Point3D &param1,const Point3D &param2){
	return ( sqrtf( (param1.x - param2.x ) * ( param1.x - param2.x ) 
		+ ( param1.y - param2.y ) * ( param1.y - param2.y )
		+ ( param1.z - param2.z ) * ( param1.z - param2.z ) ));
}
/*
// operator of Point
__device__ __host__ inline 
Point3D operator+(const Point3D &param1, const Vector3D& param2){
	Point3D result;
	result.x = param1.x + param2.x;
	result.y = param1.y + param2.y;
	result.z = param1.z + param2.z;
	return result;
}
__device__ __host__ inline 
Point3D operator-(const Point3D &param1, const Vector3D& param2){
	Point3D result;
	result.x = param1.x - param2.x;
	result.y = param1.y - param2.y;
	result.z = param1.z - param2.z;
	return result;
}
__device__ __host__ inline 
Vector3D operator-(const Point3D& param1, const Point3D& param2){
	Vector3D result;
	result.x = param1.x - param2.x;
	result.y = param1.y - param2.y;
	result.z = param1.z - param2.z;
	return result;
}
*/
/*
__device__ __host__ inline 
RGBColor operator+(const RGBColor& param1, const RGBColor &param2){
	RGBColor result;
	result.x = param1.x + param2.x;
	result.y = param1.y + param2.y;
	result.z = param1.z + param2.z;
	return result;
}

__device__ __host__ inline 
RGBColor operator-(const RGBColor& param1, const RGBColor &param2){
	RGBColor result;
	result.x = param1.x - param2.x;
	result.y = param1.y - param2.y;
	int tempZ = param1.z - param2.z;
	
	return result;
}
*//*
__device__ __host__ inline 
RGBColor operator*(const RGBColor& param1, const float &param2){
	RGBColor result;
	result.x = param1.x * param2;
	result.y = param1.y * param2;
	result.z = param1.z * param2;

	return result;
}
*/
__device__ __host__ inline 
RGBColor powc(const RGBColor& param1, const RGBColor &param2){
	RGBColor result;
	result.x = param1.x * param2.x;
	result.y = param1.y * param2.y;
	result.z = param1.z * param2.z;
	return result;
}
/*
__device__ __host__ inline 
RGBColor operator/(const RGBColor& param1, const float &param2){
	RGBColor result;
	result.x = param1.x / param2;
	result.y = param1.y / param2;
	result.z = param1.z / param2;

	return result;
}
*/
__device__ __host__ inline 
uchar3 MapToUchar(const RGBColor& param){
	uchar3 result;
	result.x = param.x * 255;
	result.y = param.y * 255;
	result.z = param.z * 255;

	if( param.x > 1.0 ){
		result.x = 255;
	}else if( param.x < 0.0 ){
		result.x = 0;
	}
	if( param.y > 1.0 ){
		result.y = 255;
	}else if( param.x < 0.0 ){
		result.y = 0;
	}
	if( param.z > 1.0 ){
		result.z = 255;
	}else if( param.x < 0.0 ){
		result.z = 0;
	}
	return result;
}
/* pseudo random number generator */
/*  linear congruential  */
__device__
static unsigned int prng_seed = 0xcdcdcdcd;

__device__ 
void rand_seed(unsigned int seed){
	prng_seed = seed;
}

__device__ inline
unsigned int uintRand(){
	prng_seed = ( prng_seed * 0xababa ) % ( MAX_RANDOM_NUM );
	return prng_seed;
}
__device__ inline
float floatRand(){
	prng_seed = ( prng_seed * 0xababa ) % ( MAX_RANDOM_NUM );

	float result = float(prng_seed) / float(MAX_RANDOM_NUM);

	return result;
}