#if !defined __TOOLS_CUH__
#define __TOOLS__CUH__

typedef unsigned char Byte;

#define hugeValue  3.402823466e+38F
#define PI         3.14f
#define invPI      0.31831f

typedef float2		Point2D;
typedef float3		Point3D;
typedef float3		Vector3D;
typedef float3		Normal;
typedef float3		RGBColor;

#define Vector3D(x,y,z)		make_float3(x,y,z)
#define Point3D(x,y,z)		make_float3(x,y,z)
#define Normal(x,y,z)		make_float3(x,y,z)
#define RGBColor(r,g,b)		make_float3(r,g,b)
#define Point2D(x,y)		make_float2(x,y)

#define white	make_float3(1.0,1.0,1.0)
#define black	make_float3(0,0,0)
#define red		make_float3(1.0,0,0)
#define green   make_float3(0,1.0,0)
#define blue	make_float3(0,0,1.0)
#define yellow	make_float3(1.0,1.0,0)
// operator of Vector3D

extern __device__ __host__ inline Vector3D operator+(const Vector3D& param1, const Vector3D& param2);
extern __device__ __host__ inline Vector3D operator-(const Vector3D& param1, const Vector3D& param2);
extern __device__ __host__ inline Vector3D operator*(const float param1,const Vector3D& param2);
extern __device__ __host__ inline Vector3D operator*(const Vector3D& param1,const float param2);
extern __device__ __host__ inline Vector3D operator/(const Vector3D& param1,const float param2);
//__global__ Vector3D& operator=(const Vector3D &param);
extern __device__ __host__ inline float Exponentiation(const Vector3D &param);
extern __device__ __host__ inline float Norm(const Vector3D &param);
extern __device__ __host__ inline float operator*(const Vector3D &param1,const Vector3D &param2);
extern __device__ __host__ inline Vector3D CrossProduct(const Vector3D &param1, const Vector3D &param2);
extern __device__ __host__ inline Vector3D Normalize(const Vector3D &param);

extern __device__ __host__ inline float Distance(const Point3D &param1,const Point3D &param2);
// operator of Point3D
//__device__ __host__ inline Point3D operator+(const Point3D &param1, const Vector3D& param2);
//__device__ __host__ inline Point3D operator-(const Point3D &param1, const Vector3D& param2);
//__device__ __host__ inline Vector3D operator-(const Point3D& param1, const Point3D& param2);
//operator of RGBColor
//extern __device__ __host__ inline RGBColor operator+(const RGBColor& param1, const RGBColor &param2);
//extern __device__ __host__ inline RGBColor operator-(const RGBColor& param1, const RGBColor &param2);
//extern __device__ __host__ inline RGBColor operator*(const RGBColor& param1, const float &param2);
extern __device__ __host__ inline RGBColor powc(const RGBColor& param1, const RGBColor &param2);
//extern __device__ __host__ inline RGBColor operator/(const RGBColor& param1, const float &param2);

extern __device__ __host__ inline uchar3 MapToUchar(const RGBColor& param);
/***** some utility function *****/

/* pseudo random number generator */
/*  linear congruential  */
#define MAX_RANDOM_NUM	0x7fffffff

/*
__device__ __host__
void rand_seed(unsigned int seed);
*/

extern __device__ __host__ inline
unsigned int uintRand();

extern __device__ __host__ inline
float floatRand();

#define STACK_MAX	10

#endif