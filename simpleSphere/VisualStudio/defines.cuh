#if !defined __DEFINES_CUH__
#define __DEFINES_CUH__

#include "Tools.cuh"

/* Sampler Definition */
typedef int SamplerType;
#define SAMPLER_PURERANDOM		0
#define SAMPLER_REGULAR			1
#define SAMPLER_JITTERED		2
#define SAMPLER_NROOKS			3
#define SAMPLER_MULTIJITTERED	4
#define SAMPLER_HAMMERSLEY		5

typedef int SampleScale;
#define SAMPLE_SCALE_1		0
#define SAMPLE_SCALE_4		1
#define SAMPLE_SCALE_16		2
#define SAMPLE_SCALE_64		3
#define SAMPLE_SCALE_256	4


/* struture definition */
struct Ray;
struct Sphere;
struct ShadeRec;
struct ViewPlane;
struct World;

struct Ray{
	Point3D o;
	Vector3D d;
};

/* Geometric Object Definitions */
typedef int GeometricObjectType;
#define GMO_TYPE_VIRTUAL	0
#define GMO_TYPE_SPHERE		1
#define GMO_TYPE_PLANE		2

__device__ static const int size_of_geometricObj[] = {
	17,25
};

struct GeometricObject{
	GeometricObjectType type;
	RGBAColor color;
};

struct Sphere{
	GeometricObjectType type;
	RGBAColor color;
	Point3D	center;
	float radius;
};

struct Plane{	
	GeometricObjectType type;
	RGBAColor color;
	Point3D point;
	Normal normal;
};


struct ShadeRec{
	bool hitAnObject;
	Point3D localHitPoint;
	Normal normal;
	RGBAColor color;	
};

struct ViewPlane{
public:
	int hres;
	int vres;
	float s;
	//float gamma;
	//float inv_gamma;

	SamplerType samplerType;
	SampleScale sampleScale;
};

struct World{
public:
	GeometricObject **object;
	int numObject;
	ViewPlane *vp;
	RGBAColor backgroundColor;
//	Sphere *sphere;

};


/*Tracer function */

extern __device__  
void singleSphereTraceRay(World *w,Sphere *s,Ray ray,RGBAColor *color);

extern __device__
void multiObjTraceRay(World *w, Ray ray, RGBAColor *color);

/* Geometric Object function */
extern __device__   
bool sphereHit(Sphere *sphere, Ray ray,float *tmin, ShadeRec *sr);

extern __device__
bool planeHit(Plane *plane,Ray ray, float *tmin, ShadeRec *sr);

extern __host__
void initPlane(Plane **pl, Point3D p,Normal n,RGBAColor color);

extern __host__
void initSphere(Sphere **s, Point3D c, float r,RGBAColor color);

extern __host__
void freePlane(Plane **pl);

extern __host__
void freeSphere(Sphere **s);

/* Sampler function */

extern __device__ inline
Point2D	getSampleUnitSquare(SamplerType type, int idx,SampleScale scale);

extern __device__ __host__ inline
int getSampleNum(SampleScale scale);

extern __device__ __host__ inline
SampleScale getSampleScale(int num);


/* World function */
extern void build_world(World **w);

extern void render_scene(World *w,int width,int height,RGBAColor *buffer);

extern void hitBareBonesObject(World *w, Ray ray,ShadeRec *sr);

#endif
