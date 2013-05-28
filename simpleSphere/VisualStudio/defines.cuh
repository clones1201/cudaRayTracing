#if !defined __DEFINES_CUH__
#define __DEFINES_CUH__

#include "Tools.cuh"

/* struture definition */
struct Ray;

struct GeometricObject;
struct Sphere;
struct Plane;

struct Camara;
struct Pinhole;

struct BRDF;
struct Lambertian;
struct PerfectSpecular;
struct GlossySpecular;

struct Light;
struct Directional;
struct PointLight;

struct Material;
struct Matte;
struct Phong;

struct ShadeRec;
struct ViewPlane;
struct World;

/* Sampler Definition */
#define SAMPLE_POOL_SIZE	100
typedef int SamplerType;
#define SAMPLER_PURERANDOM		0
#define SAMPLER_REGULAR			1
#define SAMPLER_JITTERED		2
#define SAMPLER_NROOKS			3
#define SAMPLER_MULTIJITTERED	4
#define SAMPLER_HAMMERSLEY		5
#define SAMPLER_HEMISPHERE		6

typedef int SampleScale;
#define SAMPLE_SCALE_1		0
#define SAMPLE_SCALE_4		1
#define SAMPLE_SCALE_16		2
#define SAMPLE_SCALE_64		3
#define SAMPLE_SCALE_256	4

struct Sampler{
	SamplerType type;
	int numSamples;
	unsigned long count;
	Point2D sample[SAMPLE_POOL_SIZE];
};

struct Ray{
	Point3D o;
	Vector3D d;
};

/* Geometric Object Definitions */
typedef int GeometricObjectType;
#define GMO_TYPE_VIRTUAL	0
#define GMO_TYPE_SPHERE		1
#define GMO_TYPE_PLANE		2
#define GMO_TYPE_BBOX		3

struct GeometricObject{
	GeometricObjectType type;
	RGBColor color;
	int materialIdx;
};

struct Sphere{
	GeometricObjectType type;
	RGBColor color;	
	int materialIdx;
	Point3D	center;
	float radius;
};

struct Plane{	
	GeometricObjectType type;
	RGBColor color;	
	int materialIdx;
	Point3D point;
	Normal normal;
};

/* KD Tree
*/

struct BBox{
	Point3D pMax;
	Point3D pMin;
};

typedef int KDType;

#define KD_TYPE_INTERIORNODE	0
#define KD_TYPE_LEAF	1

struct KDNode{
	KDType type;
	BBox box;
	int depth;
	KDNode *left;
	KDNode *right;
};

struct KDLeaf{
	KDType type;
	BBox box;
	int depth;
	int numObject;
	int *objects;
};

/* camara */

typedef int CamaraType;
#define CAMARA_TYPE_DEFAULT			0
#define CAMARA_TYPE_PINHOLE			1

struct Camara{
	CamaraType type;
	Point3D eye;
	Point3D lookat;
	Vector3D up;
	Vector3D u,v,w;
	float exposure_time;
};

struct Pinhole{
	CamaraType type;
	Point3D eye;
	Point3D lookat;
	Vector3D up;
	Vector3D u,v,w;
	float exposure_time;
	
	float viewDistance;
	float zoom;
};

/************************/
/*****  BRDF  ******/
typedef int BRDFType;

#define	BRDF_TYPE_LAMBERTIAN        0
#define BRDF_TYPE_PERFECTSPECULAR   1
#define BRDF_TYPE_GLOSSYSPECULAR    2

struct BRDF{
	BRDFType type;
};

struct Lambertian{    
	BRDFType type;
	float kd;
	RGBColor cd;
};

struct PerfectSpecular{
	BRDFType type;
	float kr;
	RGBColor cr;
};

struct GlossySpecular{
	BRDFType type;
	float ks;
	int exp;
	RGBColor cs;
};

/********************/
/***** Light *****/
typedef int LightType;
#define LIGHT_TYPE_AMBIENT       0
#define LIGHT_TYPE_DIRECTIONAL   1
#define LIGHT_TYPE_POINTLIGHT    2
#define LIGHT_TYPE_AMBIENTOCCLUDER 3
struct Light{
	LightType type;
	bool shadows;
};

struct Ambient{
	LightType type;
	bool shadows;

	float ls;
	RGBColor color;
};

struct Directional{
	LightType type;
	bool shadows;
	
	float ls;
	RGBColor color;
	Vector3D dir;
};

struct PointLight{
	LightType type;
	bool shadows;

	float ls;
	RGBColor color;
	Point3D pos;
};

struct AmbientOccluder{
	LightType type;
	bool shadows;
	
	float ls;
	RGBColor color;
	Vector3D u,v,w;
};

/****************/
/*   Material   */
typedef int MaterialType;
#define MATERIAL_TYPE_MATTE				0
#define MATERIAL_TYPE_PHONG				1
#define MATERIAL_TYPE_REFLECTIVE		2
#define MATERIAL_TYPE_GLOSSYREFLECTOR	3

struct Material{
	MaterialType type;
};

struct Matte{
	MaterialType type;
	Lambertian ambientBRDF;
	Lambertian diffuseBRDF;
};

struct Phong{
	MaterialType type;
	Lambertian ambientBRDF;
	Lambertian diffuseBRDF;
	GlossySpecular specularBRDF;
};

struct Reflective{
	MaterialType type;
	Lambertian ambientBRDF;
	Lambertian diffuseBRDF;
	GlossySpecular specularBRDF;
	PerfectSpecular reflectiveBRDF;
};

struct GlossyReflector{
	MaterialType type;
	Lambertian ambientBRDF;
	Lambertian diffuseBRDF;
	GlossySpecular specularBRDF;
	GlossySpecular glossySpecularBRDF;
};
/****************/
struct ShadeRec{
	bool hitAnObject;
	Material *material;
	Point3D hitPoint;
	Point3D localHitPoint;
	Normal normal;
	RGBColor color;//only used in chapter 3
	Ray ray;
	int depth;
	Vector3D dir;
	World *w;
};

struct ViewPlane{
public:
	int hres;
	int vres;
	float s;
	//float gamma;
	//float inv_gamma;

	Sampler *sampler;
};

struct World{
public:
	GeometricObject **objects;
	int numObject;
	Material **materials;
	int numMaterial;
	Light **lights;
	int numLight;
	Light *ambient;

	ViewPlane *vp;
	RGBColor backgroundColor;
//	Sphere *sphere;
	Camara *camara;

	KDNode *kdTree;
};

#endif