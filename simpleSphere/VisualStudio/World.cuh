#ifndef __WORLD_CUH__
#define __WORLD_CUH__

#include "simpleSphere.cuh"
#include "ViewPlane.cuh"

struct World{
public:

	ViewPlane *vp;
	RGBAColor background_color;
	Sphere *sphere;

};

void build_world(World *w,RGBAColor *buffer);

void render_scene(World *w);

#endif