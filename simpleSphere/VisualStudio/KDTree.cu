#include "defines.cuh"
#include "function_defines.cuh"

typedef int BoundEdgeType;
#define START	0
#define END		1

typedef int Axis;
#define AXIS_X 0
#define AXIS_Y 1
#define AXIS_Z 2

struct BoundEdge{
	BoundEdgeType type;
	float t;
	int objectIdx;
};

__host__ inline
BoundEdge newEdge( float tt, int pn, bool starting ){
	BoundEdge result;
	result.t = tt;
	result.objectIdx = pn;
	result.type = starting ? START : END;
	return result;
}

__host__
bool operator<(const BoundEdge &e1,const BoundEdge &e2){
	if( e1.t == e2.t ){
		return e1.type < e1.type;
	}
	else return e1.t < e2.t;
}

__host__
bool operator>(const BoundEdge &e1,const BoundEdge &e2){
	if( e1.t == e2.t ){
		return e1.type > e1.type;
	}
	else return e1.t > e2.t;
}

#define ARENA_MAX_SIZE	1<<STACK_MAX

struct KDNodeArena{
	KDNode block[ARENA_MAX_SIZE];
//	int valid[ ARENA_MAX_SIZE / 32];
	int nextValidBlock;

};

__host__ inline
void initArena( KDNodeArena* arena){
	arena->nextValidBlock = 0;
}

__host__ inline
KDNode* GetNextValidNode(KDNodeArena *arena){
	if( arena->nextValidBlock < ARENA_MAX_SIZE ){
		arena->nextValidBlock = arena->nextValidBlock + 1;
		return &(arena->block[arena->nextValidBlock - 1]);
	}else{
		return NULL;
	}
}

#include<vector>

using namespace std;

#define isectCost	5
#define travalCost	1

__host__
void BuildTree(int nodeNum, const BBox &nodeBounds, 
	const vector<BBox> &allObjBounds, int *numObject,
	int nObjects,int depth, BoundEdge *edges[3]){

		// choose split axis position for interior node
		int bestAxis = -1, bestOffset = -1;
		float bestCost = hugeValue;
		float oldCost = isectCost ;
		float totalSA = SurfaceArea(nodeBounds);
		float invTatalSA = 1.f / totalSA;
		Vector3D d = nodeBounds.pMax - nodeBounds.pMin;

		//choose which axis to split along 
		Axis axis = MaximumExtent( nodeBounds );

		int retries = 0;


}

__host__
KDNode* BuildKDTree(GeometricObject **objects, int numObjects){
	KDNodeArena arena;
	initArena(&arena);

	KDNode *tree;

	vector<BBox> allObjBounds;
	BBox objBox = Bounds(objects[0]);
	BBox bounds = objBox;
	allObjBounds.push_back(objBox);
	for( int i = 1 ; i < numObjects ; i ++){
		objBox = Bounds(objects[i]);
		bounds = Union(tree->box,objBox);
		allObjBounds.push_back(objBox);
	}

	BoundEdge *edges[3];
	edges[AXIS_X] = (BoundEdge*)malloc( 2 * numObjects * sizeof( BoundEdge ));
	edges[AXIS_Y] = (BoundEdge*)malloc( 2 * numObjects * sizeof( BoundEdge ));
	edges[AXIS_Z] = (BoundEdge*)malloc( 2 * numObjects * sizeof( BoundEdge ));

	for(int i = 0; i < numObjects ; i ++){

		edges[AXIS_X][2 * i] = newEdge( allObjBounds[i].pMin.x ,i, true );		
		edges[AXIS_X][2 * i + 1] = newEdge( allObjBounds[i].pMax.x ,i, false );
		
		edges[AXIS_Y][2 * i] = newEdge( allObjBounds[i].pMin.y ,i, true );		
		edges[AXIS_Y][2 * i + 1] = newEdge( allObjBounds[i].pMax.y ,i, false );
		
		edges[AXIS_Z][2 * i] = newEdge( allObjBounds[i].pMin.z ,i, true );		
		edges[AXIS_Z][2 * i + 1] = newEdge( allObjBounds[i].pMax.z ,i, false );
	}

	return tree;
}


__device__
bool KDHitObject(World *w, Ray ray, ShadeRec *sr){

	return true;
}