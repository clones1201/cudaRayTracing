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

__host__
void initLeaf(KDNode* node,int *objNums, int nObj, int depth){
	node->type = KD_TYPE_LEAF;
	KDLeaf* leaf = (KDLeaf*)node;
	leaf->numObject = nObj;
	leaf->depth = depth;
	if( nObj == 0 ){
		leaf->objects = NULL;
	}
	else{
		leaf->objects = (int*)malloc( nObj * sizeof( int ));
		for( int i = 0 ; i < nObj ; ++ i){
			leaf->objects[i] = objNums[i];
		}
	}
}

__host__
void initInterior(KDNode *node,int depth){
	node->type = KD_TYPE_INTERIORNODE;
	node->depth = depth;
}

#include<vector>

using namespace std;

#define isectCost	5
#define traversalCost	1
#define emptyBonus		-0.5
#define maxDepth		10
#define maxObjects		16

__host__
void BuildTree(KDNode** node, const BBox &nodeBounds, 
	const vector<BBox> &allObjBounds, int *objNums,
	int nObjects,int depth, BoundEdge *edges[3],
	int *objects0,int *objects1, int badRefines, KDNodeArena *arena){

		//Get next free node from nodes arena
		*node = GetNextValidNode(arena);
		(*node)->box = nodeBounds;
		//initialize leaf node if termination criteria met
		if( nObjects <= maxObjects || depth == 0){
			initLeaf(*node,objNums,nObjects,maxDepth - depth);
			return;
		}

		// choose split axis position for interior node
		int bestAxis = -1, bestOffset = -1;
		float bestCost = hugeValue;
		float oldCost = isectCost * nObjects ;
		float totalSA = SurfaceArea(nodeBounds);
		float invTotalSA = 1.f / totalSA;
		Vector3D d = nodeBounds.pMax - nodeBounds.pMin;

		//choose which axis to split along 
		Axis axis = MaximumExtent( nodeBounds );

		int retries = 0;
retrySplit:
		//Intialize edges for axis
		for( int i = 0 ; i < nObjects; ++ i){
			int pn = objNums[i];
			BBox box = allObjBounds[pn];
			edges[axis][2*i] = newEdge( ((float*)&(box.pMax))[axis] , pn,true);
			edges[axis][2*i+1] = newEdge( ((float*)&(box.pMin))[axis], pn, false);
		}
		sort( &edges[axis][0], &edges[axis][2 * nObjects] );

		/* Compute cost of all splits for axis to find best 
		*/
		int nBelow = 0, nAbove = nObjects;
		for( int i = 0 ; i < 2 * nObjects ; i ++ ){
			if( edges[axis][i].type == END ) --nAbove;
			float edget = edges[axis][i].t;
			if( edget > ((float*)&(nodeBounds.pMin))[axis] &&
				edget < ((float*)&(nodeBounds.pMax))[axis]){
					// Compute cost for split at i th edge
				Axis otherAxis0 = (axis + 1 ) % 3, otherAxis1 = (axis + 2) % 3;

				float belowSA = 2 * ( ((float*)&d)[otherAxis0] * ((float*)&d)[otherAxis1] +
					( edget - ((float*)&(nodeBounds.pMin))[axis] ) *
					( ((float*)&d)[otherAxis0] + ((float*)&d)[otherAxis1] ));
	
				float aboveSA =  2 * ( ((float*)&d)[otherAxis0] * ((float*)&d)[otherAxis1] +
					( ((float*)&(nodeBounds.pMax))[axis] - edget ) *
					( ((float*)&d)[otherAxis0] + ((float*)&d)[otherAxis1] ));

				float pBelow = belowSA * invTotalSA;
				float pAbove = aboveSA * invTotalSA;
				float eb = ( nAbove == 0 || nBelow == 0 ) ? emptyBonus : 0.f;
				float cost = traversalCost +
					isectCost * ( 1.f - eb ) * ( pBelow * nBelow + pAbove * nAbove );

				//update best split if this is lowest cost so far
				if( cost < bestCost/* && (nBelow + nAbove == nObjects )*/ ){
					bestCost = cost;
					bestAxis = axis;
					bestOffset = i;
				}
			}
			if( edges[axis][i].type == START ) ++nBelow;
		}

		//Create leaf if no good splits were found
		if( bestAxis == -1 && retries < 2 ){
			retries++;
			axis = (axis+1)%3;
			goto retrySplit;
		}
		if( bestCost > oldCost ) ++badRefines;
		if( ( bestCost > 4.f * oldCost && nObjects < 16 ) ||
			bestAxis == -1 || badRefines == 3 ){
				initLeaf(*node,objNums,nObjects,maxDepth - depth);
				return;
		}

		//Classify objects with respect to split
		int n0 = 0,n1 = 0;
		for(int i = 0 ; i < bestOffset ; ++i ){
			if (edges[bestAxis][i].type == START)
				objects0[n0++] = edges[bestAxis][i].objectIdx;
		}
		for(int i = bestOffset ; i < 2 * nObjects ; ++i ){
			if (edges[bestAxis][i].type == END)
				objects1[n1++] = edges[bestAxis][i].objectIdx;
		}

		//Recurively initialize chidren nodes
		float tsplit = edges[bestAxis][bestOffset].t;
		BBox bounds0 = nodeBounds, bounds1 = nodeBounds;
		((float*)&(bounds0.pMax))[bestAxis] = tsplit;((float*)&(bounds1.pMin))[bestAxis] = tsplit;
		initInterior(*node,maxDepth - depth);
		BuildTree( &((*node)->left) , bounds0 , allObjBounds, objects0, n0, depth - 1 , edges, objects0, objects1 + nObjects,badRefines,arena );
		BuildTree( &((*node)->right) , bounds1 , allObjBounds,objects1, n1, depth - 1 , edges, objects0, objects1 + nObjects,badRefines,arena );
}

__host__
KDNode* BuildKDTree(GeometricObject **objects, int numObjects){
	KDNodeArena arena;
	initArena(&arena);

	KDNode *tree = NULL;

	vector<BBox> allObjBounds;
	BBox objBox = Bounds(objects[0]);
	BBox bounds = objBox;
	allObjBounds.push_back(objBox);
	for( int i = 1 ; i < numObjects ; i ++){
		objBox = Bounds(objects[i]);
		bounds = Union(bounds,objBox);
		allObjBounds.push_back(objBox);
	}

	BoundEdge *edges[3];
	edges[AXIS_X] = (BoundEdge*)malloc( 2 * numObjects * sizeof( BoundEdge ));
	edges[AXIS_Y] = (BoundEdge*)malloc( 2 * numObjects * sizeof( BoundEdge ));
	edges[AXIS_Z] = (BoundEdge*)malloc( 2 * numObjects * sizeof( BoundEdge ));
	/*
	for(int i = 0; i < numObjects ; i ++){

		edges[AXIS_X][2 * i] = newEdge( allObjBounds[i].pMin.x ,i, true );		
		edges[AXIS_X][2 * i + 1] = newEdge( allObjBounds[i].pMax.x ,i, false );
		
		edges[AXIS_Y][2 * i] = newEdge( allObjBounds[i].pMin.y ,i, true );		
		edges[AXIS_Y][2 * i + 1] = newEdge( allObjBounds[i].pMax.y ,i, false );
		
		edges[AXIS_Z][2 * i] = newEdge( allObjBounds[i].pMin.z ,i, true );		
		edges[AXIS_Z][2 * i + 1] = newEdge( allObjBounds[i].pMax.z ,i, false );
	}*/
	//Allocate working memory for kd-tree construction
	int *objs0 = (int*)malloc( numObjects * sizeof(int) );
	int *objs1 = (int*)malloc( (maxDepth + 1 ) * numObjects *sizeof(int) );
	//initialize objNums for kd-tree construction
	int *objNums = (int*)malloc( numObjects * sizeof(int) );
	for( int i = 0 ; i < numObjects ; ++ i){
		objNums[i] = i ;
	}
	//start recurive construction of kd-tree
	BuildTree(&tree ,bounds,allObjBounds,objNums,numObjects,maxDepth,edges,
		objs0,objs1,0,&arena);

	//free working memory
	delete [] edges[AXIS_X];
	delete [] edges[AXIS_Y];
	delete [] edges[AXIS_Z];

	delete [] objs0;
	delete [] objs1;
	delete [] objNums;
	return tree;
}


__device__
bool KDHitObject(World *w, Ray ray, ShadeRec *sr){

	return true;
}