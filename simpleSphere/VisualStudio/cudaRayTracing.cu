#include "cudaRayTracing.cuh"

extern World *h_w;
extern World *d_w;

/* Stack definition 
*/

template <typename T>
struct Stack{
	T item[STACK_MAX];
	int top;
};

template <typename T>
__device__ __host__
void initStack( Stack<T> *stack){
	stack->top = 0;
}

template <typename T>
__device__ __host__
bool isEmpty( Stack<T> *stack){
	if( stack->top == 0){
		return true;
	}
	return false;
}

template <typename T>
__device__ __host__
bool isFull( Stack<T> *stack){
	if( stack->top > STACK_MAX){
		return true;
	}
	return false;
}

template <typename T>
__device__ __host__
void Push( Stack<T> *stack, T item){
	if( ! isFull<T>( stack ) ){
		stack->item[stack->top] = item; 
		stack->top = stack->top + 1;
	}
}

template <typename T>
__device__ __host__
T Pop( Stack<T> *stack){
/*	if( isEmpty<T>( stack ) ){
		return ;
	}*/
	stack->top = stack->top - 1;
	return (stack->item[stack->top]);
}

#include "Ray.cu"
#include "Tools.cu"
#include "Sampler.cu"
#include "ShadeRec.cu"
#include "Tracer.cu"
#include "Camara.cu"
#include "Light.cu"
#include "Material.cu"
#include "BRDF.cu"
#include "ViewPlane.cu"
#include "GeometricObject.cu"
#include "BBox.cu"
#include "KDTree.cu"
#include "World.cu"

void cudaRayTracingInit(World **h_w,World **d_w,int width,int height){
	//cudaMalloc(&w,sizeof(World));
	BuildWorld(h_w,d_w,width,height);
}

void cudaRayTracing(World *w, int width, int height, uchar3 *buffer){
	RenderScene( w, width, height, buffer);
	cudaDeviceSynchronize();
}