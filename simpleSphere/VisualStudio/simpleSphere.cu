#include "simpleSphere.cuh"

#include "Ray.cu"
#include "Tools.cu"
#include "Sampler.cu"
#include "ShadeRec.cu"
#include "Tracer.cu"
#include "ViewPlane.cu"
#include "GeometricObject.cu"
#include "World.cu"

extern GLuint pbo;
extern struct cudaGraphicsResource *cuda_pbo_resource;
extern World *w;

void simpleSphere_init(World **w,int width,int height){
	
	//cudaMalloc(&w,sizeof(World));
	build_world(w,width,height);
}

extern "C"
void simpleSphere(World *w,int width,int height){
	uchar4* devPtr;
	size_t size;
	checkCudaErrors( cudaGraphicsMapResources(1, &cuda_pbo_resource, NULL) );	
    getLastCudaError("cudaGraphicsMapResources failed");

	checkCudaErrors( 
		cudaGraphicsResourceGetMappedPointer( (void**)&devPtr,
												&size,
												cuda_pbo_resource )
												);	
    getLastCudaError("cudaGraphicsResourceGetMappedPointer failed");
	
	render_scene(w,width,height,devPtr);
    
	cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);
    getLastCudaError("cudaGraphicsUnmapResources failed");
}

__global__ 
	void whiteNoise_k( uchar4* ptr){
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;
		int offset = x + y * blockDim.x * gridDim.x;
		//unsigned char color = rand();
		ptr[offset].x = 70;
		ptr[offset].y = 140;
		ptr[offset].z = 210;
		ptr[offset].w = 255;
}

extern "C"
void whiteNoise(int x , int y){
	uchar4* devPtr;
	size_t size;
	checkCudaErrors( cudaGraphicsMapResources(1, &cuda_pbo_resource, NULL) );	
    getLastCudaError("cudaGraphicsMapResources failed");

	checkCudaErrors( 
		cudaGraphicsResourceGetMappedPointer( (void**)&devPtr,
												&size,
												cuda_pbo_resource )
												);	
    getLastCudaError("cudaGraphicsResourceGetMappedPointer failed");

	dim3 grids( x/16 , y/16 );
	dim3 threads( 16, 16 );
	whiteNoise_k<<<grids,threads>>>( devPtr );

	getLastCudaError("whiteNoise_k failed.");
    cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);
    getLastCudaError("cudaGraphicsUnmapResources failed");

}


