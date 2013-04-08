#if !defined __SIMPLESPHERE_K_CUH__
#define __SIMPLESPHERE_K_CUH__

#include <GL\glew.h>
#include <GL\freeglut.h>

// CUDA standard includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <device_launch_parameters.h>
#include <device_functions.h>

// CUDA helper functions
#include <helper_functions.h>
//#include <rendercheck_gl.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)
/*
#define getFromDevice(T,obj,source) \
	T obj,*pointer;\
	cudaMemcpy(&pointer, source,sizeof(T*),cudaMemcpyDeviceToHost);\
	cudaCheckErrors("pointer copy failed");\
	cudaMemcpy( &obj,pointer,sizeof(T),cudaMemcpyDeviceToHost);\
	cudaCheckErrors("object copy failed")\
	*/
#include "defines.cuh"
#include "function_defines.cuh"

//struct World;

__global__ void whiteNoise_k( uchar3 *ptr );

void cudaRayTracingInit(World **h_w, World **d_w, int width, int height);

void cudaRayTracing(World *w,int width, int height, RGBColor *buffer);

#endif