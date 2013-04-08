#include <GL\glew.h>
#include <GL\freeglut.h>

// Includes
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// CUDA standard includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

//#include <crt\device_runtime.h>
//#include <crt\host_runtime.h>

//#include <device_launch_parameters.h>
//#include <device_functions.h>

// CUDA helper functions
//#include <helper_functions.h>
//#include <rendercheck_gl.h>
//#include <helper_cuda.h>
//#include <helper_cuda_gl.h>
#pragma comment(lib,"glew32.lib")

#include "cudaRayTracing.cuh"

const char *sSDKname = "cudaRayTracing";

static int wWidth = 512;
static int wHeight = 512;

static int fpsCount = 0;
static int fpsLimit = 1;
StopWatchInterface *timer = NULL;

GLuint pbo = 0;		// OpenGL pixel buffer object
struct cudaGraphicsResource *cuda_pbo_resource; // handles OpenGL-CUDA exchange
World *h_w;
World *d_w;

void simpleSphere(World *w,int width,int height);

void clearup();
void reshape(int x, int y);

void display(void){

	sdkStartTimer(&timer);	
	
	glClearColor(0,0,0,1);

	glClear(GL_COLOR_BUFFER_BIT);
	
	simpleSphere(d_w,wWidth,wHeight);
	//whiteNoise(512,512);

	 // render points from vertex buffer
	glDrawPixels(wWidth,wHeight,GL_RGB,GL_UNSIGNED_BYTE,0);
	    
	fpsCount++;
	sdkStopTimer(&timer);
    glutSwapBuffers();

	if (fpsCount == fpsLimit)
    {
        char fps[256];
        float ifps = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
        sprintf(fps, "Cuda/GL Ray Tracing (%d x %d): %3.1f fps", wWidth, wHeight, ifps);
        glutSetWindowTitle(fps);
        fpsCount = 0;
        fpsLimit = (int)MAX(ifps, 1.f);
        sdkResetTimer(&timer);
    }

    glutPostRedisplay();
}

void keyboard(unsigned char key, int x, int y){
}

void click(int button,int updown, int x, int y){
}

void motion(int x, int y){
	
    glutPostRedisplay();
}

void reshape(int x, int y){
	wWidth = x;
    wHeight = y;
    glViewport(0, 0, x, y);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, 1, 1, 0, 0, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glutPostRedisplay();
}

int initGL(int argc,char *argv[]){

	glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);
    glutInitWindowSize(wWidth, wHeight);
    glutCreateWindow( "Render Simple Sphere" );
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(click);
    glutMotionFunc(motion);
    glutReshapeFunc(reshape);

	glewInit();
	
    if (! glewIsSupported(
            "GL_ARB_vertex_buffer_object"
        ))
    {
        fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
        fflush(stderr);
        return false;
    }

	return true;
}

void clearup(){
	cudaGraphicsUnregisterResource(cuda_pbo_resource);


    // Free all host and device resources

    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
    glDeleteBuffersARB(1, &pbo);
}

void initCuda(int devID){
    cudaDeviceProp deviceProps;
	cudaSetDevice(0);
	cudaCheckErrors("cuda device set failed");

    // get number of SMs on this GPU
	cudaGetDeviceProperties(&deviceProps, devID);
	cudaCheckErrors("get device properties failed");

    printf("CUDA device [%s] has %d Multi-Processors\n",
           deviceProps.name, deviceProps.multiProcessorCount);

	sdkCreateTimer(&timer);
    sdkResetTimer(&timer);
	
	glGenBuffers(1,&pbo);
	glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
	glBufferData( GL_PIXEL_UNPACK_BUFFER_ARB, wWidth * wHeight * sizeof(RGBColor) , NULL , GL_DYNAMIC_DRAW_ARB );
	
	cudaGraphicsGLRegisterBuffer( &cuda_pbo_resource, pbo, cudaGraphicsMapFlagsNone);
	cudaCheckErrors("cudaGraphicsGLRegisterBuffer failed");
	
}

void simpleSphere(World *w,int width,int height){
	uchar3* devPtr;
	size_t size;
	cudaGraphicsMapResources(1, &cuda_pbo_resource, NULL);	
    cudaCheckErrors("cudaGraphicsMapResources failed");

	cudaGraphicsResourceGetMappedPointer( (void**)&devPtr,
												&size,
												cuda_pbo_resource );	
    cudaCheckErrors("cudaGraphicsResourceGetMappedPointer failed");
	
	cudaRayTracing(w,width,height,devPtr);
    
	cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);
    getLastCudaError("cudaGraphicsUnmapResources failed");
}

int main(int argc , char *argv[] ){
	
	int devID;
    printf("%s Starting...\n\n", sSDKname);
    printf("[%s] - [OpenGL/CUDA simulation] starting...\n", sSDKname);
	
	if (false == initGL( argc, argv))
    {
        exit(EXIT_SUCCESS);
    }
	 // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    devID = findCudaGLDevice(argc, (const char **)argv);

	//w = (World*)malloc(sizeof(World));
	initCuda(devID);

	cudaRayTracingInit(&h_w,&d_w,wWidth,wHeight);

	ViewPlane* h_vp = new ViewPlane;
	World *h_wd = new World;
	GeometricObject**h_obj = new GeometricObject*[4];
	Light **h_l = new Light*[1];
	Ambient *h_ab = new Ambient;
	cudaMemcpy(h_wd,d_w,sizeof(World),cudaMemcpyDeviceToHost);
	cudaMemcpy(h_vp,h_wd->vp,sizeof(ViewPlane),cudaMemcpyDeviceToHost);
	cudaMemcpy(h_obj,h_wd->objects,4 * sizeof(GeometricObject*),cudaMemcpyDeviceToHost);
	cudaMemcpy(h_l,h_wd->lights,1 * sizeof(Light*),cudaMemcpyDeviceToHost);
	cudaMemcpy(h_ab,h_wd->ambient, sizeof(Ambient),cudaMemcpyDeviceToHost);

	Sphere *h_s1,*h_s2,*h_s3;
	h_s1 = new Sphere;h_s2 = new Sphere;h_s3 = new Sphere;
	cudaMemcpy(h_s1,h_obj[0],sizeof(Sphere),cudaMemcpyDeviceToHost);	
	cudaMemcpy(h_s2,h_obj[1],sizeof(Sphere),cudaMemcpyDeviceToHost);	
	cudaMemcpy(h_s3,h_obj[2],sizeof(Sphere),cudaMemcpyDeviceToHost);

	Plane *h_p = new Plane;
	cudaMemcpy(h_p,h_obj[3],sizeof(Plane),cudaMemcpyDeviceToHost);

	PointLight *h_pl = new PointLight;
	cudaMemcpy(h_pl,h_l[0],sizeof(PointLight),cudaMemcpyDeviceToHost);

	glutMainLoop();

	clearup();

	cudaDeviceReset();

	return 0;
}