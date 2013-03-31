#include <GL\glew.h>
#include <GL\freeglut.h>

// Includes
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// CUDA standard includes
//#include <cuda_runtime.h>
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

#include "simpleSphere.cuh"

const char *sSDKname = "simpleSphere";

static int wWidth = 512;
static int wHeight = 512;

static int fpsCount = 0;
static int fpsLimit = 1;
StopWatchInterface *timer = NULL;

GLuint pbo = 0;		// OpenGL pixel buffer object
struct cudaGraphicsResource *cuda_pbo_resource; // handles OpenGL-CUDA exchange
World *w;

extern "C" void whiteNoise(int x, int y);
extern "C" void simpleSphere(World *w,int width,int height);


void clearup();
void reshape(int x, int y);

void display(void){

	sdkStartTimer(&timer);	
	
	glClearColor(0,0,0,1);

	glClear(GL_COLOR_BUFFER_BIT);
	
	simpleSphere(w,wWidth,wHeight);
	//whiteNoise(512,512);

	 // render points from vertex buffer
	glDrawPixels(wWidth,wHeight,GL_RGBA,GL_UNSIGNED_BYTE,0);
	    
	fpsCount++;
	sdkStopTimer(&timer);
    glutSwapBuffers();

	if (fpsCount == fpsLimit)
    {
        char fps[256];
        float ifps = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
        sprintf(fps, "Cuda/GL Simple Sphere (%d x %d): %3.1f fps", wWidth, wHeight, ifps);
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
    glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE);
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
	glBufferData( GL_PIXEL_UNPACK_BUFFER_ARB, wWidth * wHeight * sizeof(RGBAColor) , NULL , GL_DYNAMIC_DRAW_ARB );
	
	cudaGraphicsGLRegisterBuffer( &cuda_pbo_resource, pbo, cudaGraphicsMapFlagsNone);
	cudaCheckErrors("cudaGraphicsGLRegisterBuffer failed");
	
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

	simpleSphere_init(&w,wWidth,wHeight);

	World h_w;
	cudaMemcpy(&h_w,w,sizeof(World),cudaMemcpyDeviceToHost);
	cudaCheckErrors("h_w copy failed");
/*
	ViewPlane h_vp;
	cudaMemcpy(&h_vp,h_w.vp,sizeof(ViewPlane),cudaMemcpyDeviceToHost);
	cudaCheckErrors("h_vp copy failed");
	h_w->vp = h_vp;

	GeometricObject **object = (GeometricObject**)malloc(h_w.numObject * sizeof(GeometricObject*));
	cudaMemcpy(object,h_w.object, h_w.numObject * sizeof(GeometricObject*) ,cudaMemcpyDeviceToHost);
	cudaCheckErrors("object copy failed");
	h_w.object = object;
	
	for( int i = 0 ; i < h_w.numObject ; ++i){
		GeometricObject temp;
		cudaMemcpy(&temp, *(h_w.object + i) , sizeof(GeometricObject) , cudaMemcpyDeviceToHost);
		cudaCheckErrors("%d object copy failed",i);

		Sphere *h_s = (Sphere*)malloc(sizeof(Sphere));
		Plane *h_p = (Plane*)malloc(sizeof(Plane));
		switch( temp.type ){
		case GMO_TYPE_SPHERE:
			cudaMemcpy(h_s,  *(h_w.object + i) , sizeof(Sphere) , cudaMemcpyDeviceToHost);
			*(h_w.object + i) = (GeometricObject*) h_s;
			free(h_p);
			break;
		case GMO_TYPE_PLANE:
			cudaMemcpy(&h_p,  *(h_w.object + i) , sizeof(Plane) , cudaMemcpyDeviceToHost);
			*(h_w.object + i) = (GeometricObject*) h_p;			
			free(h_s);
			break;
		default:
			free(h_s);free(h_p);
			break;
		}
	}*/

	glutMainLoop();

	clearup();

	cudaDeviceReset();

	return 0;
}