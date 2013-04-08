#include "Tools.cuh"
#include "defines.cuh"
#include "function_defines.cuh"


__device__ inline void sampler_PureRandom	( Point2D *p, int Idx, int num );
__device__ inline void sampler_Regular		( Point2D *p, int Idx, int num );
__device__ inline void sampler_Jittered		( Point2D *p, int Idx, int num );
__device__ inline void sampler_Nrooks		( Point2D *p, int Idx, int num );
__device__ inline void sampler_MultiJittered( Point2D *p, int Idx, int num );
__device__ inline void sampler_Hammersley	( Point2D *p, int Idx, int num );

__device__ inline
Point2D getSampleUnitSquare(SamplerType type, int Idx, SampleScale scale){
	Point2D result;
	int num = getSampleNum(scale);
	switch( type ){
	case SAMPLER_PURERANDOM :
		sampler_PureRandom(&result,Idx,num);
		break;
	case SAMPLER_REGULAR :
		sampler_Regular(&result,Idx,num);
		break;
	case SAMPLER_JITTERED :
		sampler_Jittered(&result,Idx,num);
		break;
	case SAMPLER_NROOKS :
		sampler_Nrooks(&result,Idx,num);
		break;
	case SAMPLER_MULTIJITTERED :
		sampler_MultiJittered(&result,Idx,num);
		break;
	case SAMPLER_HAMMERSLEY :
		sampler_Hammersley(&result,Idx,num);
		break;
	default:
		result = make_float2(0,0);
		break;
	}
	return result;
}

__device__ inline
void sampler_PureRandom		( Point2D *p, int Idx, int num ){
	/*not yet..*/
	p->x = 0;
	p->y = 0;
}

__device__ inline 
void sampler_Regular		( Point2D *p,int Idx, int num ){
	/*not yet..*/
	p->x = 0;
	p->y = 0;
}

__device__ inline 
void sampler_Jittered		( Point2D *p, int Idx, int num ){
	SampleScale scale = getSampleScale(num);
	int dim = sqrtf( float(num) );

	int r = Idx / dim;
	int c = Idx % dim;

	float t = 1 / dim;

	p->x = float( c * t ) + floatRand()/dim;
	p->y = float( r * t ) + floatRand()/dim;

}

__device__ inline
void sampler_Nrooks			( Point2D *p,int Idx, int num ){
	/*not yet..*/
	p->x = 0;
	p->y = 0;
}

__device__ inline 
void sampler_MultiJittered	( Point2D *p,int Idx, int num ){
	/*not yet..*/
	p->x = 0;
	p->y = 0;
}

__device__ inline 
void sampler_Hammersley		( Point2D *p,int Idx, int num ){
	/*not yet..*/
	p->x = 0;
	p->y = 0;
}

__device__ __host__ inline
int getSampleNum(SampleScale scale){
	switch(scale){
	case SAMPLE_SCALE_1:
		return 1;
	case SAMPLE_SCALE_4:
		return 4;
	case SAMPLE_SCALE_16:
		return 16;
	case SAMPLE_SCALE_64:
		return 64;
	case SAMPLE_SCALE_256:
		return 256;
	default:
		return 1;
	}
}

__device__ __host__ inline
SampleScale getSampleScale(int num){
	if( num <= 1 ){
		return SAMPLE_SCALE_1;
	}
	else if( num > 1 && num <= 4 ){
		return SAMPLE_SCALE_4;
	}
	else if( num > 4 && num <= 16){
		return SAMPLE_SCALE_16;
	}
	else if( num > 16 && num <= 64){
		return SAMPLE_SCALE_64;
	}
	else if( num > 64 && num <= 256 ){
		return SAMPLE_SCALE_256;
	}
	else{
		return SAMPLE_SCALE_1;
	}
}