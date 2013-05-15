#include "Tools.cuh"
#include "defines.cuh"
#include "function_defines.cuh"


__host__ inline void sampler_PureRandom	( Sampler *sampler );
__host__ inline void sampler_Regular		( Sampler *sampler );
__host__ inline void sampler_Jittered		( Sampler *sampler );
__host__ inline void sampler_Nrooks		( Sampler *sampler );
__host__ inline void sampler_MultiJittered( Sampler *sampler );
__host__ inline void sampler_Hammersley	( Sampler *sampler );

__host__ inline
void GenerateSample(Sampler *sampler){
	switch( sampler->type ){
	case SAMPLER_PURERANDOM :
		sampler_PureRandom(sampler);
		break;
	case SAMPLER_REGULAR :
		sampler_Regular(sampler);
		break;
	case SAMPLER_JITTERED :
		sampler_Jittered(sampler);
		break;
	case SAMPLER_NROOKS :
		sampler_Nrooks(sampler);
		break;
	case SAMPLER_MULTIJITTERED :
		sampler_MultiJittered(sampler);
		break;
	case SAMPLER_HAMMERSLEY :
		sampler_Hammersley(sampler);
		break;
	default:
		break;
	}
}

__device__ inline
Point2D getSampleUnitSquare(Sampler *sampler){
	return sampler->sample[ sampler->count++ % SAMPLE_POOL_SIZE ];
}

__host__ inline
void sampler_PureRandom		( Sampler *sampler  ){
	/*not yet..*/
	for( int i = 0 ; i < sampler->numSamples ; i ++ ){
		sampler->sample[i].x = 0;
		sampler->sample[i].y = 0;
	}
}

__host__ inline 
void sampler_Regular		( Sampler *sampler  ){
	/*not yet..*/
	for( int i = 0 ; i < sampler->numSamples ; i ++ ){
		sampler->sample[i].x = 0;
		sampler->sample[i].y = 0;
	}
}

__host__ inline 
void sampler_Jittered		( Sampler *sampler  ){

	for(int j = 0 ; j < SAMPLE_POOL_SIZE / sampler->numSamples ; j ++){
		for( int i = 0 ; i < sampler->numSamples ; i ++ ){

			int dim = sqrtf( float(sampler->numSamples ) );

			int r = i / dim;
			int c = i % dim;
	
			float t = 1 / dim;

			sampler->sample[j * sampler->numSamples + i].x = float( c * t ) + floatRand()/dim;
			sampler->sample[j * sampler->numSamples + i].y = float( r * t ) + floatRand()/dim;
		}
	}
}

__host__ inline
void sampler_Nrooks			( Sampler *sampler  ){
	/*not yet..*/
	for( int i = 0 ; i < sampler->numSamples ; i ++ ){
		sampler->sample[i].x = 0;
		sampler->sample[i].y = 0;
	}
}

__host__ inline 
void sampler_MultiJittered	( Sampler *sampler ){
	/*not yet..*/
	for( int i = 0 ; i < sampler->numSamples ; i ++ ){
		sampler->sample[i].x = 0;
		sampler->sample[i].y = 0;
	}
}

__host__ inline 
void sampler_Hammersley		( Sampler *sampler ){
	/*not yet..*/
	for( int i = 0 ; i < sampler->numSamples ; i ++ ){
		sampler->sample[i].x = 0;
		sampler->sample[i].y = 0;
	}
}

__device__ inline
Point3D MapSquareToHemiSphere(Point2D point, float exp){
	float cosPhi = cosf( point.x * 2.0 * PI );
	float sinPhi = sinf( point.x * 2.0 * PI );
	float cosTheta = powf( (1.0 - point.y ), 1.0 / ( exp + 1 ));
	float sinTheta = sqrtf( 1.0 - cosTheta * cosTheta );
	float pu = sinTheta * cosTheta;
	float pv = sinTheta * sinTheta;
	float pw = cosTheta;
	return  Point3D( pu , pv, pw );
}

__device__ inline
Point3D getSampleUnitHemiSphere(Sampler *sampler, float exp){
	return MapSquareToHemiSphere( sampler->sample[ sampler->count++ % SAMPLE_POOL_SIZE ],exp );
}