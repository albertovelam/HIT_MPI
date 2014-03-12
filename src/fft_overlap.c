
#include "turH.h"

static cufftHandle fft1_c2c; 
static cufftHandle fft2_c2r; 
static cufftHandle fft2_r2c; 

static float2* aux_host_1;
static float2* aux_host_2;

static float2* sum;

static size_t size;

static int n_steps=16;
cudaStream_t* STREAMS;

//Check

static void cufftCheck( cufftResult error, const char* function )
{
	if(error != CUFFT_SUCCESS)
	{
		printf("\n error  %s : %d \n", function, error);
		exit(1);
	}
		
	return;
}  

void setfftAsync(void){

	
	int n2[2]={NX,2*NZ-2};
	int n1[1]={NY};
	
	//2D fourier transforms

	cufftCheck(cufftPlanMany( &fft2_r2c,2,n2,NULL,1,0,NULL,1,0,CUFFT_R2C,NYSIZE/n_steps),"ALLOCATE_FFT2_R2C");
	cufftCheck(cufftPlanMany( &fft2_c2r,2,n2,NULL,1,0,NULL,1,0,CUFFT_C2R,NYSIZE/n_steps),"ALLOCATE_FFT2_C2R");

	//1D fourier transforms

	cufftCheck(cufftPlanMany(&fft1_c2c,1,n1,NULL,1,0,NULL,1,0,CUFFT_C2C,NXSIZE*NZ/n_steps),"ALLOCATE_FFT1_R2C");

	//MALLOC STREAMS

	STREAMS=(cudaStream_t*)malloc(sizeof(cudaStream_t)*NXSIZE/n_steps);
	
	//Set streams

	for(int i=0;i<NXSIZE/n_steps;i++)
	cudaStreamCreate(STREAMS+i); 

	//MALLOC aux buffer host	

	size=NXSIZE*NY*NZ*sizeof(float2);
	
	//MALLOC PINNED MEMORY TO ALLOW OVERLAPPING

	cudaCheck(cudaMallocHost((void**)aux_host_1,size),"malloc");
	cudaCheck(cudaMallocHost((void**)aux_host_2,size),"malloc");



};

void fftBack_Async_1D(float2* buffer_1,float2* buffer_host){

	
	for(int i=0;i<NXSIZE/n_steps;i++)
	{

	//TRANSPOSE 

	
	//FFT 1D

	cufftCheck(cufftSetStream(fft1_c2c,STREAMS[i]),"SetStream");
	cufftCheck(cufftExecC2C(fft1_c2c,buffer_1+i*NY*NZ*NXSIZE/n_steps,buffer_1+i*NY*NZ*NXSIZE/n_steps,CUFFT_INVERSE),"forward transform");	

	//COPY		

	//copy data from host to device memory asynchronously
	cudaMemcpyAsync((float2*)buffer_host+i*NY*NZ*NXSIZE/n_steps,(float2*)buffer_1+i*NY*NZ*NXSIZE/n_steps,size/n_steps,cudaMemcpyDeviceToHost,STREAMS[i]);

	}

	//wait until all stream are finished
	cudaThreadSynchronize(); 

}


void fftBack_Async_2D(float2* buffer_1,float2* buffer_host){

	
	for(int i=0;i<NXSIZE/n_steps;i++)
	{

	//copy data from host to device memory asynchronously
	cudaMemcpyAsync((float2*)buffer_1+i*NYSIZE*NZ*NX/n_steps,(float2*)buffer_host+i*NYSIZE*NZ*NX/n_steps,size/n_steps,cudaMemcpyHostToDevice,STREAMS[i]);

	
	//FFT 2D

	cufftCheck(cufftSetStream(fft2_c2r,STREAMS[i]),"SetStream");
	cufftCheck(cufftExecC2R(fft2_c2r,buffer_1+i*NYSIZE*NZ*NX/n_steps,(float*)(buffer_1)+i*2*NYSIZE*NZ*NX/n_steps),"forward transform");
	
	}

	//wait until all stream are finished
	cudaThreadSynchronize(); 

}

void fftForw_Async_1D(float2* buffer_1,float2* buffer_host){


	for(int i=0;i<NXSIZE/n_steps;i++)
	{

	//COPY		

	//copy data from host to device memory asynchronously
	cudaMemcpyAsync((float2*)buffer_1+i*NY*NZ*NXSIZE/n_steps,(float2*)buffer_host+i*NY*NZ*NXSIZE/n_steps,size/n_steps,cudaMemcpyHostToDevice,STREAMS[i]);

	
	//FFT 1D

	cufftCheck(cufftSetStream(fft1_c2c,STREAMS[i]),"SetStream");
	cufftCheck(cufftExecC2C(fft1_c2c,buffer_1+i*NY*NZ*NXSIZE/n_steps,buffer_1+i*NY*NZ*NXSIZE/n_steps,CUFFT_FORWARD),"forward transform");	

	//TRANSPOSE 

	//GET A FUCKING TRANPOSE ROUTINE	

	}

	//wait until all stream are finished
	cudaThreadSynchronize(); 

}

void fftForw_Async_2D(float2* buffer_1,float2* buffer_host){

	for(int i=0;i<NXSIZE/n_steps;i++)
	{
	
	//FFT 2D

	cufftCheck(cufftSetStream(fft2_c2r,STREAMS[i]),"SetStream");
	cufftCheck(cufftExecC2R(fft2_c2r,buffer_1+i*NYSIZE*NZ*NX/n_steps,(float*)(buffer_1)+i*2*NYSIZE*NZ*NX/n_steps),"forward transform");

	//copy data from host to device memory asynchronously
	cudaMemcpyAsync((float2*)buffer_host+i*NYSIZE*NZ*NX/n_steps,(float2*)buffer_1+i*NYSIZE*NZ*NX/n_steps,size/n_steps,cudaMemcpyDeviceToHost,STREAMS[i]);
	

	}

	//wait until all stream are finished
	cudaThreadSynchronize(); 

}



void fftForwardAsyn(float2* buffer_1){


	//FFT 1D copy to CPU
	// Local transpose [i j k]-->[i k j]
	fftForw_Async_1D(buffer_1,aux_host_1);


	//Transpuesta [i,k,j][NX,NZ,NY] a -----> [j,i,k][NY,NX,NZ]
	//Transpose to the right

	mpiCheck(chyzx2xyz((double *)aux_host_1,(double*)aux_host_2,NY,NX,NZ,RANK,SIZE),"T");

	//Copy to GPU and FFT 2D 
		
	fftForw_Async_2D(buffer_1,aux_host_2);



}


void fftBackwardAsyn(float2* buffer_1){


	//Copy to GPU and FFT 2D 
		
	fftBack_Async_2D(buffer_1,aux_host_1);


	//Transpuesta [j,i,k][NX,NZ,NY] a -----> [i,k,j][NY,NX,NZ]
	//Transpose to the left

	mpiCheck(chxyz2yzx((double *)aux_host_1,(double*)aux_host_2,NY,NX,NZ,RANK,SIZE),"T");


	// copy to GPU and FFT 1D
	// Local transpose [i j k]-->[i k j]
	fftForw_Async_1D(buffer_1,aux_host_2);

	
}





