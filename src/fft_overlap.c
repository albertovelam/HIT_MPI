
#include "turH.h"
#include <cublas_v2.h>

static cufftHandle fft1_c2c; 
static cufftHandle fft2_c2r; 
static cufftHandle fft2_r2c; 

static float2* aux_host_1[6];
static float2* aux_host_2[6];

static float2* aux_host1;
static float2* aux_host2;
static float2* aux_host3;
static float2* aux_host4;
static float2* aux_host5;
static float2* aux_host6;

static float2* aux_host11;
static float2* aux_host22;
static float2* aux_host33;
static float2* aux_host44;
static float2* aux_host55;
static float2* aux_host66;


static float2* buffer[6];

static float2* aux_device_1;


static cublasHandle_t cublasHandle;
static float2 alpha[1];
static float2 betha[1];

static size_t size;
static float2* sum;

static cudaStream_t STREAMS[6];


//Check

static void cublasCheck(cublasStatus_t error, const char* function )
{
	if(error !=  CUBLAS_STATUS_SUCCESS)
	{
		printf("\n error  %s : %d \n", function, error);
		exit(1);
	}
		
	return;
}  

static void cufftCheck( cufftResult error, const char* function )
{
	if(error != CUFFT_SUCCESS)
	{
		printf("\n error  %s : %d \n", function, error);
		exit(1);
	}
		
	return;
}  


void setFftAsync(void){

	
	int n2[2]={NX,2*NZ-2};
	int n1[1]={NY};
	
	//2D fourier transforms

	cufftCheck(cufftPlanMany( &fft2_r2c,2,n2,NULL,1,0,NULL,1,0,CUFFT_R2C,NYSIZE),"ALLOCATE_FFT2_R2C");
	cufftCheck(cufftPlanMany( &fft2_c2r,2,n2,NULL,1,0,NULL,1,0,CUFFT_C2R,NYSIZE),"ALLOCATE_FFT2_C2R");

	//1D fourier transforms

	cufftCheck(cufftPlanMany(&fft1_c2c,1,n1,NULL,1,0,NULL,1,0,CUFFT_C2C,NXSIZE*NZ),"ALLOCATE_FFT1_R2C");

	//Set streams

	for(int i=0;i<6;i++){
	cudaCheck(cudaStreamCreate(&STREAMS[i]),"create_streams"); 
	}


	//MALLOC aux buffer host	

	size=NXSIZE*NY*NZ*sizeof(float2);
	
	//MALLOC PINNED MEMORY TO ALLOW OVERLAPPING

	//for(int i=0;i<6;i++){
	//cudaCheck(cudaHostAlloc((void**)&aux_host_1[i],size,cudaHostAllocWriteCombined  ),"malloc_1");
	//cudaCheck(cudaHostAlloc((void**)&aux_host_2[i],size,cudaHostAllocWriteCombined  ),"malloc_2");
	//}


	/*
	cudaCheck(cudaHostAlloc((void**)&aux_host1,size,cudaHostAllocWriteCombined ),"malloc_1");
	cudaCheck(cudaHostAlloc((void**)&aux_host2,size,cudaHostAllocWriteCombined ),"malloc_2");
	*/

	aux_host1=(float2*)malloc(size);
	aux_host2=(float2*)malloc(size);	

	/*
	cudaCheck(cudaHostAlloc((void**)&aux_host_1[1],size,cudaHostAllocWriteCombined ),"malloc_1");
	cudaCheck(cudaHostAlloc((void**)&aux_host_2[1],size,cudaHostAllocWriteCombined ),"malloc_2");

	cudaCheck(cudaHostAlloc((void**)&aux_host_1[2],size,cudaHostAllocWriteCombined ),"malloc_1");
	cudaCheck(cudaHostAlloc((void**)&aux_host_2[2],size,cudaHostAllocWriteCombined ),"malloc_2");

	cudaCheck(cudaHostAlloc((void**)&aux_host_1[3],size,cudaHostAllocWriteCombined ),"malloc_1");
	cudaCheck(cudaHostAlloc((void**)&aux_host_2[3],size,cudaHostAllocWriteCombined ),"malloc_2");

	cudaCheck(cudaHostAlloc((void**)&aux_host_1[4],size,cudaHostAllocWriteCombined ),"malloc_1");
	cudaCheck(cudaHostAlloc((void**)&aux_host_2[4],size,cudaHostAllocWriteCombined ),"malloc_2");

	cudaCheck(cudaHostAlloc((void**)&aux_host_1[5],size,cudaHostAllocWriteCombined  ),"malloc_1");
	cudaCheck(cudaHostAlloc((void**)&aux_host_2[5],size,cudaHostAllocWriteCombined  ),"malloc_2");

	cudaCheck(cudaHostAlloc((void**)&aux_host1,size,cudaHostAllocWriteCombined ),"malloc_1");
	cudaCheck(cudaHostAlloc((void**)&aux_host2,size,cudaHostAllocWriteCombined  ),"malloc_2");
	
	cudaCheck(cudaHostAlloc((void**)&aux_host3,size,cudaHostAllocWriteCombined  ),"malloc_1");
	cudaCheck(cudaHostAlloc((void**)&aux_host4,size,cudaHostAllocWriteCombined  ),"malloc_2");

	cudaCheck(cudaHostAlloc((void**)&aux_host5,size,cudaHostAllocWriteCombined  ),"malloc_1");
	cudaCheck(cudaHostAlloc((void**)&aux_host6,size,cudaHostAllocWriteCombined  ),"malloc_2");

	cudaCheck(cudaHostAlloc((void**)&aux_host11,size,cudaHostAllocWriteCombined  ),"malloc_1");
	cudaCheck(cudaHostAlloc((void**)&aux_host22,size,cudaHostAllocWriteCombined  ),"malloc_2");
	
	cudaCheck(cudaHostAlloc((void**)&aux_host33,size,cudaHostAllocWriteCombined  ),"malloc_1");
	cudaCheck(cudaHostAlloc((void**)&aux_host44,size,cudaHostAllocWriteCombined  ),"malloc_2");

	cudaCheck(cudaHostAlloc((void**)&aux_host55,size,cudaHostAllocWriteCombined  ),"malloc_1");
	cudaCheck(cudaHostAlloc((void**)&aux_host66,size,cudaHostAllocWriteCombined  ),"malloc_2");
*/

	//MALLOC CUDA
	cudaCheck(cudaMalloc((void**)&aux_device_1,size),"malloc_device");
	

	//SET TRANSPOSE
		
	cublasCheck(cublasCreate(&cublasHandle),"Cre");

	alpha[0].x=1.0f;
	alpha[0].y=0.0f;

}

void transpose_A(float2* u_2,float2* u_1){

	//[NY,NZ]--->[NZ,NY]

	for(int i=0;i<NXSIZE;i++){
	cublasCheck(cublasCgeam(cublasHandle,CUBLAS_OP_T,CUBLAS_OP_T,NY,NZ,alpha,(const float2*)u_1+i*NY*NZ,NZ,0,0,NZ,(float2*)u_2+i*NY*NZ,NY),"Tr");
	}

	return;


}

void transpose_B(float2* u_2,float2* u_1){

	//[NZ,NY]--->[NY,NZ]

	for(int i=0;i<NXSIZE;i++){
	cublasCheck(cublasCgeam(cublasHandle,CUBLAS_OP_T,CUBLAS_OP_T,NZ,NY,alpha,(const float2*)u_1+i*NY*NZ,NY,0,0,NY,(float2*)u_2+i*NY*NZ,NZ),"Tr");
	}

	return;

}

void fftBack1T(float2* u1){

	//Transpose from [x,y,z] to [x,z,y]

	transpose_A(aux_device_1,u1);
	
	//FFT 1D on Y
		
	cufftCheck(cufftExecC2C(fft1_c2c,aux_device_1,u1,CUFFT_INVERSE),"forward transform");

	cudaCheck(cudaMemcpy((float2*)aux_host1,(float2*)u1,size,cudaMemcpyDeviceToHost),"copy");

	//Transpose from [x,z,y] to [y,x,z]
	
	mpiCheck(chyzx2xyz((double *)aux_host1,(double*)aux_host2,NY,NX,NZ,RANK,SIZE),"T");

	cudaCheck(cudaMemcpy((float2*)u1,(float2*)aux_host2,size,cudaMemcpyHostToDevice),"copy");

	//FFT 2D on X	

	cufftCheck(cufftExecC2R(fft2_c2r,u1,(float*)u1),"forward transform");


}


void fftForw1T(float2* u1){

		//FFT 2D
	
		cufftCheck(cufftExecR2C(fft2_r2c,(float*)u1,(float2*)u1),"forward transform");

		cudaCheck(cudaMemcpy((float2*)aux_host1,(float2*)u1,size,cudaMemcpyDeviceToHost),"copy");
			
		//Transpose from [y,x,z] to [x,z,y]
		
		mpiCheck(chxyz2yzx((double *)aux_host1,(double*)aux_host2,NY,NX,NZ,RANK,SIZE),"T");

		
		cudaCheck(cudaMemcpy((float2*)u1,(float2*)aux_host2,size,cudaMemcpyHostToDevice),"copy");

		//FFT 1D

		cufftCheck(cufftExecC2C(fft1_c2c,u1,aux_device_1,CUFFT_FORWARD),"forward transform");	 

		//Transpose from [x,z,y] to [x,y,z]		

		transpose_B(u1,aux_device_1);

}

void fftBackMultiple(float2* u1,float2* u2,float2* u3,float2* u4,float2* u5,float2* u6){

		//Handle buffers

		buffer[0]=u1;
		buffer[1]=u2;
		buffer[2]=u3;
		buffer[3]=u4;
		buffer[4]=u5;
		buffer[5]=u6;

		//FIRST SIX FFTS 1D


		for(int j=0;j<6;j++){

		//Transpose
		cublasCheck(cublasSetStream(cublasHandle,STREAMS[j]),"stream");
		transpose_A(aux_device_1,buffer[j]);

	
		//FFT 1D

		cufftCheck(cufftSetStream(fft1_c2c,STREAMS[j]),"SetStream");
		cufftCheck(cufftExecC2C(fft1_c2c,aux_device_1,buffer[j],CUFFT_INVERSE),"forward transform");	
		cudaCheck(cudaMemcpyAsync((float2*)aux_host_1[j],(float2*)buffer[j],size,cudaMemcpyDeviceToHost,STREAMS[j]),"copy");
		
		}


		//COPY TO CPU
		
		

		for(int j=0;j<6;j++){

		cudaCheck(cudaStreamSynchronize(STREAMS[j]),"event_synchronise"); 	
		mpiCheck(chyzx2xyz((double *)aux_host_1[j],(double*)aux_host_2[j],NY,NX,NZ,RANK,SIZE),"T");
 

		cudaCheck(cudaMemcpyAsync((float2*)buffer[j],(float2*)aux_host_2[j],size,cudaMemcpyHostToDevice,STREAMS[j]),"copy");
	
		cufftCheck(cufftSetStream(fft2_c2r,STREAMS[j]),"SetStream");
		cufftCheck(cufftExecC2R(fft2_c2r,buffer[j],(float*)buffer[j]),"forward transform");

		}

		//Device synchronise

		cudaCheck(cudaDeviceSynchronize(),"synchro"); 
		

		return;

}


void fftForwMultiple(float2* u1,float2* u2,float2* u3){


		//Handle buffers

		buffer[0]=u1;
		buffer[1]=u2;
		buffer[2]=u3;


		//FIRST THREE FFTS 2D

		

		for(int j=0;j<3;j++){

		//FFT 2D

		cufftCheck(cufftSetStream(fft2_r2c,STREAMS[j]),"SetStream");		
		cufftCheck(cufftExecR2C(fft2_r2c,(float*)buffer[j],(float2*)buffer[j]),"forward transform");

		cudaCheck(cudaMemcpyAsync((float2*)aux_host_1[j],(float2*)buffer[j],size,cudaMemcpyDeviceToHost,STREAMS[j]),"copy");
		
		}


		
		for(int j=0;j<3;j++){

		cudaCheck(cudaStreamSynchronize(STREAMS[j]),"event_synchronise"); 
		
		mpiCheck(chxyz2yzx((double *)aux_host_1[j],(double*)aux_host_2[j],NY,NX,NZ,RANK,SIZE),"T");

		//Copy to gpu
		cudaCheck(cudaMemcpyAsync((float2*)buffer[j],(float2*)aux_host_2[j],size,cudaMemcpyHostToDevice,STREAMS[j]),"copy"); 		

		//FFT 1D
		cufftCheck(cufftSetStream(fft1_c2c,STREAMS[j]),"SetStream");
		cufftCheck(cufftExecC2C(fft1_c2c,buffer[j],aux_device_1,CUFFT_FORWARD),"forward transform");	 
		
		//Transpose
		cublasCheck(cublasSetStream(cublasHandle,STREAMS[j]),"stream");
		transpose_A(buffer[j],aux_device_1);
		
		}

		//Device synchronise

		cudaCheck(cudaDeviceSynchronize(),"synchro"); 
		

		return;


}





void calcUmaxV2(vectorField t,float* ux,float* uy,float* uz)
{


	int size_l=2*NXSIZE*NY*NZ;
	int index;

	cublasCheck(cublasIsamax (cublasHandle,size_l, (const float *)ux,1,&index),"Isa");
	cudaCheck(cudaMemcpy(&ux,(float*)ux+index-1, sizeof(float), cudaMemcpyDeviceToHost),"MemInfo_isa");
	
	cublasCheck(cublasIsamax (cublasHandle,size_l, (const float *)uy,1,&index),"Isa");
	cudaCheck(cudaMemcpy(&uy,(float*)uy+index-1, sizeof(float), cudaMemcpyDeviceToHost),"MemInfo_isa");
	
	cublasCheck(cublasIsamax (cublasHandle,size_l, (const float *)uz,1,&index),"Isa");
	cudaCheck(cudaMemcpy(&uz,(float*)uz+index-1, sizeof(float), cudaMemcpyDeviceToHost),"MemInfo_isa");

	
	*ux=abs(*ux);
	*uy=abs(*uy);
	*uz=abs(*uz);
	

	//MPI reduce
	reduceMAX(ux,uy,uz);

	return;

}



float sumElementsV2(float2* buffer_1){

	//destroza lo que haya en el buffer

	float sum_all=0;
	

	cufftCheck(cufftExecR2C(fft2_r2c,(float*)(buffer_1),buffer_1),"forward transform");


	for(int i=0;i<NXSIZE;i++){

	cudaCheck(cudaMemcpy((float2*)sum+i,(float2*)buffer_1+i*NY*NZ,sizeof(float2),cudaMemcpyDeviceToHost),"MemInfo1");

	};
	
	for(int i=1;i<NXSIZE;i++){

	sum[0].x+=sum[i].x;
	}

	//MPI SUM

	reduceSUM((float*)sum,&sum_all);


	return sum_all;

};



