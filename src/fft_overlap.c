
/* #include "turH.h" */


/* static cufftHandle fft2_r2c;  */
/* static cufftHandle fft2_c2r;  */
/* static cufftHandle fft1_c2c;  */



/* static float2* aux_host_1; */
/* static float2* aux_host_2; */

/* static float2* sum; */

/* size_t size; */


/* static const int n_steps=16; */

/* //Check */

/* static void cufftCheck( cufftResult error, const char* function ) */
/* { */
/* 	if(error != CUFFT_SUCCESS) */
/* 	{ */
/* 		printf("\n error  %s : %d \n", function, error); */
/* 		exit(1); */
/* 	} */
		
/* 	return; */
/* }   */


/* void fftSetup(void) */
/* { */

/* 	int n2[2]={NY,2*NZ-2}; */
/* 	int n1[1]={NX}; */
	
/* 	//2D fourier transforms */

/* 	cufftCheck(cufftPlanMany( &fft2_r2c,2,n2,NULL,1,0,NULL,1,0,CUFFT_R2C,NXSIZE/n_steps),"ALLOCATE_FFT2_R2C"); */
/* 	cufftCheck(cufftPlanMany( &fft2_c2r,2,n2,NULL,1,0,NULL,1,0,CUFFT_C2R,NXSIZE/n_steps),"ALLOCATE_FFT2_C2R"); */

/* 	//1D fourier transforms */

/* 	cufftCheck(cufftPlanMany(&fft1_c2c,1,n1,NULL,1,0,NULL,1,0,CUFFT_C2C,NYSIZE*NZ/n_steps),"ALLOCATE_FFT1_R2C"); */



/* 	//MALLOC aux buffer host	 */

/* 	size=NXSIZE*NY*NZ*sizeof(float2); */

/* 	aux_host_1=(float2*)malloc(size); */
/* 	aux_host_2=(float2*)malloc(size); */
			
/* 	//Set up for sum */
/* 	sum=(float2*)malloc(NXSIZE*sizeof(float2));			 */

/* 	return; */
/* } */

/* void fftDestroy(void) */
/* { */
/*  	cufftDestroy(fft2_r2c); */
/* 	cufftDestroy(fft2_c2r); */

/* 	return; */
/* } */

/* static void transposeForward(float2* buffer_1){ */

/* 	//Copy from gpu to cpu */
	
/* 	cudaCheck(cudaMemcpy(aux_host_1,buffer_1,size,cudaMemcpyDeviceToHost),"MemInfo_TranFOR"); */
	
/* 	//Transpuesta [i,j,k][NX,NY,NZ] a -----> [j,k,i][NY,NZ,NX] */

/* 	mpiCheck(chxyz2yzx((double *)aux_host_1,(double*)aux_host_2,NX,NY,NZ,RANK,SIZE),"T"); */

/* 	//Copy from gpu to cpu */
	
/* 	cudaCheck(cudaMemcpy(buffer_1,aux_host_2,size,cudaMemcpyHostToDevice),"MemInfo_TranFOR"); */

/* 	return; */
/* } */

/* static void transposeBackward(float2* buffer_1){ */

/* 	//Copy from gpu to cpu */
	
/* 	cudaCheck(cudaMemcpy(aux_host_1,buffer_1,size,cudaMemcpyDeviceToHost),"MemInfo_TranBack"); */
	
/* 	//Transpuesta [i,j,k][NX,NY,NZ] a -----> [j,k,i][NY,NZ,NX] */

/* 	mpiCheck(chyzx2xyz((double *)aux_host_1,(double*)aux_host_2,NX,NY,NZ,RANK,SIZE),"T"); */

/* 	//Copy from gpu to cpu */
	
/* 	cudaCheck(cudaMemcpy(buffer_1,aux_host_2,size,cudaMemcpyHostToDevice),"MemInfo_TranBack"); */

/* 	return; */

/* } */


/* void fftForward(float2* buffer_1) */
/* { */


/* 	//NX transformadas en 2D 	 */

/* 	for(int i=0;i<n_steps;i++){	 */

/* 	cufftCheck(cufftExecR2C(fft2_r2c,(float*)(buffer_1)+i*2*NY*NZ*NXSIZE/n_steps,buffer_1+i*NY*NZ*NXSIZE/n_steps),"forward transform"); */

/* 	}	 */

	
/* 	//Transpose */

/* 	transposeForward(buffer_1); */
	

/* 	//Transformada 1D NY*NZ */

	
/* 	for(int i=0;i<n_steps;i++){	 */

/* 	cufftCheck(cufftExecC2C(fft1_c2c,buffer_1+i*NY*NZ*NXSIZE/n_steps,buffer_1+i*NY*NZ*NXSIZE/n_steps,CUFFT_FORWARD),"forward transform");	 */

/* 	} */
	

/* 	//Transpose */

/* 	transposeBackward(buffer_1); */
		

/* 	return; */
/* } */


/* void fftBackward(float2* buffer_1) */
/* { */

/* 	//Transpose */

/* 	transposeForward(buffer_1); */

/* 	//NY*NZ transformadas 1D en X */
		
/* 	for(int i=0;i<n_steps;i++){	 */

/* 	cufftCheck(cufftExecC2C(fft1_c2c,buffer_1+i*NY*NZ*NXSIZE/n_steps,buffer_1+i*NY*NZ*NXSIZE/n_steps,CUFFT_INVERSE),"forward transform");	 */

/* 	} */

/* 	//Transpose */

/* 	transposeBackward(buffer_1); */

/* 	//NX transformadas en 2D  */

/* 	for(int i=0;i<n_steps;i++){	 */

/* 	cufftCheck(cufftExecC2R(fft2_c2r,buffer_1+i*NY*NZ*NXSIZE/n_steps,(float*)(buffer_1)+i*2*NY*NZ*NXSIZE/n_steps),"forward transform"); */
	
/* 	} */


/* 	return;	 */

/* } */

/* static cufftHandle fft2_c2c;  */
/* static cufftHandle fft1_c2r;  */
/* static cufftHandle fft1_r2c;  */

/* static float2* aux_host_1; */
/* static float2* aux_host_2; */

/* static float2* sum; */

/* size_t size; */

/* int nsteps=16; */

/* void setfftAsync(void){ */

	
/* 	int n2[2]={NX,2*NZ-2}; */
/* 	int n1[1]={NY}; */
	
/* 	//2D fourier transforms */

/* 	cufftCheck(cufftPlanMany( &fft2_r2c,2,n2,NULL,1,0,NULL,1,0,CUFFT_R2C,NYSIZE/n_steps),"ALLOCATE_FFT2_R2C"); */
/* 	cufftCheck(cufftPlanMany( &fft2_c2r,2,n2,NULL,1,0,NULL,1,0,CUFFT_C2R,NYSIZE/n_steps),"ALLOCATE_FFT2_C2R"); */

/* 	//1D fourier transforms */

/* 	cufftCheck(cufftPlanMany(&fft1_c2c,1,n1,NULL,1,0,NULL,1,0,CUFFT_C2C,NXSIZE*NZ/n_steps),"ALLOCATE_FFT1_R2C"); */

/* 	//MALLOC STREAMS */

/* 	cudaStream_t* STREAMS=(cudaStream_t*)malloc(sizeof(cudaStream_t)*NXSIZE)); */
	
/* 	//Set streams */

/* 	for(int i=0;i<NXSIZE;i++) */
/* 	cudaStreamCreate(STREAMS+i);  */


/* 		//MALLOC aux buffer host	 */

/* 	size=NXSIZE*NY*NZ*sizeof(float2); */

/* 	cudaCheck(aux_host_1,size),"malloc"); */
/* 	cudaCheck(aux_host_2,size),"malloc"); */



/* }; */

/* void fftForward_Async_1(float2* buffer_1){ */

	
/* 	for(int i=0;i<NXSIZE/nsteps;i++) */
/* 	{ */


/* 	//TRANSPOSE  */

	
/* 	//FFT 1D */

/* 	cufftCheck(cufftSetStream(cufftHandle plan,stream[i]),"SetStream"); */
/* 	cufftCheck(cufftExecC2C(fft1_c2c,buffer_1+i*NY*NZ*NXSIZE/n_steps,buffer_1+i*NY*NZ*NXSIZE/n_steps,CUFFT_BACKWARD),"forward transform");	 */

/* 	//COPY		 */

/* 	//copy data from host to device memory asynchronously */
/* 	cudaMemcpyAsync((float2*)aux_host_1+i*NY*NZ*NXSIZE/n_steps,(float2*)buffer_1+i*NY*NZ*NXSIZE/n_steps,size/n_steps,cudaMemcpyDeviceToHost,stream[i]); */

/* 	} */

/* 	//wait until all stream are finished */
/* 	cudaThreadSynchronize();  */

/* } */

/* void fftBack_Async_1D(float2* buffer_1,float2* aux_host_1){ */

	
/* 	for(int i=0;i<NXSIZE/nsteps;i++) */
/* 	{ */


/* 	//TRANSPOSE  */

	
/* 	//FFT 1D */

/* 	cufftCheck(cufftSetStream(fft1_c2c,stream[i]),"SetStream"); */
/* 	cufftCheck(cufftExecC2C(fft1_c2c,buffer_1+i*NY*NZ*NXSIZE/n_steps,buffer_1+i*NY*NZ*NXSIZE/n_steps,CUFFT_BACKWARD),"forward transform");	 */

/* 	//COPY		 */

/* 	//copy data from host to device memory asynchronously */
/* 	cudaMemcpyAsync((float2*)aux_host_1+i*NY*NZ*NXSIZE/n_steps,(float2*)buffer_1+i*NY*NZ*NXSIZE/n_steps,size/n_steps,cudaMemcpyDeviceToHost,stream[i]); */

/* 	} */

/* 	//wait until all stream are finished */
/* 	cudaThreadSynchronize();  */

/* } */

/* void fftBack_Async_2D(float2* buffer_1,float2* aux_host_1){ */

	
/* 	for(int i=0;i<NYSIZE/nsteps;i++) */
/* 	{ */

/* 	//copy data from host to device memory asynchronously */
/* 	cudaMemcpyAsync((float2*)buffer_1+i*NYSIZE*NZ*NX/n_steps,(float2*)aux_host_1+i*NYSIZE*NZ*NX/n_steps,size/n_steps,cudaMemcpyHostToDevice,stream[i]); */

	
/* 	//FFT 2D */

/* 	cufftCheck(cufftSetStream(fft2_c2r,stream[i]),"SetStream"); */
/* 	cufftCheck(cufftExecC2R(fft2_c2r,buffer_1+i*NYSIZE*NZ*NX/n_steps,(float*)(buffer_1)+i*2*NYSIZE*NZ*NX/n_steps),"forward transform"); */
	
/* 	} */

/* 	//wait until all stream are finished */
/* 	cudaThreadSynchronize();  */

/* } */


/* void fftForw_Async_2D(float2* buffer_1,float2* aux_host_1){ */

/* 	for(int i=0;i<NYSIZE/nsteps;i++) */
/* 	{ */

/* 		//FFT 2D */

/* 	cufftCheck(cufftSetStream(fft2_r2c,stream[i]),"SetStream"); */
/* 	cufftCheck(cufftExecC2R(fft2_r2c,(float*)(buffer_1)+i*2*NYSIZE*NZ*NX/n_steps,buffer_1+i*NYSIZE*NZ*NX/n_steps),"forward transform"); */

/* 	//copy data from host to device memory asynchronously */
/* 	cudaMemcpyAsync((float2*)aux_host_1+i*NYSIZE*NZ*NX/n_steps,bytes,(float2*)buffer_1+i*NYSIZE*NZ*NX/n_steps,bytes,cudaMemcpyDeviceToHost,stream[i]); */

	
/* 	//FFT 2D */

/* 	cufftCheck(cufftSetStream(fft2_c2r,stream[i]),"SetStream"); */
/* 	cufftCheck(cufftExecC2R(fft2_c2r,buffer_1+i*NYSIZE*NZ*NX/n_steps,(float*)(buffer_1)+i*2*NYSIZE*NZ*NX/n_steps),"forward transform"); */
	
/* 	} */

/* 	//wait until all stream are finished */
/* 	cudaThreadSynchronize();  */

/* } */

/* void fftForw_Async_1D(float2* buffer_1,float2* aux_host_1){ */


/* 	for(int i=0;i<NXSIZE/nsteps;i++) */
/* 	{ */

/* 	//COPY		 */

/* 	//copy data from host to device memory asynchronously */
/* 	cudaMemcpyAsync((float2*)buffer_1+i*NY*NZ*NXSIZE/n_steps,(float2*)aux_host_1+i*NY*NZ*NXSIZE/n_steps,size/n_steps,cudaMemcpyHostToDevice,stream[i]); */

	
/* 	//FFT 1D */

/* 	cufftCheck(cufftSetStream(fft1_c2c,stream[i]),"SetStream"); */
/* 	cufftCheck(cufftExecC2C(fft1_c2c,buffer_1+i*NY*NZ*NXSIZE/n_steps,buffer_1+i*NY*NZ*NXSIZE/n_steps,CUFFT_BACKWARD),"forward transform");	 */

/* 	//TRANSPOSE  */
	
/* 	} */

/* 	//wait until all stream are finished */
/* 	cudaThreadSynchronize();  */

/* } */






