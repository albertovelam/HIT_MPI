
#include"turH.h"
#include<cublas_v2.h>

static float2 alpha[1];
static cublasHandle_t cublasHandle;

static float2* aux_host1;
static float2* aux_host2;

static size_t size;
static int MPIErr;

static void cublasCheck(cublasStatus_t error, const char* function )
{
	if(error !=  CUBLAS_STATUS_SUCCESS)
	{
		printf("\n error  %s : %d \n", function, error);
		exit(1);
	}
		
	return;
}  

void setTransposeCudaMpi(void){


	cublasCheck(cublasCreate(&cublasHandle),"Cre");

	alpha[0].x=1.0f;
	alpha[0].y=0.0f;

	size=NXSIZE*NY*NZ*sizeof(float2);
		
	aux_host1=(float2*)malloc(size);
	aux_host2=(float2*)malloc(size);	

        cudaHostRegister(aux_host1,size,0);
        cudaHostRegister(aux_host2,size,0);

	return;
}

//Transpose [Ny,Nx] a [Nx,Ny]

static void transpose(float2* u_2,const float2* u_1,int Nx,int Ny){

	//Transpuesta de [i,k,j][NX,NZ,NY] a -----> [j,i,k][NY,NX,NZ]

	cublasCheck(cublasCgeam(cublasHandle,CUBLAS_OP_T,CUBLAS_OP_T,Ny,Nx,alpha,(const float2*)u_1,Nx,0,0,Nx,(float2*)u_2,Ny),"Tr_1");
	//printf("\n%f,%f",alpha[0].x,alpha[0].y);
	return;


}

//Transpose [Ny,Nx] a [Nx,Ny]

static void transposeBatched(float2* u_2,const float2* u_1,int Nx,int Ny,int batch){

	//Transpuesta de [i,k,j][NX,NZ,NY] a -----> [j,i,k][NY,NX,NZ]

	for(int nstep=0;nstep<batch;nstep++){
	
	int stride=nstep*Nx*Ny;

	cublasCheck(cublasCgeam(cublasHandle,CUBLAS_OP_T,CUBLAS_OP_T,Ny,Nx,alpha,(const float2*)u_1+stride,Nx,0,0,Nx,(float2*)u_2+stride,Ny),"Tr_2");

	}	
	
	//printf("\n%f,%f",alpha[0].x,alpha[0].y);
	return;


}

void transposeXYZ2YZX(float2* u1,int Nx,int Ny,int Nz,int rank,int sizeMpi){

	 int myNx = Nx/sizeMpi;
  	 int myNy = Ny/sizeMpi;

	//Transpose [NXISZE,NY,NZ] ---> [NY,myNx,NZ]

	transposeBatched(AUX,(const float2*)u1,Nz,NY,myNx);
	transpose(u1,(const float2*)AUX,NY,myNx*Nz);

	//COPY TO HOST
	cudaCheck(cudaMemcpy((float2*)aux_host1,(float2*)u1,size,cudaMemcpyDeviceToHost),"copy");

	
	MPIErr = MPI_Alltoall(aux_host1,Nz*myNx*myNy,MPI_DOUBLE,
			      aux_host2,Nz*myNx*myNy,MPI_DOUBLE,
			      MPI_COMM_WORLD);

	mpiCheck(MPIErr,"transpoze");

	//COPY TO DEVICE
	cudaCheck(cudaMemcpy((float2*)AUX,(float2*)aux_host2,size,cudaMemcpyHostToDevice),"copy");

	//Transpose [sizeMpi,myNy,myNx,Nz] ---> [myNy,Nz,sizeMpi,myNx]

	transposeBatched(u1,(const float2*)AUX,myNx*Nz,myNy,sizeMpi);
	transposeBatched(AUX,(const float2*)u1,myNy,Nz,sizeMpi*myNx);
	transpose(u1,(const float2*)AUX,myNy*Nz,sizeMpi*myNx);



	
}	

void transposeYZX2XYZ(float2* u1,int Nx,int Ny,int Nz,int rank,int sizeMpi){

	 int myNx = Nx/sizeMpi;
  	 int myNy = Ny/sizeMpi;


	//Transpose [myNy,Nz,sizeMpi,myNx] ---> [sizeMpi,NYISZE,myNx,Nz]

	transpose(AUX,(const float2*)u1,sizeMpi*myNx,myNy*Nz);
	transposeBatched(u1,(const float2*)AUX,myNy*Nz,myNx,sizeMpi);
	transposeBatched(AUX,(const float2*)u1,myNx,Nz,sizeMpi*myNy);
	
	//COPY TO HOST
	cudaCheck(cudaMemcpy((float2*)aux_host1,(float2*)AUX,size,cudaMemcpyDeviceToHost),"copy");
	
 	/* Communications */
 	MPIErr = MPI_Alltoall(aux_host1,Nz*myNx*myNy,MPI_DOUBLE,
			      aux_host2,Nz*myNx*myNy,MPI_DOUBLE,
			      MPI_COMM_WORLD);
	
	mpiCheck(MPIErr,"transpoze");	

	//COPY TO DEVICE
	cudaCheck(cudaMemcpy((float2*)u1,(float2*)aux_host2,size,cudaMemcpyHostToDevice),"copy");
	

	//Transpose [NY,myNx,Nz]--->[NXISZE,NY,Nz]  

	transpose(AUX,(const float2*)u1,myNx*Nz,NY);
	transposeBatched(u1,(const float2*)AUX,NY,Nz,myNx);
	


	return;
	
}




