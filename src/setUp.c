
#include "turH.h"

static vectorField u;
static vectorField u_host;


int read_parallel_double(char *filename, double *x, int Nx, int Ny, int Nz,
			 int rank, int size);
int wrte_parallel_double(char *filename, double *x, int Nx, int Ny, int Nz,
			 int rank, int size);

void setUp(void){

	//1 NODO

	//CUDA DEVICE

	int Ndevices;	
	int device;
	
	cudaCheck(cudaGetDeviceCount(&Ndevices),"Set");  			
	
	/*
	if(SIZE%Ndevices!=0){
	printf("Error_ndevices");
	exit(1);
	}

	int A=SIZE/Ndevices;
	*/	

	cudaCheck(cudaSetDevice(RANK%Ndevices),"Set");
	
	//Setups
	fftSetup();
	RK2setup();
			
	return;

}

void starSimulation(void){

	//Size 
		
	size_t size=NXSIZE*NY*NZ*sizeof(float2);


	// Memory Alloc	

	u_host.x=(float2*)malloc(size);
	u_host.y=(float2*)malloc(size);
	u_host.z=(float2*)malloc(size);


	// Allocate memory in device and host 	

	cudaCheck(cudaMalloc( (void**)&u.x,size),"malloc_t1");
	cudaCheck(cudaMalloc( (void**)&u.y,size),"malloc_t1");
	cudaCheck(cudaMalloc( (void**)&u.z,size),"malloc_t1");

	//MPI COPY to nodes
	
	
	mpiCheck(read_parallel_float("./data_inicial/Usx.h5",(float*)u_host.x,NX,NY,2*NZ,RANK,SIZE),"read");
	mpiCheck(read_parallel_float("./data_inicial/Usy.h5",(float*)u_host.y,NX,NY,2*NZ,RANK,SIZE),"read");
	mpiCheck(read_parallel_float("./data_inicial/Usz.h5",(float*)u_host.z,NX,NY,2*NZ,RANK,SIZE),"read");
	
	//COPY to GPUs

	cudaCheck(cudaMemcpy(u.x,u_host.x, size, cudaMemcpyHostToDevice),"MemInfo1_A");
	cudaCheck(cudaMemcpy(u.y,u_host.y, size, cudaMemcpyHostToDevice),"MemInfo1_A");
	cudaCheck(cudaMemcpy(u.z,u_host.z, size, cudaMemcpyHostToDevice),"MemInfo1_A");

	//U sep up

	dealias(u);
	projectFourier(u);

	//RK integration
	
	float time=10;
	int counter=0;
	
	counter=RK2step(u,&time);
	
	printf("\ncounter=%d\n",counter);

	//COPY to CPU

	
	cudaCheck(cudaMemcpy(u_host.x,u.x,size,cudaMemcpyDeviceToHost),"MemInfo1_B");
	cudaCheck(cudaMemcpy(u_host.y,u.y,size,cudaMemcpyDeviceToHost),"MemInfo1_B");
	cudaCheck(cudaMemcpy(u_host.z,u.z,size,cudaMemcpyDeviceToHost),"MemInfo1_B");
		

	//MPI COPY to nodes
	
	
	mpiCheck(wrte_parallel_float("./data_inicial/Usx_w.h5",(float*)u_host.x,NX,NY,2*NZ,RANK,SIZE),"write");
	mpiCheck(wrte_parallel_float("./data_inicial/Usy_w.h5",(float*)u_host.y,NX,NY,2*NZ,RANK,SIZE),"write");
	mpiCheck(wrte_parallel_float("./data_inicial/Usz_w.h5",(float*)u_host.z,NX,NY,2*NZ,RANK,SIZE),"write");
	


return;

}


