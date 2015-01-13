#include "turH.h"
#include <string.h>


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

	cudaCheck(cudaSetDevice(RANK/2/*RANK%Ndevices*/),"Set");

	printf("\n%d using DEVICE=%d\n",RANK,RANK/2);//RANK%Ndevices);	

	//Setups
	fftSetup();
	setFftAsync();
	
	//RK2setup();
	RK3setup();
	
	size_t size=NXSIZE*NY*NZ*sizeof(float2);
	cudaCheck(cudaMalloc( (void**)&AUX,size),"malloc_t1");		

	setTransposeCudaMpi();	

	return;

}

void starSimulation(void){

        config_t config;
	config_setting_t *read;
	config_setting_t *write;
	const char *str;
	
	// Read configuration file
	config_init(&config);
  
	if (! config_read_file(&config, "run.conf")){
	  fprintf(stderr, "%s:%d - %s\n", config_error_file(&config),
		  config_error_line(&config), config_error_text(&config));
	  config_destroy(&config);
	  return;
	}
	case_config_t case_config = {
	  (float) config_setting_get_float(config_lookup(&config,"application.CFL")),
	  (float) config_setting_get_float(config_lookup(&config,"application.time")),
	  (float) config_setting_get_float(config_lookup(&config,"application.RES")),
	  (int) config_setting_get_bool(config_lookup(&config,"application.forcing")),
	  (int) config_setting_get_int(config_lookup(&config,"application.stats_every")),
	  (char *) config_setting_get_string(config_lookup(&config,"application.read.U")),
	  (char *) config_setting_get_string(config_lookup(&config,"application.read.V")),
	  (char *) config_setting_get_string(config_lookup(&config,"application.read.W")),
	  (char *) config_setting_get_string(config_lookup(&config,"application.statfile")),
	  (char *) config_setting_get_string(config_lookup(&config,"application.write.U")),
	  (char *) config_setting_get_string(config_lookup(&config,"application.write.V")),
	  (char *) config_setting_get_string(config_lookup(&config,"application.write.W")),
	};

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
	if (strcmp(case_config.readU, "-") == 0){
	    // If the file name is -, then a dummy field is created
	    if (RANK == 0){ printf("Creating dummy file.\n");}
	    mpiCheck(create_parallel_float((float*)u_host.x,NX,NY,2*NZ,RANK,SIZE),"read");
	    mpiCheck(create_parallel_float((float*)u_host.x,NX,NY,2*NZ,RANK,SIZE),"read");
	    mpiCheck(create_parallel_float((float*)u_host.x,NX,NY,2*NZ,RANK,SIZE),"read");
	  }
	else{
	  if (RANK == 0){
	    printf("Something really wrong happened \n");
	    printf("%s \n", case_config.readU);
	  }
	  exit(1);
	  /* mpiCheck(read_parallel_float(case_config.readU,(float*)u_host.x,NX,NY,2*NZ,RANK,SIZE),"read"); */
	  /* mpiCheck(read_parallel_float(case_config.readV,(float*)u_host.y,NX,NY,2*NZ,RANK,SIZE),"read"); */
	  /* mpiCheck(read_parallel_float(case_config.readW,(float*)u_host.z,NX,NY,2*NZ,RANK,SIZE),"read"); */
	}
/*
        for(int i=0; i<NXSIZE*NY*NZ; i++){
          u_host.x[i] = { 0.5, 0.5 };
          u_host.y[i] = { 0.0, 0.0 };
          u_host.z[i] = { 0.0, 0.0 };
        }	
*/

	//COPY to GPUs

	cudaCheck(cudaMemcpy(u.x,u_host.x, size, cudaMemcpyHostToDevice),"MemInfo1_A");
	cudaCheck(cudaMemcpy(u.y,u_host.y, size, cudaMemcpyHostToDevice),"MemInfo1_A");
	cudaCheck(cudaMemcpy(u.z,u_host.z, size, cudaMemcpyHostToDevice),"MemInfo1_A");

	//U set up

	dealias(u);
	projectFourier(u);

	//RK integration
	

	float time = (float) config_setting_get_float(config_lookup(&config,"application.time"));

	int counter=0;
	
	counter=RK3step(u,&time,&case_config);

	int mpierr = MPI_Barrier(MPI_COMM_WORLD);

	if (RANK == 0){ printf("RK iterations finished.\n");}

	//COPY to CPU

	cudaCheck(cudaMemcpy(u_host.x,u.x,size,cudaMemcpyDeviceToHost),"MemInfo1_B");
	cudaCheck(cudaMemcpy(u_host.y,u.y,size,cudaMemcpyDeviceToHost),"MemInfo1_B");
	cudaCheck(cudaMemcpy(u_host.z,u.z,size,cudaMemcpyDeviceToHost),"MemInfo1_B");
		

	//MPI COPY to nodes
	

	if (RANK == 0){ printf("Writing output.\n");}
	
	mpiCheck(wrte_parallel_float(case_config.writeU,(float*)u_host.x,NX,NY,2*NZ,RANK,SIZE),"write");
	mpiCheck(wrte_parallel_float(case_config.writeV,(float*)u_host.y,NX,NY,2*NZ,RANK,SIZE),"write");
	mpiCheck(wrte_parallel_float(case_config.writeW,(float*)u_host.z,NX,NY,2*NZ,RANK,SIZE),"write");
	
	if (RANK == 0){ printf("Nothing important left to do.\n");}

	config_destroy(&config);

return;

}


