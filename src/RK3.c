
#include "turH.h"


//RK2 CODE

static vectorField uw;
static vectorField r;
static float2* t1;


void RK3setup(void)
{
	

	size_t size=NXSIZE*NY*NZ*sizeof(float2);

	cudaCheck(cudaMalloc( (void**)&uw.x,size),"malloc_t1");
	cudaCheck(cudaMalloc( (void**)&uw.y,size),"malloc_t1");
	cudaCheck(cudaMalloc( (void**)&uw.z,size),"malloc_t1");
	
	set2zero(uw.x);
	set2zero(uw.y);
	set2zero(uw.z);

	cudaCheck(cudaMalloc( (void**)&t1,size),"malloc_t1");	

	set2zero(t1);
	
	cudaCheck(cudaMalloc( (void**)&r.x,size),"malloc_t1");
	cudaCheck(cudaMalloc( (void**)&r.y,size),"malloc_t1");
	cudaCheck(cudaMalloc( (void**)&r.z,size),"malloc_t1");

	set2zero(r.x);
	set2zero(r.y);
	set2zero(r.z);

	return;

}


static void collect_statistics(int step, float dt, vectorField u, case_config_t *config){

  float* E=(float*)malloc(sizeof(float));
  float* D=(float*)malloc(sizeof(float));
  
  calc_E(u,t1,E);
  calc_D(u,t1,D);
  
  float u_p=sqrt((2.0f/3.0f)*E[0]);	
  float omega_p=sqrt(REYNOLDS*D[0]);
  
  float lambda=sqrt(15.0f*u_p*u_p/(omega_p*omega_p));
  float eta=pow(REYNOLDS,-3.0f/4.0f)*pow(D[0],-1.0f/4.0f);
  
  float Rl=u_p*lambda*REYNOLDS;
  int kmax=sqrt(2.0f)/3.0f*N;	

	
  if(RANK == 0){
    FILE *statfilep = fopen(config->statfile,"a");
    printf("Appending file %s\n",config->statfile);
    fprintf(statfilep,"%e,%e,%e,%e,%e,%e,%e,%e,%e\n",
    	    dt,E[0],D[0],u_p,omega_p,eta,lambda,Rl,eta*kmax);
    fclose(statfilep);
  }

  free(D);
  free(E);
}

static float calcDt(vectorField uw){	
	
	const float cfl=0.5;
	float dt=0.0f;

	float dtc=0.0f;	
	float dtf=0.0f;
	float dtv=0.0f;	
	
	float N3=N*N*N;
	
	float* umax=(float*)malloc(3*sizeof(float));
	
	calcUmax(uw,umax,umax+1,umax+2);

	float c=(abs(umax[0]/N3)+abs(umax[1]/N3)+abs(umax[2]/N3));
	
	dtc=cfl/((N/3)*c);	
	dtv=cfl*REYNOLDS/((N/3)*(N/3));
	/*
	if(RANK == 0){
	printf("\nVmax=(%f,%f,%f)\n",umax[0]/N3,umax[1]/N3,umax[2]/N3);
	}
	*/
	dt=fmin(dtc,dtv);
	//dt=fmin(dt,dtf);

	free(umax);

	return dt;

}

static float caclCf(vectorField u,float2* t,int kf)
{

	//conserving keta=2

	int kmax=sqrt(2.0f)*N/3.0f;

	float energy;
	
	calc_energy_shell(u,t,kf);	

	energy=sumElements(t);
	
	//if(RANK==0){
	//printf("\nenergy_shell=%f\n",energy/2.0f);
	//};

	float Cf=pow(REYNOLDS,-3.0f)*pow(kmax/2.0f,4.0f)/(energy);
	
	return Cf;

}

int RK3step(vectorField u,float* time, case_config_t *config)
{
	
	static float time_elapsed=0.0f;
	static int counter=0;	

	float pi=acos(-1.0f);
	float om=2.0f*pi/N;	
	
	float Delta[3];
	float Delta_1[3];
	float Delta_2[3];
	float Delta_3[3];  

	int frec=2000;

	int kf=2;
		
	float dt=0.0f;
	float Cf;	

	//RK2 time steps	
	printf("\n time=%f",*time);
	while(time_elapsed < *time){

	//Calc forcing	
	  if(config->forcing){
	    Cf=caclCf(u,t1,kf);
	  }
	  else{
	    Cf = 0.0;
	  }   

	//Initial dealiasing

	dealias(u);	
	
	//Generate delta for dealiasing	
	
	genDelta(Delta);
	

	//printf("\n%f,%f,%f\n",Delta[0],Delta[1],Delta[2]);	
	
	for(int i=0;i<3;i++){
	Delta_1[i]=om*Delta[i];
	Delta_2[i]=om*(Delta[i]+1.0f/3.0f);
	Delta_3[i]=om*(Delta[i]+2.0f/3.0f);}

	//First substep	

	copyVectorField(uw,u);	
	F(uw,r,Delta_1); 

	dt=calcDt(uw);	

	if( counter%config->stats_every == 0 ){
	  if (RANK == 0){ printf("Computing statistics.\n");}
	  collect_statistics(counter,dt,u,config);
	}

	RK3_step_1(u,uw,r,REYNOLDS,dt,Cf,kf,0);
	
	RK3_step_2(u,uw,r,REYNOLDS,dt,Cf,kf,0);


	//Second substep	

	RK3_step_1(u,uw,r,REYNOLDS,dt,Cf,kf,1); 
	
	F(u,r,Delta_2); 

	RK3_step_2(u,uw,r,REYNOLDS,dt,Cf,kf,1); 
	
	//Third substep

	RK3_step_1(u,uw,r,REYNOLDS,dt,Cf,kf,2); 
	
	F(u,r,Delta_3); 

	RK3_step_2(u,uw,r,REYNOLDS,dt,Cf,kf,2); 
	


	counter++;
	time_elapsed+=dt;

	if(RANK==0){
	  printf("Timestep: %d, ",counter);
	  printf("Simulation time: %f, ",time_elapsed);
	  printf("Forcing coefficient: %f\n",Cf);
	}

	//End of step
	}
	
	*time=time_elapsed;


	if (RANK == 0){ printf("RK iterations finished.\n");}

	return counter;	
	
}
