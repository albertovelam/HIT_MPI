
#include "turH.h"


//RK2 CODE

static vectorField uw;
static vectorField r;
static float2* t1;

FILE *fp_rk2_1;
FILE *fp_rk2_2;

void RK2setup(void)
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

	//SET BLOCK DIMENSIONS

	// Set the file to write in
		
	fp_rk2_1=fopen("./data/data_1.dat","w");
	fp_rk2_2=fopen("./data/data_2.dat","w");


	return;

}



static float calcDt(vectorField uw,vectorField u){

	
	
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
	
	
	dt=fmin(dtc,dtv);
	//dt=fmin(dt,dtf);

	//Calculate data

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

	
	// Print data to screen	

	if(RANK==0){
	printf("\n(dtc,dts,dtf)=(%f,%f)",dtc,dtv);
	printf("\nvmax=(%f,%f,%f)",umax[0]/N3,umax[1]/N3,umax[2]/N3);	
	printf("\n(E,D)=(%e,%e)",E[0],D[0]);
	printf("\nu_p=%f",u_p);
	printf("\nomega_p=%f",omega_p);	
	printf("\n(eta,lamb)=(%f,%f)",eta,lambda);
	printf("\nRl=%f",Rl);
	printf("\netak=%f",eta*kmax);	
	printf("\n************************\n");
	}

		
	free(D);
	free(E);
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

int RK2step(vectorField u,float* time, config_t *config)
{
	
	static float time_elapsed=0.0f;
	static int counter=0;	

	float pi=acos(-1.0f);
	float om=2.0f*pi/N;	
	
	float* Delta=(float*)malloc(3*sizeof(float));
	float* Delta_1=(float*)malloc(3*sizeof(float));
	float* Delta_2=(float*)malloc(3*sizeof(float));	

	int frec=2000;

	int kf=2;
		
	float dt=0.0f;
	float Cf;	

	//RK2 time steps	

	while(time_elapsed < *time){

	//Calc forcing	
	  if (config_setting_get_bool(config_lookup(config,"application.forcing"))){
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
	Delta_2[i]=om*(Delta[i]+0.5f);}
	
	//First substep

	copyVectorField(uw,u);	

	F(uw,r,Delta_1); 

	dt=calcDt(uw,u);	
	printf("dt = %f\n",dt);

	RK_step_1(uw,u,r,REYNOLDS,dt,Cf,kf);

	//Second substep
	
	RK_step_05(u,uw,REYNOLDS,dt,Cf,kf);	
	
	F(uw,r,Delta_2); 

	RK_step_2(u,r,REYNOLDS,dt,Cf,kf);	 
	
	counter++;
	time_elapsed+=dt;

	if(RANK==0){
	printf("\nsimtime=%f",time_elapsed);
	printf("\ncounter=%d",counter);
	printf("\nCf=%f\n",Cf);
	}

	//Write to disc	

	if(counter%frec==0){
	//velocityH5(u,t1,counter/frec,N);
	//dissipationH5(modS,Cs,counter/frec,N);
	//enstrophyH5(u,t3,t2.x,counter/frec,N);
	//writeVectorField(u,N);
	}

	//End of step
	}
	
	*time=time_elapsed;

	free(Delta);
	free(Delta_1);
	free(Delta_2);

	if (RANK == 0){ printf("RK iterations finished.");}

	return counter;	
	
}

