
#include "turH.h"

static __global__ void calcEnergyShellKernel(float2* ux,float2* uy,float2* uz,float2* t,int ks,int IGLOBAL,int NXSIZE)
{
	

	int i  = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	float k1,k2,k3;
	float kk;

	float N3=(float)N*N*N;

	int k=j%NZ;
	j=(j-k)/NZ;
	
	int h=i*NY*NZ+j*NZ+k;

	if (i<NXSIZE && j<NY && k<NZ)
	{

	// X indices		
	k1=(i+IGLOBAL)<NX/2 ? (float)(i+IGLOBAL) : (float)(i+IGLOBAL)-(float)NX ;

	// Y indice
	k2=j<NY/2 ? (float)j : (float)j-(float)NY ;
	
	// Z indices
	k3=(float)k;

	// Wave numbers

	kk=k1*k1+k2*k2+k3*k3;

	float e1,e2;

	if(kk<ks*ks){		

	// Read {u1,u2,u3}	
	
	float2 u1=ux[h];
	float2 u2=uy[h];
	float2 u3=uz[h];


	u1.x=u1.x/N3;
	u2.x=u2.x/N3;
	u3.x=u3.x/N3;

	u1.y=u1.y/N3;
	u2.y=u2.y/N3;
	u3.y=u3.y/N3;

	float E1=(u1.x*u1.x+u1.y*u1.y);
	float E2=(u2.x*u2.x+u2.y*u2.y);	
	float E3=(u3.x*u3.x+u3.y*u3.y);

	e1=2.0f*(E1+E2+E3);	
	e2=0.0f;

	}else{
	
	e1=0.0f;
	e2=0.0f;
	
	}


	if(k==0){
	e1*=1.0f/2.0f;	
	e2*=1.0f/2.0f;
	}
	
	if(h==0){
	e1=0.0f;
	e2=0.0f;
	}

	t[h].x=e1;
	t[h].y=e2;

	}
}

static dim3 threadsPerBlock;
static dim3 blocksPerGrid;



void calc_energy_shell(vectorField u,float2* t,int ks)
{

	threadsPerBlock.x=THREADSPERBLOCK_IN;
	threadsPerBlock.y=THREADSPERBLOCK_IN;

	blocksPerGrid.x=NXSIZE/threadsPerBlock.x;
	blocksPerGrid.y=NY*NZ/threadsPerBlock.y;

	calcEnergyShellKernel<<<blocksPerGrid,threadsPerBlock>>>(u.x,u.y,u.z,t,ks,IGLOBAL,NXSIZE);
	kernelCheck(RET,"rk_initstep",1);
		
	return;

}

float caclCf(vectorField u,float2* t,int kf, case_config_t *config)
{

	//conserving keta=2

	int kmax=sqrt(2.0f)*N/3.0f;

	float energy;
	
	calc_energy_shell(u,t,kf);	

	energy=sumElements(t);
	
	//if(RANK==0){
	//printf("\nenergy_shell=%f\n",energy/2.0f);
	//};

	float Cf=ENERGY_IN/(energy);
	
	return Cf;

}

