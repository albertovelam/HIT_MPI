
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cublas.h>
#include <cufft.h>

extern int pipe_xfer;
extern int min_kb_xfer;
extern char mybus[16];
extern int together;


#include <hdf5.h>
#include <hdf5_hl.h>

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

#include <libconfig.h>

#ifndef NSS
#define NSS 256
#endif

#ifndef RES
#define RES 2.0f
#endif

typedef struct { float2* x;float2* y;float2* z;} vectorField;

/* 
   Structure that contains the configuration of the run. NSS is
   missing, because it is used at compile time
 */

typedef struct case_config_t {
  float CFL;
  float time;
  float resolution;
  int forcing;
  int stats_every;
  char *readU;
  char *readV;
  char *readW;
  char *statfile;
  char *writeU;
  char *writeV;
  char *writeW;
} case_config_t;


//ONLY FOR NY=NX

static const int N =NSS;
static const int NX=NSS;
static const int NY=NSS;
static const int NZ=NSS/2+1;

static const int THREADSPERBLOCK_IN=16;

//static const float REYNOLDS=NSS;
//static const float ENERGY_IN=powf(sqrt(3.0f)/2.0f*NSS/RES,4.0f)*powf(1.0f/REYNOLDS,3.0f);

static const float ENERGY_IN=1.0f;
static const float REYNOLDS=powf(sqrt(2.0f)/3.0f*NSS/RES,4.0f/3.0f)*powf(ENERGY_IN,-1.0f/3.0f);


//Global variables

extern cudaError_t RET;

extern int RANK;
extern int SIZE;

extern int NXSIZE;
extern int NYSIZE;
extern int IGLOBAL;

//AUX BUFFER

extern float2* AUX;

/////////////////////////// C FUNCTIONS /////////////////////////////////////////

//Set up

void setUp(void);
void starSimulation(void);

//RK2

void RK2setup(void);
int RK2step(vectorField u,float* time, case_config_t *config);

void RK3setup(void);
int  RK3step(vectorField u,float* time, case_config_t *config);

//Random

int randomNumberGen(int T);
void seedGen(void);
void genDelta(float* Delta);


//Fft

void fftSetup(void);
void fftDestroy(void);

void fftForward(float2* buffer_1);
void fftBackward(float2* buffer_1);

void calcUmax(vectorField t,float* ux,float* uy,float* uz);
float sumElements(float2* buffer);

//F

void copyVectorField(vectorField u1,vectorField u2);
void F( vectorField u, vectorField r,float* Delta);

void mpiCheck(int error, const char* function );

//Check

void cudaCheck( cudaError_t error, const char* function);
void mpiCheck( int error, const char* function);

//hit_mpi

void reduceMAX(float* u1,float* u2,float* u3);
void reduceSUM(float* sum,float* sum_all);

int chyzx2xyz(double *y, double *x, int Nx, int Ny, int Nz,
	      int rank, int size);
int chxyz2yzx(double *x, double *y, int Nx, int Ny, int Nz,
	      int rank, int size);
int read_parallel_float(char *filename, float *x, int NX, int NY, int NZ,
			 int rank, int size);
int wrte_parallel_float(char *filename, float *x, int NX, int NY, int NZ,
			 int rank, int size);

//hit_cuda MPI
void setTransposeCudaMpi(void);
void transposeXYZ2YZX(float2* u1,int Nx,int Ny,int Nz,int rank,int size);
void transposeYZX2XYZ(float2* u1,int Nx,int Ny,int Nz,int rank,int size);

//Statistics
void calc_E( vectorField u, float2* t,float* E);
void calc_D( vectorField u, float2* t,float* D);

//Fft overlap

void setFftAsync(void);

void transpose_A(float2* u_2,float2* u_1);
void transpose_B(float2* u_2,float2* u_1);

void fftBack1T(float2* u1);
void fftForw1T(float2* u1);

void fftBack1T_A(float2* u1,int id);
void fftBack1T_B(float2* u1,int id);

void fftForw1T_A(float2* u1,int id);
void fftForw1T_B(float2* u1,int id);

void fftBackMultiple(float2* u1,float2* u2,float2* u3,float2* u4,float2* u5,float2* u6);
void fftForwMultiple(float2* u1,float2* u2,float2* u3);
void calcUmaxV2(vectorField t,float* ux,float* uy,float* uz);
float sumElementsV2(float2* buffer_1);

//Routine check
void fftCheck(void);

///////////CUDA FUNCTIONS////////////////////////////////////////////

extern cudaStream_t compute_stream;
extern cudaStream_t d2h_stream;
extern cudaStream_t h2d_stream;
extern cudaEvent_t events[1000];

//transpose
extern void trans_zyx_to_yzx(float2* input, float2* output,cudaStream_t stream);
extern void trans_yzx_to_zyx(float2* input, float2* output,cudaStream_t stream);
extern void trans_yzx_to_zyx_yblock(float2* input, float2* output,cudaStream_t stream);
extern void trans_zxy_to_yzx(float2* input, float2* output,cudaStream_t stream);
extern void trans_zxy_to_zyx(float2* input, float2* output,cudaStream_t stream);
extern void trans_zyx_to_zxy(float2* input, float2* output,cudaStream_t stream);
extern void trans_zyx_yblock_to_yzx(float2* input, float2* output,cudaStream_t stream);


//RK2_kernels
extern void RK2_step_1(vectorField uw,vectorField u,vectorField r,float Re,float dt,float Cf,int kf);
extern void RK2_step_05(vectorField u,vectorField uw,float Re,float dt,float Cf,int kf);
extern void RK2_step_2(vectorField uw,vectorField r,float Re,float dt,float Cf,int kf);

//RK3_kernels

extern void RK3_step_1(vectorField uw,vectorField u,vectorField r,float Re,float dt,float Cf,int kf,int nc);
extern void RK3_step_2(vectorField uw,vectorField u,vectorField r,float Re,float dt,float Cf,int kf,int nc);

//Dealias

extern void dealias(vectorField t);
extern void projectFourier(vectorField u);
extern void set2zero(float2* u);

//check
void kernelCheck(  cudaError_t error,const char* function , int a);

//Memory
extern void memoryInfo(void);

//Rotor convolution

extern void calc_conv_rotor(vectorField r, vectorField s);
extern void calc_conv_rotor_3(vectorField r, vectorField s);
extern void calc_conv_rotor_12(vectorField r, vectorField s, float2* temp);


//Shift

extern void shift(vectorField t,float* Delta);

//UW

extern void calc_U_W( vectorField U,vectorField W);

//imposeSymetries
extern void imposeSymetry(vectorField t);

//Routine check
extern void routineCheck(vectorField t);

//Statistics Kernels
extern void calc_E_kernel( vectorField u, float2* t);
extern void calc_D_kernel( vectorField u, float2* t);

//Forcing
void calc_energy_shell(vectorField u,float2* t,int ks);
float caclCf(vectorField u,float2* t,int kf, case_config_t *config);

