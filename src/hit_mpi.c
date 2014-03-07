
#include "turH.h"

int chxyz2yzx(double *x, double *y, int Nx, int Ny, int Nz,
	      int rank, int size){
  int i,j,k,n;
  int myNx, myNy;
  int MPIErr;
  
  myNx = Nx/size;
  myNy = Ny/size;
  
  
  /* First step. First transpose in memory */
  for (i=0; i<myNx; i++){
    for (j=0; j<Ny; j++){
      for (k=0; k<Nz; k++){
  	y[j*myNx*Nz + i*Nz + k] = x[i*Ny*Nz + j*Nz + k];
      }
    }
  }
  
  /* Second step. Alltoall */
  
  MPIErr = MPI_Alltoall(y,Nz*myNx*myNy,MPI_DOUBLE,
			x,Nz*myNx*myNy,MPI_DOUBLE,
			MPI_COMM_WORLD);

  /* Final step for forward. Transpose in memory */

  for (n=0; n<size; n++){
    for (j=0; j<myNy; j++){
      for (i=0; i<myNx; i++){
	for (k=0; k<Nz; k++){
	  y[j*Nz*size*myNx +
	    k*size*myNx +
	    n*myNx +
	    i] = x[n*myNy*myNx*Nz +
	  	   j*myNx*Nz +
	  	   i*Nz +
	  	   k];
	}
      }
    }
  }
  
  return MPIErr;
}

int chyzx2xyz(double *y, double *x, int Nx, int Ny, int Nz,
	      int rank, int size){
  int i,j,k,n;
  int myNx, myNy;
  int MPIErr;
  
  myNx = Nx/size;
  myNy = Ny/size;

  for (n=0; n<size; n++){
    for (j=0; j<myNy; j++){
      for (i=0; i<myNx; i++){
	for (k=0; k<Nz; k++){
	  x[n*myNy*myNx*Nz +
	    j*myNx*Nz +
	    i*Nz +
	    k] = y[j*Nz*size*myNx +
		   k*size*myNx +
		   n*myNx +
		   i];
	}
      }
    }
  }

  /* Communications */
  MPIErr = MPI_Alltoall(x,Nz*myNx*myNy,MPI_DOUBLE,
			y,Nz*myNx*myNy,MPI_DOUBLE,
			MPI_COMM_WORLD);


  /* Fourth and final transpose */
  for (i=0; i<myNx; i++){
    for (j=0; j<Ny; j++){
      for (k=0; k<Nz; k++){
	x[i*Ny*Nz + j*Nz + k] = y[j*myNx*Nz + i*Nz + k];
      }
    }
  }
  return MPIErr;
}

int wrte_parallel_double(char *filename, double *x, int Nx, int Ny, int Nz,
			 int rank, int size){
  int i,j,k;
  int MPIErr;
  herr_t H5Err;
  hid_t fileid, dsetid, dspaceid, mspaceid;
  hsize_t sizem[3], sized[3], start[3], stride[3], count[3];
  double *aux;
  
  int myNx = Nx/size;

  aux = (double *) malloc(Ny*Nz*sizeof(double));

  if (rank == 0){
    fileid = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    start[0] = 0; start[1] = 0; start[2] = 0;
    sized[0] = Nx; sized[1] = Ny; sized[2] = Nz;
    sizem[0] = 1; sizem[1] = Ny; sizem[2] = Nz;
    stride[0] = 1; stride[1] = Ny; stride[2] = Nz;
    count[0] = 1; count[1] = 1; count[2] = 1;
    dspaceid = H5Screate_simple(3,sized,sized);
    dsetid = H5Dcreate(fileid, "u", H5T_NATIVE_DOUBLE, dspaceid,
  		       H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  }
  
  for (i=0; i<Nx; i++){
    if (i/myNx == 0){
      if (rank == 0){
	for (j=0; j<Ny; j++){
	  for (k=0; k<Nz; k++){
	    aux[j*Nz + k] = x[(i%myNx)*Ny*Nz + j*Nz + k];
	  }
	}
	mspaceid = H5Screate_simple(3,sizem,sizem);
	start[0] = 0;
	H5Err = H5Sselect_hyperslab(mspaceid,H5S_SELECT_SET,
				    start,stride,count,sizem);
	start[0] = i;
	H5Err = H5Sselect_hyperslab(dspaceid,H5S_SELECT_SET,
				    start,stride,count,sizem);
	H5Err = H5Dwrite(dsetid,H5T_NATIVE_DOUBLE,mspaceid,dspaceid,
			 H5P_DEFAULT,aux);
	H5Err = H5Sclose(mspaceid);
      }
    }
    else{
      if (i/myNx == rank){
	for (j=0; j<Ny; j++){
	  for (k=0; k<Nz; k++){
	    aux[j*Nz + k] = x[(i%myNx)*Ny*Nz + j*Nz + k];
	  }
	}
	MPIErr = MPI_Send(aux,Ny*Nz,MPI_DOUBLE,0,11,MPI_COMM_WORLD); 
      }
      
      else if (rank == 0){
	MPIErr = MPI_Recv(aux,Ny*Nz,MPI_DOUBLE,i/myNx,11,MPI_COMM_WORLD,
			  MPI_STATUS_IGNORE);
	mspaceid = H5Screate_simple(3,sizem,sizem);
	start[0] = 0;
	H5Err = H5Sselect_hyperslab(mspaceid,H5S_SELECT_SET,
				    start,stride,count,sizem);
	start[0] = i;
	H5Err = H5Sselect_hyperslab(dspaceid,H5S_SELECT_SET,
				    start,stride,count,sizem);
	H5Err = H5Dwrite(dsetid,H5T_NATIVE_DOUBLE,mspaceid,dspaceid,
			 H5P_DEFAULT,aux);
	H5Err = H5Sclose(mspaceid);
      }
    }
    MPIErr = MPI_Barrier(MPI_COMM_WORLD);
  }
  
  
  if (rank == 0){
    H5Err = H5Dclose(dsetid);
    H5Err = H5Sclose(dspaceid);
    H5Err = H5Fclose(fileid);
  }
  
  return MPIErr;
}

int read_parallel_double(char *filename, double *x, int Nx, int Ny, int Nz,
			 int rank, int size){
  int i,j,k;
  int MPIErr;
  herr_t H5Err;
  hid_t fileid, dsetid, dspaceid, mspaceid;
  hsize_t sizem[3], sized[3], start[3], stride[3], count[3];
  double *aux;
  
  int myNx = Nx/size;

  aux = (double *) malloc(Ny*Nz*sizeof(double));

  if (rank == 0){
    fileid = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    start[0] = 0; start[1] = 0; start[2] = 0;
    sized[0] = Nx; sized[1] = Ny; sized[2] = Nz;
    sizem[0] = 1; sizem[1] = Ny; sizem[2] = Nz;
    stride[0] = 1; stride[1] = Ny; stride[2] = Nz;
    count[0] = 1; count[1] = 1; count[2] = 1;
    dsetid = H5Dopen(fileid, "u", H5P_DEFAULT);
    dspaceid = H5Dget_space(dsetid);
  }
  
  for (i=0; i<Nx; i++){
    if (i/myNx == 0){
      if (rank == 0){
	mspaceid = H5Screate_simple(3,sizem,sizem);
	start[0] = 0;
	H5Err = H5Sselect_hyperslab(mspaceid,H5S_SELECT_SET,
				    start,stride,count,sizem);
	start[0] = i;
	H5Err = H5Sselect_hyperslab(dspaceid,H5S_SELECT_SET,
				    start,stride,count,sizem);
	H5Err = H5Dread(dsetid,H5T_NATIVE_DOUBLE,mspaceid,dspaceid,
			H5P_DEFAULT,aux);
	H5Err = H5Sclose(mspaceid);
	for (j=0; j<Ny; j++){
	  for (k=0; k<Nz; k++){
	    x[(i%myNx)*Ny*Nz + j*Nz + k] = aux[j*Nz + k];
	  }
	}
      }
    }
    else{
      if (i/myNx == rank){
	MPIErr = MPI_Recv(aux,Ny*Nz,MPI_DOUBLE,0,11,MPI_COMM_WORLD,
			  MPI_STATUS_IGNORE); 
	for (j=0; j<Ny; j++){
	  for (k=0; k<Nz; k++){
	    x[(i%myNx)*Ny*Nz + j*Nz + k] = aux[j*Nz + k];
	  }
	}
      }
      
      else if (rank == 0){
	mspaceid = H5Screate_simple(3,sizem,sizem);
	start[0] = 0;
	H5Err = H5Sselect_hyperslab(mspaceid,H5S_SELECT_SET,
				    start,stride,count,sizem);
	start[0] = i;
	H5Err = H5Sselect_hyperslab(dspaceid,H5S_SELECT_SET,
				    start,stride,count,sizem);
	H5Err = H5Dread(dsetid,H5T_NATIVE_DOUBLE,mspaceid,dspaceid,
			H5P_DEFAULT,aux);
	H5Err = H5Sclose(mspaceid);
	MPIErr = MPI_Send(aux,Ny*Nz,MPI_DOUBLE,i/myNx,11,MPI_COMM_WORLD);
      }
    }
    MPIErr = MPI_Barrier(MPI_COMM_WORLD);
  }
  
  
  if (rank == 0){
    H5Err = H5Dclose(dsetid);
    H5Err = H5Sclose(dspaceid);
    H5Err = H5Fclose(fileid);
  }
  
  return MPIErr;
}

int wrte_parallel_float(char *filename, float *x, int NX, int NY, int NZ,
			 int rank, int size){
  int i,j,k;
  int MPIErr;
  herr_t H5Err;
  hid_t fileid, dsetid, dspaceid, mspaceid;
  hsize_t sizem[3], sized[3], start[3], stride[3], count[3];
  float *aux;
  
  int myNX = NX/size;

  aux = (float *) malloc(NY*NZ*sizeof(float));

  if (rank == 0){
    fileid = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    start[0] = 0; start[1] = 0; start[2] = 0;
    sized[0] = NX; sized[1] = NY; sized[2] = NZ;
    sizem[0] = 1; sizem[1] = NY; sizem[2] = NZ;
    stride[0] = 1; stride[1] = NY; stride[2] = NZ;
    count[0] = 1; count[1] = 1; count[2] = 1;
    dspaceid = H5Screate_simple(3,sized,sized);
    dsetid = H5Dcreate(fileid, "u", H5T_NATIVE_FLOAT, dspaceid,
  		       H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  }
  
  for (i=0; i<NX; i++){
    if (i/myNX == 0){
      if (rank == 0){
	for (j=0; j<NY; j++){
	  for (k=0; k<NZ; k++){
	    aux[j*NZ + k] = x[(i%myNX)*NY*NZ + j*NZ + k];
	  }
	}
	mspaceid = H5Screate_simple(3,sizem,sizem);
	start[0] = 0;
	H5Err = H5Sselect_hyperslab(mspaceid,H5S_SELECT_SET,
				    start,stride,count,sizem);
	start[0] = i;
	H5Err = H5Sselect_hyperslab(dspaceid,H5S_SELECT_SET,
				    start,stride,count,sizem);
	H5Err = H5Dwrite(dsetid,H5T_NATIVE_FLOAT,mspaceid,dspaceid,
			 H5P_DEFAULT,aux);
	H5Err = H5Sclose(mspaceid);
      }
    }
    else{
      if (i/myNX == rank){
	for (j=0; j<NY; j++){
	  for (k=0; k<NZ; k++){
	    aux[j*NZ + k] = x[(i%myNX)*NY*NZ + j*NZ + k];
	  }
	}
	MPIErr = MPI_Send(aux,NY*NZ,MPI_FLOAT,0,11,MPI_COMM_WORLD); 
      }
      
      else if (rank == 0){
	MPIErr = MPI_Recv(aux,NY*NZ,MPI_FLOAT,i/myNX,11,MPI_COMM_WORLD,
			  MPI_STATUS_IGNORE);
	mspaceid = H5Screate_simple(3,sizem,sizem);
	start[0] = 0;
	H5Err = H5Sselect_hyperslab(mspaceid,H5S_SELECT_SET,
				    start,stride,count,sizem);
	start[0] = i;
	H5Err = H5Sselect_hyperslab(dspaceid,H5S_SELECT_SET,
				    start,stride,count,sizem);
	H5Err = H5Dwrite(dsetid,H5T_NATIVE_FLOAT,mspaceid,dspaceid,
			 H5P_DEFAULT,aux);
	H5Err = H5Sclose(mspaceid);
      }
    }
    MPIErr = MPI_Barrier(MPI_COMM_WORLD);
  }
  
  
  if (rank == 0){
    H5Err = H5Dclose(dsetid);
    H5Err = H5Sclose(dspaceid);
    H5Err = H5Fclose(fileid);
  }

  free(aux);
  return MPIErr;
}

int read_parallel_float(char *filename, float *x, int NX, int NY, int NZ,
			 int rank, int size){
  int i,j,k;
  int MPIErr;
  herr_t H5Err;
  hid_t fileid, dsetid, dspaceid, mspaceid;
  hsize_t sizem[3], sized[3], start[3], stride[3], count[3];
  float *aux;
  
  int myNX = NX/size;

  aux = (float *) malloc(NY*NZ*sizeof(float));

  if (rank == 0){
    fileid = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    start[0] = 0; start[1] = 0; start[2] = 0;
    sized[0] = NX; sized[1] = NY; sized[2] = NZ;
    sizem[0] = 1; sizem[1] = NY; sizem[2] = NZ;
    stride[0] = 1; stride[1] = NY; stride[2] = NZ;
    count[0] = 1; count[1] = 1; count[2] = 1;
    dsetid = H5Dopen(fileid, "u", H5P_DEFAULT);
    dspaceid = H5Dget_space(dsetid);
  }
  
  for (i=0; i<NX; i++){
    if (i/myNX == 0){
      if (rank == 0){
	mspaceid = H5Screate_simple(3,sizem,sizem);
	start[0] = 0;
	H5Err = H5Sselect_hyperslab(mspaceid,H5S_SELECT_SET,
				    start,stride,count,sizem);
	start[0] = i;
	H5Err = H5Sselect_hyperslab(dspaceid,H5S_SELECT_SET,
				    start,stride,count,sizem);
	H5Err = H5Dread(dsetid,H5T_NATIVE_FLOAT,mspaceid,dspaceid,
			H5P_DEFAULT,aux);
	H5Err = H5Sclose(mspaceid);
	for (j=0; j<NY; j++){
	  for (k=0; k<NZ; k++){
	    x[(i%myNX)*NY*NZ + j*NZ + k] = aux[j*NZ + k];
	  }
	}
      }
    }
    else{
      if (i/myNX == rank){
	MPIErr = MPI_Recv(aux,NY*NZ,MPI_FLOAT,0,11,MPI_COMM_WORLD,
			  MPI_STATUS_IGNORE); 
	for (j=0; j<NY; j++){
	  for (k=0; k<NZ; k++){
	    x[(i%myNX)*NY*NZ + j*NZ + k] = aux[j*NZ + k];
	  }
	}
      }
      
      else if (rank == 0){
	mspaceid = H5Screate_simple(3,sizem,sizem);
	start[0] = 0;
	H5Err = H5Sselect_hyperslab(mspaceid,H5S_SELECT_SET,
				    start,stride,count,sizem);
	start[0] = i;
	H5Err = H5Sselect_hyperslab(dspaceid,H5S_SELECT_SET,
				    start,stride,count,sizem);
	H5Err = H5Dread(dsetid,H5T_NATIVE_FLOAT,mspaceid,dspaceid,
			H5P_DEFAULT,aux);
	H5Err = H5Sclose(mspaceid);
	MPIErr = MPI_Send(aux,NY*NZ,MPI_FLOAT,i/myNX,11,MPI_COMM_WORLD);
      }
    }
    MPIErr = MPI_Barrier(MPI_COMM_WORLD);
  }
  
  
  if (rank == 0){
    H5Err = H5Dclose(dsetid);
    H5Err = H5Sclose(dspaceid);
    H5Err = H5Fclose(fileid);
  }

  free(aux);
  return MPIErr;
}


//ALL to ALL reduce

void reduceMAX(float* u1,float* u2,float* u3){

	float* ux=(float*)malloc(sizeof(float));
	float* uy=(float*)malloc(sizeof(float));
	float* uz=(float*)malloc(sizeof(float));

	mpiCheck(MPI_Allreduce(u1,ux,1,MPI_FLOAT,MPI_MAX,MPI_COMM_WORLD),"caca");
	mpiCheck(MPI_Allreduce(u2,uy,1,MPI_FLOAT,MPI_MAX,MPI_COMM_WORLD),"caca");
	mpiCheck(MPI_Allreduce(u3,uz,1,MPI_FLOAT,MPI_MAX,MPI_COMM_WORLD),"caca");

	*u1=*ux;
	*u2=*uy;
	*u3=*uz;

	free(ux);
	free(uy);
	free(uz);

	return;

}


void reduceSUM(float* sum,float* sum_all){

	mpiCheck(MPI_Allreduce(sum,sum_all,1,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD),"caca");
	return;

}




