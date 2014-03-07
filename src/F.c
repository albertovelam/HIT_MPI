#include "turH.h"



void copyVectorField(vectorField u1,vectorField u2){

	// Copy to u
	
	size_t size=NXSIZE*NY*NZ*sizeof(float2);	
	
	cudaCheck(cudaMemcpy(u1.x,u2.x, size, cudaMemcpyDeviceToDevice),"MemInfo1");
	cudaCheck(cudaMemcpy(u1.y,u2.y, size, cudaMemcpyDeviceToDevice),"MemInfo1");
	cudaCheck(cudaMemcpy(u1.z,u2.z, size, cudaMemcpyDeviceToDevice),"MemInfo1");
	
	return;

}

void F( vectorField u, vectorField r,float* Delta)
{
	
	// Calculate vorticity w and move u to u_w workarray	
	// s -> u
	// r -> w

	calc_U_W(u,r);

	// Phase shift U and W

	shift(r,Delta);
	shift(u,Delta);		

	
	// Fill with zeros	

	dealias(u);
	
	// Transform u to real space

	
	fftBackward(u.x);
	fftBackward(u.y);
	fftBackward(u.z);
		

	// Fill with zeros
	
	dealias(r);	
	
	// Transform w to real space

	
	fftBackward(r.x);
	fftBackward(r.y);	
	fftBackward(r.z);	
	

	// Calculate the convolution rotor of u and w


	calc_conv_rotor(r,u);

	// Transform rotor to fourier space

	
	fftForward(r.x);
	fftForward(r.y);
	fftForward(r.z);
	

	// Dealiase high frecuencies	

	dealias(r);

	//shift back
	Delta[0]=-Delta[0];
	Delta[1]=-Delta[1];
	Delta[2]=-Delta[2];
	
	shift(r,Delta);	

}

