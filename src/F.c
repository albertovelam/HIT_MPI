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
/*
	fftBackward(u.x);
	fftBackward(u.y);
	fftBackward(u.z);
*/
#if 0
START_RANGE("FFTbt",4)
	fftBack1T(u.x);
END_RANGE
START_RANGE("FFTbt",4)
	fftBack1T(u.y);
END_RANGE
START_RANGE("FFTbt",4)
	fftBack1T(u.z);
END_RANGE
#endif
#if 1
        fftBack1T_A(u.x,0);
        fftBack1T_A(u.y,1);
        fftBack1T_A(u.z,2);
        fftBack1T_B(u.x,0);
        fftBack1T_B(u.y,1);
        fftBack1T_B(u.z,2);
#endif
#if 0
        fftBack1T_A(u.x,0);
        fftBack1T_B(u.x,0);
cudaDeviceSynchronize();
        fftBack1T_A(u.y,1);
        fftBack1T_B(u.y,1);
cudaDeviceSynchronize();
        fftBack1T_A(u.z,2);
        fftBack1T_B(u.z,2);
cudaDeviceSynchronize();
#endif
	// Fill with zeros
START_RANGE("dealias",3)	
	dealias(r);	
END_RANGE	
	// Transform w to real space

/*	
	fftBackward(r.x);
	fftBackward(r.y);	
	fftBackward(r.z);
*/
#if 0
START_RANGE("FFTbt",4)
	fftBack1T(r.x);
END_RANGE
START_RANGE("FFTbt",4)
	fftBack1T(r.y);
END_RANGE
START_RANGE("FFTbt",4)
	fftBack1T(r.z);	
END_RANGE	
#endif
#if 1
        fftBack1T_A(r.x,0);
        fftBack1T_A(r.y,1);
        fftBack1T_A(r.z,2);
        fftBack1T_B(r.x,0);
        fftBack1T_B(r.y,1);
        fftBack1T_B(r.z,2);
#endif
#if 0
        fftBack1T_A(r.x,0);
        fftBack1T_B(r.x,0);
cudaDeviceSynchronize();
        fftBack1T_A(r.y,1);
        fftBack1T_B(r.y,1);
cudaDeviceSynchronize();
        fftBack1T_A(r.z,2);
        fftBack1T_B(r.z,2);
cudaDeviceSynchronize();

#endif
	// Calculate the convolution rotor of u and w

START_RANGE("conv_rotor",2)
	calc_conv_rotor(r,u);
END_RANGE
	// Transform rotor to fourier space

/*	
	fftForward(r.x);
	fftForward(r.y);
	fftForward(r.z);
*/
#if 0
START_RANGE("FFTft",2)
	fftForw1T(r.x);
END_RANGE
START_RANGE("FFTft",2)
	fftForw1T(r.y);
END_RANGE
START_RANGE("FFTft",2)
	fftForw1T(r.z);	
END_RANGE
#endif
#if 1
        fftForw1T_A(r.x,0);
        fftForw1T_A(r.y,1);
        fftForw1T_A(r.z,2);
        fftForw1T_B(r.x,0);
        fftForw1T_B(r.y,1);
        fftForw1T_B(r.z,2);
#endif
#if 0
        fftForw1T_A(r.x,0);
        fftForw1T_B(r.x,0);
cudaDeviceSynchronize();
        fftForw1T_A(r.y,1);
        fftForw1T_B(r.y,1);
cudaDeviceSynchronize();
        fftForw1T_A(r.z,2);
        fftForw1T_B(r.z,2);
cudaDeviceSynchronize();
#endif
	// Dealiase high frecuencies	
START_RANGE("dealias",3)
	dealias(r);
END_RANGE
START_RANGE("SHIFT_delta",6)
	//shift back
	Delta[0]=-Delta[0];
	Delta[1]=-Delta[1];
	Delta[2]=-Delta[2];
END_RANGE
START_RANGE("SHIFT_func",1)	
	shift(r,Delta);	
END_RANGE
}

