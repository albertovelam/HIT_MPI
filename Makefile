CC = mpic++
NVCC = nvcc
LD = mpic++
LIBS = -lcudart -lcufft -lcublas -lcuda  -lstdc++ -lm -lhdf5  -lhdf5 -lhdf5_hl
PATHS = -L/opt/cuda/lib64/ -L/usr/lib64 -L/usr/lib
INCLUDES = -I/opt/cuda/include
DEBUG = -g

all: calc_conv_rotor.o check.o F.o hit_mpi.o main.o phaseShift.o RK2.o routineCheck.o statistics.o calc_U_W.o dealias.o fft.o kernelCheck.o memory.o random.o RK_kernels.o setUp.o statisticsKernels.o
	$(LD) $(DEBUG) -o hitMPI *.o $(PATHS) $(LIBS)

calc_conv_rotor.o: 
	$(NVCC) $(INCLUDES) -c src/calc_conv_rotor.cu

calc_U_W.o: 
	$(NVCC) $(INCLUDES) -c src/calc_U_W.cu

dealias.o: 
	$(NVCC) $(INCLUDES) -c src/dealias.cu

kernelCheck.o: 
	$(NVCC) $(INCLUDES) -c src/kernelCheck.cu

phaseShift.o: 
	$(NVCC) $(INCLUDES) -c src/phaseShift.cu

RK_kernels.o: 
	$(NVCC) $(INCLUDES) -c src/RK_kernels.cu

routineCheck.o: 
	$(NVCC) $(INCLUDES) -c src/routineCheck.cu

statisticsKernels.o: 
	$(NVCC) $(INCLUDES) -c src/statisticsKernels.cu

check.o: 
	$(CC) $(INCLUDES) -c src/check.c

F.o: 
	$(CC) $(INCLUDES) -c src/F.c

fft.o: 
	$(CC) $(INCLUDES) -c src/fft.c

hit_mpi.o: 
	$(CC) $(INCLUDES) -c src/hit_mpi.c

main.o: 
	$(CC) $(INCLUDES) -c src/main.c

memory.o: 
	$(CC) $(INCLUDES) -c src/memory.c

random.o: 
	$(CC) $(INCLUDES) -c src/random.c

RK2.o: 
	$(CC) $(INCLUDES) -c src/RK2.c

setUp.o: 
	$(CC) $(INCLUDES) -c src/setUp.c

statistics.o: 
	$(CC) $(INCLUDES) -c src/statistics.c

clean:
	rm *.o hitMPI
