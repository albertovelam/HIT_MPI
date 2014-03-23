CC = mpic++ -O3
NVCC = nvcc -O3  -arch=sm_20
LD = mpic++ -O3
LIBS = -lcudart -lcufft -lcublas -lcuda  -lstdc++ -lm -lhdf5  -lhdf5 -lhdf5_hl -lconfig
PATHS = -L/opt/cuda/lib64/ -L/usr/lib64 -L/usr/lib
INCLUDES = -I/opt/cuda/include
DEBUG = -g
NSS:= $(shell python stripNSS.py)
SIZE = -DNSS=${NSS}
GPU_SOURCES = $(wildcard src/*.cu)
CPU_SOURCES = $(wildcard src/*.c)
GPU_OBJECTS = $(GPU_SOURCES:.cu=.o)
CPU_OBJECTS = $(CPU_SOURCES:.c=.o)


all: $(GPU_OBJECTS) $(CPU_OBJECTS)
	$(LD) -o hitMPI $(CPU_OBJECTS) $(GPU_OBJECTS) $(PATHS) $(LIBS)

$(CPU_OBJECTS): src/%.o: src/%.c
	$(CC) -c $(INCLUDES) $(SIZE) $< -o $@

$(GPU_OBJECTS): src/%.o: src/%.cu
	$(NVCC) -c $(INCLUDES) $(SIZE) $< -o $@

clean:
	rm src/*.o hitMPI
