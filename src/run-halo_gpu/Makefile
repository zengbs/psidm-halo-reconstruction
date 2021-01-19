CUDIR   := /software/cuda/default
CC      := gcc
CFLAGS  := -O3 -Wall -g
CPP     := g++
NVCC    := $(CUDIR)/bin/nvcc
#NVFLAGS := -O3 -DOCTANT_DECOMPOSE -I$(CUDIR)/include -m64 -arch=compute_75 -code=sm_75 -Xptxas -v 
#NVFLAGS := -O3 -DOCTANT_DECOMPOSE -I$(CUDIR)/include -m64 -arch=compute_75 -code=sm_75 -Xptxas -v -G -g # -G: device debug flag -g: host debug flag 
#NVFLAGS := -O3 -DOCTANT_DECOMPOSE -I$(CUDIR)/include -m64 -arch=compute_52 -code=sm_52 -Xptxas -v  # titanxq
NVFLAGS := -O3 -DOCTANT_DECOMPOSE -I$(CUDIR)/include -m64 -arch=compute_61 -code=sm_61 -Xptxas -v  # gtx1080ti
LDFLAGS :=
#TEST_FLAG=-DTEST_FLAG
#DEBUG_FLAG=-DDEBUG

OBJ=arr.o ylm.o text_io.o main.o solve_eigenvalues.o extension.o run_halo.o

all: RUNHALO

arr.o: arr.cu
	$(NVCC) $(NVFLAGS) $(TEST_FLAG) $(DEBUG_FLAG) -dc arr.cu

ylm.o: ylm.cu
	$(NVCC) $(NVFLAGS) $(TEST_FLAG) $(DEBUG_FLAG) -dc ylm.cu

text_io.o: text_io.cu
	$(NVCC) $(NVFLAGS) $(TEST_FLAG) $(DEBUG_FLAG) -dc text_io.cu

solve_eigenvalues.o: solve_eigenvalues.cu
	$(NVCC) $(NVFLAGS) $(TEST_FLAG) $(DEBUG_FLAG) -dc solve_eigenvalues.cu

main.o: main.cu
	$(NVCC) $(NVFLAGS) $(TEST_FLAG) $(DEBUG_FLAG) -dc main.cu

extension.o: extension.cu
	$(NVCC) $(NVFLAGS) $(TEST_FLAG) $(DEBUG_FLAG) -dc extension.cu

run_halo.o: run_halo.cu
	$(NVCC) $(NVFLAGS) $(TEST_FLAG) $(DEBUG_FLAG) -dc run_halo.cu

RUNHALO: $(OBJ)
	$(NVCC) $(NVFLAGS) $(TEST_FLAG) $(DEBUG_FLAG) -o $@ $^ $(LDFLAGS) -g -lstdc++

clean:
	rm -f *.o RUNHALO

