# for quick test, uncomment the TEST_FLAG, the compile line with TEST_FLAG

CC=mpicxx
OBJ=arr.o ylm.o ubound.o  solve_eigenvalues.o text_io.o main.o extension.o

#TEST_FLAG=-DTEST_FLAG

all: CONF RECONPOT RECONHALO RECONAMP

CONF:
	@echo "=================MKL_FLAG:$(MKL_FLAG)================="
	@echo "=================OCTANT_FLAG:$(OCTANT_FLAG)================="
ifeq ("$(OCTANT_FLAG)","ON")
# compute one octant at time
            CFLAGS =-DOCTANT_DECOMPOSE
else
            CFLAGS = 
endif
ifeq ("$(MKL_FLAG)","ON")
            CFLAGS +=-O3 -Wall -g -mkl=cluster -I/software/intel/default/mkl/include 
            CFLAGS +=-DMKL
            LDFLAGS = 
else
            CFLAGS +=-O3 -Wall -g -I/work1/koarakawaii/LAPACK/lapack/LAPACKE/include
            LDFLAGS =-lm -L/work1/koarakawaii/LAPACK/lapack -llapacke -llapack -ltmglib -lrefblas -lgfortran
endif

.cpp.o:
	$(CC) $(CFLAGS) -c -o $@ $<
#	$(CC) $(CFLAGS) $(TEST_FLAG) -c -o $@ $<

RECONPOT: $(OBJ) run_pot.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) -lstdc++
#	$(CC) $(CFLAGS) $(TEST_FLAG) -o $@ $^ $(LDFLAGS) -lstdc++
	
RECONHALO: $(OBJ) run_halo.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) -g -lstdc++

RECONAMP: $(OBJ) run_amp.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) -g -lstdc++
clean:
	rm -f *.o RECONPOT RECONHALO RECONAMP TEST
