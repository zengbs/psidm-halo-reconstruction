/* add timing for total computation 2020.12.27 */
/* add timing for eigen-problem 2020.12.29 */
/* comment out store_ylm 2020.12.30 */
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<mpi.h>
#include"macros.h"
#include"text_io.h"
#include"main.h"
#include"arr.h"
#include"solve_eigenvalues.h"
#include"ylm.h"
#include"ubound.h"

void run_amp(void);

void run(void)
{
    run_amp();
}

void run_amp(void)
{
// timing 2020.12.27
        double start, stop, total_time; 
        if(rank==0)
            start = MPI_Wtime();
//
	solve_init();
	
	int l_max=0;
	int l_init=0;
	int glo_l_max=0;

	/*** solve eigenfunctions and eigenvalues in different nodes ***/
// add flexibility to read different file name of final potential 
	l_max = do_solve(true,final_pot);
// timing for solving eigen-problem 2020.12.29
        if (rank==0)
        {
            stop = MPI_Wtime();
            total_time = stop-start;
            int hr = (int)(total_time/3600);
            int min = (int)((total_time-3600*hr)/60);
            double sec = total_time - 3600.*hr - 60.*min;
            printf("Eigen problem solved.\n");
            printf("Total computation time for solving eigen problem is %d hr %d min %.2f s .\n", hr, min, sec);
            fflush(stdout);
        }
//
//	l_init=rank%lnod*lsiz;
// modify lsiz calculation 2020.12.25
        int lnodidx = rank%lnod;
        if (rank<l_rest)    
            l_init=lnodidx*lsiz;
        else    
            l_init=(lsiz+1)*l_rest+(lnodidx-l_rest)*lsiz;
//
	
	rho_init(rho);
	ylm_init(l_init,l_max);

	/*** save eigenfunctions in different nodes ***/
    	store_eigenstates(l_max, "egn");
	printf("eigenfunction stored! l_max=%d,l_init=%d rank=%d\n",l_max,l_init,rank);
        fflush(stdout);

	/*** generate amplitudes in different nodes ***/
	gen_amp(l_init,l_max,A,beta,Ec,mu0,amp_gs_r,amp_gs_i,true);
//	read_amp(l_max,"amp_rcon.bin");
//	printf("l_max=%d l_init=%d rank=%d\n",l_max,l_init,rank);//debug
	printf("gen_amp,rank=%d\n",rank);
        fflush(stdout);
	
	/*** show "amp_rcon.bin" and "amp_rcon.txt" in different nodes***/  //seems like only store_ylm matters for simulation (used by RECONHALO), show_ylm look like producing log file merely.
//  comment out show_ylm 2020.12.30
//	show_ylm(l_init,l_max, "amp_rcon",-1);
	store_ylm(l_init,l_max,"amp_only");
	printf("amplitudes files finished\n");
        fflush(stdout);
	free_egn(l_max,l_init);
	ylm_fin(l_init,l_max);
	solve_fin();

// timing for total computation time 2020.12.27
        if (rank==0)
        {
            stop = MPI_Wtime();
            total_time = stop-start;
            int hr = (int)(total_time/3600);
            int min = (int)((total_time-3600*hr)/60);
            double sec = total_time - 3600.*hr - 60.*min;
            printf("Total computation time for reconstructing amplitude is %d hr %d min %.2f s .\n", hr, min, sec);
        }
//
	return;
}
