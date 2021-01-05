/* add timing 2020.12.27 */
/* remove extern int Lidx_global 2020.12.29 */
/* adding ignoring all operations in the for loop if l_max<l_init 2020.12.29 */
/* adding timing for each and all Lidx loops 2020.12.31 */
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

// remove Lidx_global 2020.12.29 */
//extern int Lidx_global;
//

void run_halo(void);

void run(void)
{
    run_halo();
}

void run_halo(void)
{
// timing 2020.12.27
    double start, stop, eigen_copy_start, eigen_copy_stop, lidx_start, lidx_stop, lidx_time_acc;
    if (rank==0)
        start = MPI_Wtime();
//
    solve_init();
    if(rank==0)
        gen_r(r_1,r);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(r, rsiz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(r_1, rsiz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    //printf("r[0]=%e 0\n",r[0]);//debug
	
    int l_max=0;
    int l_init=0;
    int glo_l_max=0;
/*
    //// solve eigenfunctions and eigenvalues in different nodes ////
    l_max = do_solve(true,"pot_selfcon.bin");
    l_init=rank%lnod*lsiz;
    
    rho_init(rho);
    ylm_init(l_init,l_max);
    
    //// save eigenfunctions in different nodes ////
    store_eigenstates(l_max, "egn");
    printf("eigenfunction stored! l_max=%d,l_init=%d rank=%d\n",l_max,l_init,rank);
    
    //// generate amplitudes in different nodes ////
    gen_amp(l_init,l_max,A,beta,Ec,mu0,amp_gs_r,amp_gs_i,true);
//    read_amp(l_max,"amp_rcon.bin");
//    printf("l_max=%d l_init=%d rank=%d\n",l_max,l_init,rank);//debug
    printf("gen_amp,rank=%d\n",rank);
	
    //// show "amp_rcon.bin" and "amp_rcon.txt" in different nodes////
    show_ylm(l_init,l_max, "amp_rcon",-1);
    store_ylm(l_init,l_max,"amp_only");
    printf("amplitudes files finished\n");
    free_egn(l_max);
    ylm_fin(l_init,l_max);
*/		
    /// call r_max ///
#ifdef OCTANT_DECOMPOSE
    cal_r_max_O();
#else
    cal_r_max();
#endif
//    printf("r[0]=%e 1\n",r[0]);//debug
    /*** array initial ***/
    array_init();
    	
    for(int Lidx=0;Lidx<lnod;Lidx++)
    {
        Lidx_global = Lidx;           
// add timing for each Lidx loop and eigen_copy 2020.12.31
        if (rank==0)
        {
            lidx_start = MPI_Wtime();
            eigen_copy_start = MPI_Wtime();
        }
/*
        if ( rank == 0 )
        {
            printf( "Working on Lidx %2d/%2d ...\n", Lidx+1, lnod );
            fflush( stdout );
        }
*/
        /*** load eigenstates and broadcast to every nodes***/
        load_eigenstates(&l_init,&l_max,"egn",Lidx);
        if (rank==0)
        {
            eigen_copy_stop = MPI_Wtime();
            printf("Total time for copying eigen states from l_init=%d to l_max=%d is %.4e s.\n", l_init, l_max, eigen_copy_stop-eigen_copy_start);
        }
//        printf("load_eigenstates,rank=%d\n",rank);
//  ignoring all operations if l_max<l_init 2020.12.29
        if (l_max<l_init)
        {
            if (rank==0)
                printf("l_init = %d @ Lidx = %d will be ignored since l_max < l_init !\n", l_init, Lidx);
            MPI_Barrier(MPI_COMM_WORLD);
            continue;
        }
        /*** generate upper bound ***/
        gen_ubound(rv[0][0] * 0., l_max,l_init);
        ylm_init(l_init,l_max);
        /*** read amplitudes and Bcast ***/
        load_ylm(l_init,l_max,"amp_only",Lidx);
//        printf("load_ylm,rank=%d\n",rank);
        /*** generate density in simulation box ***/
        if(Lidx==0)
            array_add_ylm(l_max,l_init,true);
        else
            array_add_ylm(l_max,l_init,false);
//        printf("array finished,#%d\n",Lidx);
//        printf("r[0]=%e 2\n",r[0]);//debug
        if(Lidx!=(lnod-1))
        {
            ylm_fin(l_init,l_max);
            free_egn(l_max,l_init);
        }
// add timing for each Lidx loop 2020.12.31 
        if (rank==0)
        {
            lidx_stop = MPI_Wtime();
            lidx_time_acc += lidx_stop-lidx_start;
            printf("Total time for running through l_init=%d to l_max=%d takes %.4e s.\n",l_init, l_max, lidx_stop-lidx_start);
        }
//
    }
// add timing for all Lidx loops 2020.12.31
    if (rank==0)
    {
        int lidx_hr = (int)(lidx_time_acc/3600);
        int lidx_min = (int)((lidx_time_acc-3600*lidx_hr)/60);
        double lidx_sec = lidx_time_acc - 3600.*lidx_hr - 60.*lidx_min;
        printf("Total time for running through all Lidx loops is %d hr %d min %.2f s .\n", lidx_hr, lidx_min, lidx_sec);
    }
//
    /// generate density profile and potential ///
    array_pro(rho); 
//    printf("r[0]=%e 3\n",r[0]);//debug
//    printf("array_pro end!\n");//debug
    gen_output(rho,true);
    show_output(-1,false);
    
    /// write array into file_r and file_i///
    write_array(file_r,file_i);
    printf("array files finished\n");
    
    //write_slices
//    write_sliceXY("Real_sliceXY128","Imag_sliceXY128");
//    write_sliceYZ("Real_sliceYZ128","Imag_sliceYZ128");
/*
    /// set 2nd level box at center ///
    set_2ndbox();
    
    /// add ylm to the box ///
    array_add_ylm(l_max,l_init);
    
    /// generate density profile and potential ///
    array_pro(rho); 
    gen_output(rho,true);
    show_output(-2,false);
    
    /// write array into file_r and file_i///
    write_array(file_r,file_i);
    printf("array files 2 finished\n");
*/	
    solve_fin();
// timing 2020.12.27
    if (rank==0)
    {
        stop = MPI_Wtime();
        double total_time = stop-start;
        int hr = (int)(total_time/3600);
        int min = (int)((total_time-3600*hr)/60);
        double sec = total_time - 3600.*hr - 60.*min;
        printf("Total computation time for reconstructing halo is %d hr %d min %.2f s .\n", hr, min, sec);
    }
//
    return;
}
