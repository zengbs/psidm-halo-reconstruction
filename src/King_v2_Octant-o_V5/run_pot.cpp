/* add reading initial potential guess filename */
/* add include header extension.h 2020.12.12 */
/* add free memory and convergence check 2020.12.14 */
/* add print soliton wave function 2020.12.17 */
/* add fflush to renew the error convergence log more frequently 2020.12.21 */
/* add timing 2020.12.22 */
/* add print out extra physical quantities 2020.12.23 */
/* modify lsiz calculation so sum of lsiz over all ranks equal Lsiz 2020.12.25 */
/* add output final potential profile 2020.12.27 */
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
// add header extension.h 2020.12.12
#include "extension.h"

void run_pot(void);

void run(void)
{
    run_pot();
}

void run_pot(void)
{
// timing 2020.12.22
	double start, stop;

	if(rank==0)
	{
// timing 2020.12.22
	    start = MPI_Wtime();
	    printf("perturb fraction=%e\n",perturb_frac);
	}
	clock_t t;	
	int l_max=0;
	int l_init=0;
// modify lsiz calculation 2020.12.25
        int lnodidx=0;
//
	int glo_l_max=0;
	
#ifdef TEST_FLAG
	gen_init_pot_guess(init_pot);
        gen_init_den_profile("den_profile_init.bin");
#else
	gen_init_pot_guess(init_pot);
        gen_init_den_profile("den_profile_init.bin");
	solve_init();

	bool firstsolve=true;
// adding convergence check 2020.12.14
        bool converge_flag=false;
        double error;
//        FILE *error_output = fopen(convergence_file,"w");
        FILE *error_output = fopen("Record__Convergence","w");
//
	for(int iter=iter_i;iter<=iter_max;iter++)//iteration
	{
		l_max = do_solve(firstsolve,init_pot);
//		l_init=rank%lnod*lsiz;
// modify lsiz calculation 2020.12.25
                lnodidx = rank%lnod;
	        if (rank<l_rest)
	            l_init=lnodidx*lsiz;
	        else
//
	            l_init=(lsiz+1)*l_rest+(lnodidx-l_rest)*lsiz;
		printf("eigenfunction complete l_init=%d rank=%d!\n",l_init,rank);

		/// set rho[i]=0 ///
		rho_init(rho);

		/// allocate memory of ylm and amp /// all nodes have different amplitude array
//		glo_l_max=Global_l_max(l_max);//find global l_max for this system
		//if(rank==0) printf("glo_l_max=%d\n",glo_l_max);//debug
		ylm_init(l_init,l_max);
		if(rank==0)printf("allocate ylm and amp\n");

		/// generate amp and rho ///
		gen_arti_halo(t,l_init,l_max);      // rho calculated here has been multiplied by r[i]^2*dr
		if(rank==0)printf("rho complete\n");//debug

		/// print out soliton wave function 2020.12.17 modified to print out only once 2020.12.22
//		if (rank==0)
		if (rank==0&&iter==0)
		{
// adding print out sliton wave function 2020.12.17
		    char swf_filename [strsiz];
                    FILE *soliton_wf;
      		    printf("Print out soliton wave function:\n");
//      		    sprintf(swf_filename, "soliton_wf_%d.txt", iter);
      		    sprintf(swf_filename, "soliton_wf.txt");
      		    soliton_wf = fopen(swf_filename, "w");
      		    for (int r_i=0; r_i<rsiz; r_i++) 
//      		        fprintf(soliton_wf,"%.16e\t%.16e\n", r[r_i], pow(rfunc[0][0][r_i],2.)/(4.*M_PI*pow(r[r_i],2.)*dr)*(pow(amplitude[0][0][0][0],2.)+pow(amplitude[0][0][0][1],2.)));
      		        fprintf(soliton_wf,"%.16e\t%.16e\n", r[r_i], pow(rfunc[0][0][r_i],2.)/4./M_PI*(pow(amplitude[0][0][0][0],2.)+pow(amplitude[0][0][0][1],2.)));
                    fclose(soliton_wf);
		}
		MPI_Barrier(MPI_COMM_WORLD);

		/// generate pot_arti ///
		gen_output(rho,true);  // arti = true
		if(rank==0) printf("gen_output\n");

		/// show rho and pot ///
		show_output(iter,false);  // iter = false
		if(rank==0)printf("show output\n");

		/// do eigenvalue perturbation ///
		do_perturb(l_init,l_max);
		if(rank==0) printf("do perturb rank=%d\n",rank);

		/// generate amp and rho ///
		gen_arti_halo(t,l_init,l_max);

		/// generate pot ///
		gen_output(rho, false);  // arti = false
		if(rank==0)printf("gen_output\n");

		/// show potential and density profile ///
		show_output(iter,true);  // iter = true
		if(rank==0)printf("show output\n");

		firstsolve=false;

		/// free memory of ylm and amp ///
		ylm_fin(l_init,l_max);

		/// free rfunc, rv, and rfunc /// 2020.12.14
		for (int l=0; l<=l_max-l_init; l++)
		{
		    free(rv[l]);
		    for (int ev=0; ev<rfunc_num[l]; ev++)
		       free(rfunc[l][ev]);
		    free(rfunc[l]);
		}
		free(rfunc_num);
                free(rv);
		free(rfunc);

                /// do convergence check 2020.12.14 and change to current position on 2020.12.24///
		if(rank==0)
                {
                    printf("gen_output\n");
                    if ( (error = convergence_check()) <= criteria )
                        converge_flag = true;
//                        converge_flag = false;  // for convergence error test
		    fprintf(error_output, "%d\t%.8e\n", iter, error);
		    fflush(error_output);
                }
		MPI_Bcast(&converge_flag, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);

                // add print out extra physical quantities 2020.12.23
		if(rank==0) 
		{
                    print_out_extra_quantities(iter);
                    printf("iteration %d done, error is %.8e\n",iter, error);
		}
                // add convergence check 2020.12.14
                if (converge_flag)
                {
                    if (rank==0)
                        printf("Convergence criteria is reached with error %.8e ! Cease @ iteration %d .\n", error, iter);
                    break;
	        }
		MPI_Barrier(MPI_COMM_WORLD);
	}
// add output final potential profile
        if (rank==0)
            gen_final_pot_profile(final_pot);
//
	solve_fin();
#endif
// timing 2020.12.22
        if (rank==0)
        {
	    stop = MPI_Wtime();
	    double total_time = stop-start;
            int hr = (int)(total_time/3600);
	    int min = (int)((total_time-3600*hr)/60);
	    double sec = total_time - 3600.*hr - 60.*min;
            printf("Total computation time for reconstructing potential is %d hr %d min %.2f s .\n", hr, min, sec);
	}

	return;
}
