/* fix the bug in gen_output and do_perturb 2020.12.16 */
/* modify function Set_GS_rfunc (use external variable h0, r_c) and drop some unncessary calculation 2020.12.16*/
/* modify the lsiz calculation so sum of lsiz over all rank equals to Lsiz 2020.12.25 (only available for LNOD=MPI_THREAD)*/
/* add print e_max 2021.01.02 */

#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include"macros.h"
#include"text_io.h"
#include"main.h"
#include"arr.h"
#include"solve_eigenvalues.h"

double *r    ;
double *r_1  ;
double *pot  ;
double *pot_arti;//reconstructed potential
double *pot_l;
double *rho  ;

double kfac;
double lfac;
double h_over_m;
double g_a;
