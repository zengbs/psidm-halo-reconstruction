/* fix the bug in gen_output and do_perturb 2020.12.16 */
/* modify function Set_GS_rfunc (use external variable h0, r_c) and drop some unncessary calculation 2020.12.16*/
/* modify the lsiz calculation so sum of lsiz over all rank equals to Lsiz 2020.12.25 (only available for LNOD=MPI_THREAD)*/
/* add print e_max 2021.01.02 */

#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#ifdef MKL
#include<mkl_lapacke.h>
#else
#include<lapacke.h>
#endif
#include<mpi.h>
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

/*
 *  Redefine rho[i] = dr * 4*pi * rho(r[i]) * r[i]^2 
 */
double pot_diff(void)
{
	double diff=0.;
	if(rank==0)
		for(int i=0;i<rsiz;i++)
			diff+=(pot[i]-pot_arti[i]);
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Bcast(&diff, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	return diff;	
}//end of pot_diff
void rho_init(double* rho)
{
	for(int ir=0;ir<rsiz;ir++)
		rho[ir]=0.;
	return;
}


void den_profile(double* rho,int l_init,int l_max)
{
	int lnodidx=rank%lnod;
	double *loc_rho;
	loc_rho=(double*) malloc(rsiz*sizeof(double));

	if(rank==lnodidx) printf("density profile start rank=%d\n",rank);
//	double imaginary=0.;
	
	for(int ir=0;ir<rsiz;ir++)
	{
		loc_rho[ir]=0.;
//		if(ir%size==rank)
//		{
//		if(ir%10==0) printf("ir=%d rank=%d\n",ir,rank);

		for(int l=l_init;l<=l_max;l++)
		{
			int ll=l-l_init;
//			printf("in den_profile rank=%d l=%d\n",rank,l);//debug
			for(int n=0;n<rfunc_num[l-l_init];n++)
			{
				for(int m=0;m<2*l+1;m++)
				{
						loc_rho[ir]+=((amplitude[ll][n][m][0]*amplitude[ll][n][m][0]+
							 amplitude[ll][n][m][1]*amplitude[ll][n][m][1])*
							 rfunc[ll][n][ir]*rfunc[ll][n][ir]*dr*r[ir]*r[ir]);
				}
			}
		}
//		printf("imaginary=%e\n",imaginary);
//		imaginary=0.;
//		printf("rho[%d]=%e\n",ir,rho[ir]);//debug
//		}
	}
//	printf("in den_profile rank=%d\n",rank);//debug
    	MPI_Barrier(MPI_COMM_WORLD);
   	MPI_Reduce(loc_rho,rho, rsiz, MPI_DOUBLE,MPI_SUM, 0, MPI_COMM_WORLD);

	free(loc_rho);
	if(rank==lnodidx)	printf("density profile end\n");
    	MPI_Barrier(MPI_COMM_WORLD);
	return;
}

void do_perturb(int l_init,int l_max)  // bug fixed 2020.12.16
{
    int l,n;
    double delta_rv=0.;

    for(l = l_init;l <= l_max;l++)
    {
//        if(rank == 0)
//            fwrite(rv[l], rfunc_num[l], sizeof(double), pf);
        for(n = 0;n < rfunc_num[l-l_init];n++)
        {
//                fwrite(rfunc[l][n], rsiz, sizeof(double), pf);
		delta_rv=eigenv_perturb(l,n,l_init)*perturb_frac;
//		printf("delta_rv=%e l=%d n=%d\n",delta_rv,l,n);
		rv[l-l_init][n]=rv[l-l_init][n]+delta_rv;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    return;
}
double eigenv_perturb(int l,int n,int l_init)
{
	double eigenv_1=0.;
	
	for(int ri=0;ri<rsiz;ri++)
	{
		eigenv_1+=dr*r[ri]*r[ri]*(pot[ri]-pot_arti[ri])*rfunc[l-l_init][n][ri]*rfunc[l-l_init][n][ri];
	}

	return eigenv_1;
}//end eigenv_perturb

int solve_init(void)
{
    r     = (double*) malloc(rsiz*sizeof(double));
    r_1   = (double*) malloc(rsiz*sizeof(double));
    pot   = (double*) malloc(rsiz*sizeof(double));
    pot_arti= (double*) malloc(rsiz*sizeof(double));
    pot_l = (double*) malloc(rsiz*sizeof(double));
    rho   = (double*) malloc(rsiz*sizeof(double));

    if(eta == 0.)
        h_over_m = planck_h / mass;
    else
        h_over_m = 1. / eta;
    if(scale_fac == 0.)
        g_a = newton_g / (red_shift + 1.);
    else
        g_a = newton_g * scale_fac;
    kfac = h_over_m / dr;
    lfac = h_over_m;
    kfac = kfac * kfac;
    lfac = lfac * lfac;

    return 0;
}

int solve_fin(void)
{
    free(r    );
    free(r_1  );
//    free(pot  );
//    free(pot_arti);
    free(pot_l);
    free(rho  );
    return 0;
}

void Set_GS_rfunc(double GSrfunc[]) // modified 2020.12.16
{
	double core_mass=0.;
	for(int i=0;i<rsiz;i++)
	{
//	GSrfunc[i]=sqrt(4.*M_PI*1.9/pow(mass/1.e-23,2)/pow((r_c*1.e3/h),4)/pow((1.+9.1e-2*pow(r[i]/r_c,2)),8)
	GSrfunc[i]=sqrt(1./pow(mass/1.e-23,2)/pow((r_c*1.e3/h0),4)/pow((1.+9.1e-2*pow(r[i]/r_c,2)),8));
	core_mass+=GSrfunc[i]*GSrfunc[i]*r[i]*r[i]*dr;
	}
	for(int i=0;i<rsiz;i++)
		GSrfunc[i]=GSrfunc[i]/sqrt(core_mass);

	return;
}
int do_solve(bool readpot,char *potfile)
{
    int l;
    int i,j;
    double *ev_buf;
    double *ef_buf;
    double e_max;
    int total_count = 0;
    FILE *pf1;
    FILE *pf2;
    FILE *pf3;
    char str[80];
    int l_init;//solve eigenfunctions in different nodes
    int lnodidx;//solve eigenfunctions in different nodes
    
    lsiz=Lsiz/lnod;
// modify lsiz calculation 2020.12.25
    if (rank<l_rest)
       lsiz ++;
//
    lnodidx=rank%lnod;

    if(rank == lnodidx)
    {
	sprintf(str,"eval%d.txt",lnodidx);
        pf1 = fopen(str,"w");
    }
    if(rank == lnodidx)
    {
	sprintf(str,"efun%d.txt",lnodidx);
        pf2 = fopen(str,"w");
    }
    if(rank == lnodidx)
    {
	sprintf(str,"star%d.txt",lnodidx);
        pf3 = fopen(str,"w");
    }

    double vfac = 1. / sqrt(dr);

    if(rank == 0)
    {
//	r_1 is the reciprocal of r, i.e. 1/r
	if(readpot==true)
    	{
        	gen_r(r_1, r);
        	load_pot(pot,potfile);
                printf("Loading potential from file %s completed.\n", potfile);
                fflush(stdout);
    	}
    	else
    	{
	        gen_r(r_1, r);
//        	gen_pot(pot, rho, r_1, g_a);
    	}
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(r, rsiz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(r_1, rsiz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(pot, rsiz, MPI_DOUBLE, 0, MPI_COMM_WORLD);

//	printf("hi\n");//debug

//    char file_rho[strsiz] = "rho.txt"; 
    char file_pot[strsiz] = "pot_input.txt"; 
    if(rank == 0 && readpot==true)
    {
        double *(val[2]);
        val[0] = r;
//        val[1] = rho;
//        show_txt(NULL, 0, val, 2, rsiz, file_rho);
        val[1] = pot;
        show_txt(NULL, 0, val, 2, rsiz, file_pot);
    }

    rfunc_num = (int*)      malloc(lsiz*sizeof(int));
    rv        = (double**)  malloc(lsiz*sizeof(double*));
    rfunc     = (double***) malloc(lsiz*sizeof(double**));

    e_max = -1e100;
    if(r_eigen==0.) r_eigen=nbox/2*box_dr;

    for(i = 0;i < rsiz;i++)
    {
        if(i * dr < r_eigen)
            e_max = max(e_max,pot[i]);
    }
//  add print out e_max 2021.01.02
    if (rank==0)
        printf("EMAX = %.8e \n", e_max);
//

//    e_max *= 2;

// modify lsiz calculation 2020.12.25
    if (rank<l_rest) 
        l_init=lnodidx*lsiz;
    else 
        l_init=(lsiz+1)*l_rest+(lnodidx-l_rest)*lsiz;
//

    ev_buf = (double*) malloc(nsiz * 2 * sizeof(double));
    ef_buf = (double*) malloc(rsiz * nsiz * 2 * sizeof(double));


//    for(lnodidx=1;lnodidx<lnod;lnodidx++)
//    {
    l = l_init;
    while(1)
    {
        int eg_count=0;
            
        if(rank == lnodidx)
        {
            gen_ang(pot_l, pot, r_1, l);
            solve_eigenvalue(pot_l, e_max, &eg_count, ev_buf, ef_buf);
//            if(rank==1) printf("rank=%d,l=%d,l_init=%d\n",rank,l,l_init);//debug
        }
//        MPI_Bcast(pot_l, rsiz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
//        MPI_Bcast(&eg_count, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        rfunc_num[l-l_init] = eg_count;
	

        if(eg_count > nsiz)
        {
//            fprintf(stderr,"N ERR\n");
            printf("N ERR @ rank %d with l=%d .\n", rank, l);  // for debug
            eg_count = nsiz;
        }
            
        rv   [l-l_init] = (double*)  malloc(eg_count*sizeof(double));
        rfunc[l-l_init] = (double**) malloc(eg_count*sizeof(double*));
        for(i = 0;i < eg_count;i++)
            rfunc[l-l_init][i] = (double*) malloc(rsiz*sizeof(double));

        for(i = 0;i < eg_count;i++)
        {
            if(rank == lnodidx)
            {
                rv[l-l_init][i] = ev_buf[i];
            }
            if(rank == lnodidx)
            {
                double norm_buf = 0;  // normalize the eigen function
                for(j = 0;j < rsiz;j++)
                {
                    norm_buf += ef_buf[i*rsiz + j] * ef_buf[i*rsiz + j];
                }
                norm_buf = sqrt(norm_buf);
                for(j = 0;j < rsiz;j++)
                {
                    ef_buf[i*rsiz + j] /= norm_buf;
                }
		if(l==0 && i==0){
			Set_GS_rfunc(rfunc[l][i]);//fix ground state radial wavefunction
		}
		else{
			for(j = 0;j < rsiz;j++)
			    rfunc[l-l_init][i][j] = r_1[j] * ef_buf[i*rsiz + j] * vfac;  // later will multiply by dr, so (rfun*r)^2dr will cancel each other
		}
            }
//            MPI_Bcast(rfunc[l][i], rsiz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }
	
//        MPI_Bcast(rv[l], eg_count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
        if(rank == lnodidx)
        {
//            printf("%5d %5d\n", l, eg_count);
            if(show_eigenvalues)
                for(i = 0;i < eg_count;i++)
                    fprintf(pf1, "%5d %5d %15.10le \n", l, i, rv[l-l_init][i]);
            if(show_eigenvectors)
                for(i = 0;i < eg_count;i++)
                    for(j = 0;j < rsiz;j++)
                        fprintf(pf2, "%5d %5d %15.10le %15.10le \n", l, i, r[j], rfunc[l-l_init][i][j]);
            if(show_average_r)
                for(i = 0;i < eg_count;i++)
                {
                    double sum  = 0;
                    double sum2 = 0;
                    for(j = 0;j < rsiz;j++)
                    {
                        double den = rfunc[l-l_init][i][j] * r[j];
                        den = den * den;
                        sum  += den;
                        sum2 += den * (r[j] * r[j]);
                    }
                    fprintf(pf3, "%5d %5d %15.10le \n", l, i, sqrt(sum2/sum) );
                }
        }
    
        total_count += eg_count;
//	printf("rank=%d,l=%d\n",rank,l);//debug
        if(eg_count == 0)
            break;
        l++;
	if((l-l_init)>=lsiz)
	    break;
    }
//    if(rank==lnodidx) printf("l=%d, l_init=%d, rank=%d end solve\n",l,l_init,rank);
//    MPI_Barrier(MPI_COMM_WORLD);
//    }
    
    free(ev_buf);
    free(ef_buf);
    
    if(rank == lnodidx)
    {
        fclose(pf1);
        fclose(pf2);
        fclose(pf3);
    }
    int l_max = l-1;

//    printf("l_max=%d,rank=%d\n",l_max,rank);
    MPI_Barrier(MPI_COMM_WORLD);
    return l_max;
}

int gen_r(double *r_1, double *r)
{
    int i;

    for(i = 0;i < rsiz;i++)
    {
        r[i] = (i+1.) * dr;
        r_1[i] = 1. / r[i];
    }
    return 0;
}

/*******
  The gravitational potential given by a spherical density profile rho(r) is
  V(r) = -G S (4\pi) dr' rho(r') r'^2 /max(r',r).
  Redefine rho[i] = dr * 4\pi * rho(r[i]) * r[i]^2,
  The formula is simplified to V[i] = sum_j rho[j] / max(r[i],r[j]).

  The subroutine reuses the integrals iteratively to calculate the potential with O(rsiz) complexity.
*/


int gen_pot(double *potential, double *rho, double *r_1,double g)
{
    int i;
    double *r1_sum_rho;
    double *sum_r1_rho;
    r1_sum_rho = (double*)malloc(rsiz*sizeof(double));
    sum_r1_rho = (double*)malloc(rsiz*sizeof(double));
//	printf("rho[1]=%d\n",rho[1]);//debug
    for(i = 0;i < rsiz;i++)
    {
        if(i > 0)
            r1_sum_rho[i] = r1_sum_rho[i-1];
        else
            r1_sum_rho[i] = 0;
        r1_sum_rho[i] += rho[i];
    }
    for(i = rsiz-1;i >= 0;i--)
    {
        if(i < rsiz-1)
            sum_r1_rho[i] = sum_r1_rho[i+1];
        else
            sum_r1_rho[i] = 0;
        sum_r1_rho[i] += rho[i] * r_1[i];
        r1_sum_rho[i] *= r_1[i];
    }

    for(i = 0;i < rsiz;i++)
    {
        potential[i] = 0;
        potential[i] += r1_sum_rho[i];
        if(i!=rsiz-1)
            potential[i] += sum_r1_rho[i+1];
        potential[i] *= -g;
    }
    free(r1_sum_rho);
    free(sum_r1_rho);

	printf("gen_pot end!\n");//debug
    return 0;
}

int gen_ang(double *pot_l, double *pot, double *r_1, int l)
{
    int i;
    for(i = 0;i < rsiz;i++)
    {
        pot_l[i] = pot[i] + l*(l+1)/2*(r_1[i]*r_1[i])*lfac;
//	if(rank==0)printf("pot_l[%d]=%e\n",i,pot_l[i]);//debug
    }
    return 0;
}

/**
 use the convension that dr = 1, m = 1, 4pi G = 1, M = 1,
 ptr *d is the diagonal elements; ptr *e is the sub-diagonal elements.
 **/
int solve_eigenvalue(double *pot, double e_max, int *eg_count, double *eigenvalues, double *eigenfunctions)
{
    int i;
    double *d;
    double *e;

    double *w = eigenvalues;
    double *z = eigenfunctions;
    d = (double*)malloc(rsiz*sizeof(double));
    e = (double*)malloc(rsiz*sizeof(double));
    
    for(i=0;i < rsiz;i++)
    {
        d[i] = kfac;
        if(i != rsiz-1)
            e[i] = -.5*kfac;
    }
    for(i = 0;i < rsiz;i++)
    {
        d[i] += pot[i];
//	if(i%1000==0)printf("e[%d]=%e\n",i,e[i]);//debug
    }
    double vl = 0., vu = 0.;
    int il = 0,iu = 0;
    int m;
    double abstol = 0;
    int ifail[rsiz*2];
    double e_min = 1e100;
    for(i = 0;i < rsiz;i++)
    {
        e_min = min(e_min,pot[i]);
    }

    vl = e_min;
    vu = e_max;

    if(vl > vu)
    {
        m = 0;
        *eg_count = m;
        return 0;
    }
//    if(rank==1) printf("rank=%d vl=%e vu=%e\n",rank,vl,vu);//debug

/*  LAPACK reference: http://www.netlib.org/lapack/explore-html/dd/d67/a18579_ga874417315bccf2de7547e30338da4101.html
    solve the eigen values and vectors for a symmetric tridiagonal matrix
*/
    LAPACKE_dstevx ( LAPACK_COL_MAJOR, 'V', 'V', rsiz, d, e, vl, vu, il, iu, abstol, &m, w, z, rsiz, ifail);
//    if(rank==1) printf("rank=%d eg_count=%d\n",rank,m);//debug

    *eg_count = m;

    free(d);
    free(e);
    return 0;
}

void load_pot(double *pot,char potfile[])
{
	FILE *pf;
//	printf("in load_pot\n");//debug
		
        
	printf("Potential guess initial file is %s\n", potfile);//debug
	pf=fopen(potfile,"rb");
	fread(pot,rsiz,sizeof(double),pf);
//	printf("read pot\n");//debug
}

void gen_output(double *rho,bool arti)  // bug fixed 2020.12.16
{
	if(rank == 0)
	{
//		gen_r(r_1, r);
		if(arti==false)
		    gen_pot(pot,rho,r_1,g_a);
		else
	 	    gen_pot(pot_arti,rho,r_1,g_a);
//		printf("rho[5]=%e\n",rho[5]);//debug
	}

	MPI_Barrier(MPI_COMM_WORLD);
        if (arti==false)
            MPI_Bcast(pot, rsiz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        else
   	    MPI_Bcast(pot_arti, rsiz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
//  	MPI_Bcast(r, rsiz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
//  	MPI_Bcast(r_1, rsiz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
//   	MPI_Bcast(pot, rsiz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
//   	MPI_Bcast(pot_arti, rsiz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	
	return;
}

void show_output(int iteration,bool perturb)
{
	char filename[100];
    	if(rank == 0)
    	{
        	double *(val[2]);
        	val[0] = r;
        	val[1] = rho;
		if(perturb){
			sprintf(filename,"%s%d%s","rho_perturb",iteration,".txt");
			show_txt(NULL, 0, val, 2, rsiz, filename);
			val[1] = pot;
			sprintf(filename,"%s%d%s","pot_perturb",iteration,".txt");
		}
		else{
			sprintf(filename,"%s%d%s","rho",iteration,".txt");
			show_txt(NULL, 0, val, 2, rsiz, filename);
			val[1]  =pot_arti;
			sprintf(filename,"%s%d%s","pot",iteration,".txt");
		}
        	show_txt(NULL, 0, val, 2, rsiz, filename);
    	}
	MPI_Barrier(MPI_COMM_WORLD);
	return;
}
int Global_l_max(int l_max)
{	
	int glo_l_max=l_max;
	int temp=0;
	int l_init;
	// modify lsiz calculation 2020.12.25
	int lnodidx = rank%lnod;
	if (rank<l_rest) 
	    l_init=lnodidx*lsiz;
	else 
	    l_init=(lsiz+1)*l_rest+(lnodidx-l_rest)*lsiz;
	//

	for(int i=1;i<lnod;i++)
	{	
		if(rank==i)
		{
    			//int l_init=i*lsiz;	
			if(l_max>=l_init)
				MPI_Send(&l_max, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
			else
				MPI_Send(&temp, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
		}
		else if(rank==0)
		{
			MPI_Recv(&temp,1,MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			if(l_max<temp) glo_l_max=temp;
		}
	}
	if(rank==0) printf("real l_max=%d\n",glo_l_max);

	MPI_Barrier(MPI_COMM_WORLD);
   	MPI_Bcast(&glo_l_max, 1, MPI_INT, 0, MPI_COMM_WORLD);

	return glo_l_max;
}
