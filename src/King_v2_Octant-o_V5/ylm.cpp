/* modify lsiz calculation so sum of lsiz over all ranks equals Lsiz 2020.12.25 */
/* comment out unnecessary calculation 2020.12.30 */
/* add timing for cal_ylm() and expand_ylm() and timing_flag for do_expand_ylm() 2020.12.31 */
/* add debug information in expand_ylm() 2020.12.31 */
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<mpi.h>
#include"macros.h"
#include"main.h"
#include"ylm.h"
#include<string.h>

void cal_index_r(double r, int *index_r, double *frac_r);


/*********************
  Y^m_l(x,y,z)
*********************/
double **ylmr   ;
double **ylmi   ;
//double **pow_ylm;
/****
 components
 ****/
cmpx ***amplitude;

void ylm_init(int l_init,int l_max)
{

//    int size_tmp = l_max + 1;
    ylmr      = (double**)malloc((l_max+1)*sizeof(double*));
    ylmi      = (double**)malloc((l_max+1)*sizeof(double*));
    amplitude = (cmpx***) malloc((l_max+1-l_init)*sizeof(cmpx**));

    int l,m,n;

    for(l = 0;l <= l_max;l++)
    {
        ylmr[l]        = (double*)  malloc((2*l+1)*sizeof(double));
        ylmi[l]        = (double*)  malloc((2*l+1)*sizeof(double));

	if(l<l_init) continue;

        amplitude[l-l_init] = (cmpx**) malloc((rfunc_num[l-l_init])*sizeof(cmpx*));

        cmpx *amplitude_buf = (cmpx*) malloc((rfunc_num[l-l_init])*(2*l+1)*sizeof(cmpx));
        memset(amplitude_buf, 0, rfunc_num[l-l_init]*(2*l+1)*sizeof(cmpx));

//	printf("rank=%d l=%d rfunc_num[%d]=%d\n",rank,l,l-l_init,rfunc_num[l-l_init]);//debug
        for(n = 0;n < rfunc_num[l-l_init];n++)
        {
            amplitude[l-l_init][n] = amplitude_buf;
            amplitude_buf += (2*l+1);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    return;
}

void ylm_fin(int l_init,int l_max)
{
    int l,m,n;

    for(l = l_init;l <= l_max;l++)
    {
	int ll=l-l_init;
        free(amplitude[ll][0]);

        free(ylmr[ll]     );
        free(ylmi[ll]     );
        free(amplitude[ll]);
    }

    free(ylmr       );
    free(ylmi       );
    free(amplitude  );

    return;
}

void store_eigenstates(int l_max, char *filename)
{
    int l,n,m;
    FILE *pf;
    char str[80];
    int l_init;
    int lnodidx;

    lnodidx=rank%lnod;
    
//    l_init=lnodidx*lsiz;
// modify lsiz calculation 2020.12.25
    if (rank<l_rest)
        l_init=lnodidx*lsiz;
    else
        l_init=(lsiz+1)*l_rest+(lnodidx-l_rest)*lsiz;
//

    if(rank == lnodidx)
    {
	sprintf(str,"%s%d.bin",filename,lnodidx);
        pf = fopen(str, "wb");
    }
    if(rank == lnodidx)
    {
	fwrite(&l_init, 1, sizeof(int), pf);
	fwrite(&l_max, 1, sizeof(int), pf);
	fwrite(rfunc_num, l_max+1-l_init, sizeof(int), pf);
	for(l = l_init;l <= l_max;l++)
	{
	    	fwrite(rv[l-l_init], rfunc_num[l-l_init], sizeof(double), pf);
		for(n = 0;n < rfunc_num[l-l_init];n++)
		{
			fwrite(rfunc[l-l_init][n], rsiz, sizeof(double), pf);
		}
	}
    }
    if(rank == lnodidx)
        fclose(pf);

    MPI_Barrier(MPI_COMM_WORLD);

    return;
}
void free_egn(int l_max,int l_init)
{
//	int lnodidx=rank%lnod;
//	int l_init=lnodidx*lsiz;
	for(int l=l_init;l<=l_max;l++)
	{
//			printf("rank=%d rfunc_num[%d]=%d\n",rank,l-l_init,rfunc_num[l-l_init]);//debug
	    for(int n=0;n<rfunc_num[l-l_init];n++)
                free(rfunc[l-l_init][n]);
	    free(rv[l-l_init]);
	}
	free(rv);
	free(rfunc_num);
	free(rfunc);
	printf("free egn\n");//debug
    	MPI_Barrier(MPI_COMM_WORLD);
}

void load_eigenstates(int *l_init,int *l_max, char *filename,int lnodidx)
{
    int l,n,m;
    FILE *pf;
    char str[80];
    if(rank == lnodidx)
    {
//	printf("In load_eigen rank=%d\n",rank);//debug
	sprintf(str,"%s%d.bin",filename,lnodidx);
//	printf("str=%s\n",str);//debug
        pf = fopen(str, "rb");
//	printf("file opened!\n");//debug 
	fread(l_init, 1, sizeof(int), pf);
        fread(l_max, 1, sizeof(int), pf);
//	printf("l_init=%d l_max=%d rank=%d\n",*l_init,*l_max,rank);//debug
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    MPI_Bcast(l_init, 1, MPI_INT, lnodidx, MPI_COMM_WORLD);
    MPI_Bcast(l_max , 1, MPI_INT, lnodidx, MPI_COMM_WORLD);
    printf("l_init=%d l_max=%d rank=%d\n",*l_init,*l_max,rank);//debug

    rfunc_num = (int*)      malloc((*l_max+1-*l_init) * sizeof(int)     );//rfunc_num[l] is the number of eigenvalues
    rv        = (double**)  malloc((*l_max+1-*l_init) * sizeof(double*) );//rv[l][n] is the eigenvalue_ln
    rfunc     = (double***) malloc((*l_max+1-*l_init) * sizeof(double**));//rfunc[l][n] is the radial eigenfunction(function of r

    if(rank == lnodidx)
        fread(rfunc_num, *l_max+1-*l_init, sizeof(int), pf);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(rfunc_num, *l_max+1-*l_init, MPI_INT, lnodidx, MPI_COMM_WORLD);
//    printf("rfunc_num! rank=%d\n",rank);//debug
    for(l = *l_init;l <= *l_max;l++)
    {
//	printf("rank=%d l=%d\n",rank,l);//debug
	rv   [l-*l_init] = (double*)  malloc((rfunc_num[l-*l_init]) * sizeof(double) );
	rfunc[l-*l_init] = (double**) malloc((rfunc_num[l-*l_init]) * sizeof(double*));
	if(rank == lnodidx)
	    fread(rv[l-*l_init], rfunc_num[l-*l_init], sizeof(double), pf);
    	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Bcast(rv[l-*l_init], rfunc_num[l-*l_init], MPI_DOUBLE, lnodidx, MPI_COMM_WORLD);
//	printf("rank=%d rv[%d][0]=%e\n",rank,l-*l_init,rv[l-*l_init][0]);//debug
	for(n = 0;n < rfunc_num[l-*l_init];n++)
	{
	    rfunc[l-*l_init][n] = (double*) malloc((rsiz) * sizeof(double));
	    if(rank == lnodidx)
		fread(rfunc[l-*l_init][n], rsiz, sizeof(double), pf);
    	    MPI_Barrier(MPI_COMM_WORLD);
	    MPI_Bcast(rfunc[l-*l_init][n], rsiz, MPI_DOUBLE, lnodidx, MPI_COMM_WORLD);
	}
    }
//	printf("rfunc complete! rank=%d\n",rank);//debug

    if(rank == lnodidx)
        fclose(pf);
    MPI_Barrier(MPI_COMM_WORLD);
    return;
}

void store_ylm(int l_init, int l_max, char *filename)
{
    int l,m,n;
    FILE *pf;
    char filen[80];
    int lnodidx=rank%lnod;

    if(rank == lnodidx){
	sprintf(filen,"%s%s%d%s",filename,"L",lnodidx,".bin");
        pf = fopen(filen, "wb");
    }

    for(l = l_init;l <= l_max;l++)
    {
	int ll=l-l_init;
        void *addr_of_buf = amplitude[ll][0];
        int   size_of_buf = rfunc_num[ll] * (2 * l + 1) * 2;
//        MPI_Allreduce(MPI_IN_PLACE, addr_of_buf, size_of_buf, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        if(rank ==lnodidx )
            fwrite(addr_of_buf, sizeof(double), size_of_buf, pf);
    }
    if(rank == lnodidx)
        fclose(pf);

    return;
}

void load_ylm(int l_init, int l_max, char *filename,int lnodidx)
{
    int l,m,n;
    FILE *pf;
    char filen[80];

    if(rank == lnodidx)
    {
	sprintf(filen,"%s%s%d%s",filename,"L",lnodidx,".bin");
        pf = fopen(filen, "rb");
    }
    for(l = l_init;l <= l_max;l++)
    {
        void *addr_of_buf = amplitude[l-l_init][0];
        int   size_of_buf = rfunc_num[l-l_init] * (2 * l + 1) * 2;
        if(rank == lnodidx)
            fread(addr_of_buf, sizeof(double), size_of_buf, pf);
        MPI_Bcast(addr_of_buf, size_of_buf, MPI_DOUBLE, lnodidx, MPI_COMM_WORLD);
    }
    if(rank == lnodidx)
        fclose(pf);

    return;
}

void show_ylm(int l_init,int l_max, char *filename,int iter)
{
    FILE *pf;
    FILE *pf_text;
    int n,l,m;
    char filen[80];
    
    int lnodidx=rank%lnod;

    if(rank == lnodidx)
    {
	sprintf(filen,"%s%s%d%s",filename,"L",lnodidx,".bin");
        pf = fopen(filen, "wb");
    }
    if(rank == lnodidx)
    {
// change the way naming the file 2020.12.28
//        sprintf(filen,"%s%d%s","amp_rconL",lnodidx,".txt");
        sprintf(filen,"%s%s%d%s",filename,"L",lnodidx,".txt");
//
//		printf("filename=%s\n",filen);//debug
        pf_text = fopen(filen, "w");
    }

    for(l = l_init;l <= l_max;l++)
        for(n = 0;n < rfunc_num[l-l_init];n++)
            for(m = 0; m < l * 2 + 1;m++)
            {
                int mm;
                mm = m > l ? m - (2*l+1) : m;
                double eigenvalue = rv[l-l_init][n];
                double amp_r = amplitude [l-l_init][n][m][0];
                double amp_i = amplitude [l-l_init][n][m][1];
                double amp_r_out = amp_r;
                double amp_i_out = amp_i;
//                MPI_Reduce(&amp_r, &amp_r_out, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
//                MPI_Reduce(&amp_i, &amp_i_out, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
//                MPI_Barrier(MPI_COMM_WORLD);
                if(rank == lnodidx)
                {
                    fwrite(&l ,         sizeof(int),    1, pf);
                    fwrite(&n ,         sizeof(int),    1, pf);
                    fwrite(&mm,         sizeof(int),    1, pf);
                    fwrite(&eigenvalue, sizeof(double), 1, pf);
                    fwrite(&amp_r_out , sizeof(double), 1, pf);
                    fwrite(&amp_i_out , sizeof(double), 1, pf);

                    fprintf(pf_text, "%5d ", l );
                    fprintf(pf_text, "%5d ", n );
//  change the order of n and mm 2020.12.28
                    fprintf(pf_text, "%5d ", mm);
//
                    fprintf(pf_text, "%15.10le ", eigenvalue);
                    fprintf(pf_text, "%15.10le ", amp_r_out );
                    fprintf(pf_text, "%15.10le ", amp_i_out );
                    fprintf(pf_text, "\n");
                }
            }
    if(rank == lnodidx)
        fclose(pf);
    if(rank == lnodidx)
        fclose(pf_text);

    MPI_Barrier(MPI_COMM_WORLD);

    return;
}

void cal_index_r(double r, int *index_r, double *frac_r)
{
    double rr = r/dr;
    int i = (int)floor(rr);
    if(i >= rsiz)
        i = rsiz - 1;
    if(i <= 0)
        i = 1;
    *frac_r = rr - i;
    *index_r = i;
    return;
}

void decompose_ylm(double r, double **ylm, double arr_r, double arr_i, int l_max)
{
    int n,l,m;

    int index_r;
    double frac_r;
    cal_index_r(r, &index_r, &frac_r);

    for(l = 0;l <= l_max;l++)
    {
        if(l > ubound_l)
            break;
        for(n = 0;n < rfunc_num[l];n++)
        {
            if(n > ubound_n[l])
                break;
            if(l <= lbound_l)
                if(n <= lbound_n[l])
                    continue;
            double rfunc_interp = rfunc[l][n][index_r-1] * (1-frac_r) + rfunc[l][n][index_r] * (frac_r);
            double rfunc_arr[2];
            rfunc_arr[0] = rfunc_interp * arr_r;
            rfunc_arr[1] = rfunc_interp * arr_i;
            for(m = 0; m < l*2 + 1;m++)
            {
                amplitude [l][n][m][0] += rfunc_arr[0] * ylm[l][m];
                amplitude [l][n][m][1] += rfunc_arr[1] * ylm[l][m];
            }
        }
    }
    return;
}

void expand_ylm(double r, double **ylm, double *arr_r, double *arr_i, int l_max,int l_init)
{
    int n,l,m;

    int index_r;
    double frac_r;
//    if(rank==18)printf("in expand_ylm rank=%d l_init=%d,l_max=%d\n",rank,l_init,l_max);//debug
    cal_index_r(r, &index_r, &frac_r);

    *arr_r = 0;
    *arr_i = 0;

    for(l = l_init;l <= l_max;l++)
    {
//	if(rank==18)printf("l=%d,rank=%d,ubound_l=%d,ubound_n[0]=%d,lbound_l=%d\n",l,rank,ubound_l,ubound_n[0],lbound_l);//debug
//	if(rank==18)printf("rfuc_num[0]=%d,rank=%d\n",rfunc_num[0],rank);//debug
        if(l > ubound_l)
        {
// add debug information 2020.12.31
            if (rank==0)
                printf("l larger than ubound_l!\n");
//
            break;
        }
	int ll=l-l_init;
        for(n = 0;n < rfunc_num[ll];n++)
        {
            if(n > ubound_n[ll])
            {
// add debug information 2020.12.31
                if (rank==0)
                    printf("n larger than ubound_n!\n");
//
                break;
            }
            if(l <= lbound_l)
            {
// add debug information 2020.12.31
                if (rank==0)
                    printf("l smaller than lbound_l!\n");
//
                if(n <= lbound_n[ll])
                {
// add debug information 2020.12.31
                    if (rank==0)
                        printf("n smaller than lbound_n!\n");
//
                    continue;
                }
            }
//	    if(rank==18)printf("l=%d,n=%d,rfunc_num[%d]=%d\n",l,n,ll,rfunc_num[ll]);//debug
//	    printf("n=%d,ylm[%d][%d]=%e rank=%d\n",n,l,m,ylm[l][m],rank);//debug
            double rfunc_interp = rfunc[ll][n][index_r-1] * (1-frac_r) + rfunc[ll][n][index_r] * (frac_r);
            double ampl_ylm[2];
            ampl_ylm[0] = 0;
            ampl_ylm[1] = 0;
            for(m = 0; m < l*2 + 1;m++)
            {
                ampl_ylm[0] += amplitude [ll][n][m][0] * ylm[l][m];
                ampl_ylm[1] += amplitude [ll][n][m][1] * ylm[l][m];
            }
//	    printf("rfunc_interp=%e rank=%d\n",rfunc_interp,rank);//debug
//	    printf("ampl_ylm[0]=%e rank=%d\n",ampl_ylm[0],rank);//debug
            *arr_r += ampl_ylm[0] * rfunc_interp;
            *arr_i += ampl_ylm[1] * rfunc_interp;
        }
//		printf("rank=%d,l=%d\n",rank,l);//debug
    }
//    printf("end of expand_ylm, rank=%d\n",rank);//debug
//    MPI_Barrier(MPI_COMM_WORLD);
    return;
}

void do_decompose_ylm(double x, double y, double z, double r, double arr_r, double arr_i, int l_max)
{
//    cal_ylm(x,y,z,l_max);
    cal_ylm(x,y,z,ubound_l);
    if(real_ylm)
    {
        decompose_ylm( r, ylmr, arr_r, arr_i, l_max);
    }
    else
    {
        decompose_ylm( r, ylmr, arr_r, arr_i, l_max);
        decompose_ylm( r, ylmi, arr_i,-arr_r, l_max);
    }
    return;
}

// add timing_flag 2020.12.31
void do_expand_ylm(double x, double y, double z, double r, double *arr_r, double *arr_i, int l_max,int l_init, bool timing_flag)
{
// add timing 2020.12.31
    double start, stop;
    if (timing_flag)
       start = MPI_Wtime();
//
// generate ylmr at (x,y,z)
//    cal_ylm(x,y,z,ubound_l);
//    printf("rank=%d before cal_ylm\n",rank);
    cal_ylm(x,y,z,l_max);
//    printf("rank=%d after cal_ylm\n",rank);
// add timing 2020.12.31
    if (timing_flag)
    {
        stop = MPI_Wtime();
        printf("Time for calculating ylm @ rank %d is %.4e s.\n", rank, stop-start);
    }
//
    
    double r_tmp;
    double i_tmp;

    if(real_ylm)
    {
// add timing 2020.12.31
        if (timing_flag)
           start = MPI_Wtime();
//
        expand_ylm( r, ylmr, &r_tmp, &i_tmp, l_max,l_init);
//	printf("r_tmp=%e,rank=%d\n",r_tmp,rank);//debug
        *arr_r = r_tmp;
        *arr_i = i_tmp;
// add timing 2020.12.31
        if (timing_flag)
        {
            stop = MPI_Wtime();
            printf("Time for expanding ylm @ rank%d is %.4e s.\n", rank, stop-start);
        }
//
    }
    else
    {
// add timing 2020.12.31
        if (timing_flag)
           start = MPI_Wtime();
//
        expand_ylm( r, ylmr, &r_tmp, &i_tmp, l_max,l_init);
        *arr_r = r_tmp;
        *arr_i = i_tmp;
        expand_ylm( r, ylmi, &r_tmp, &i_tmp, l_max,l_init);
        *arr_r += -i_tmp;
        *arr_i +=  r_tmp;
// add timing 2020.12.31
        if (timing_flag)
        {
            stop = MPI_Wtime();
            printf("Time for expanding ylm @ rank%d is %.4e s.\n", rank, stop-start);
        }
//
    }
//	printf("end of do_expand_ylm,rank=%d\n",rank);//debug
    return;
}


/*
 The primary function in this library.
 This function generates all the Ylm(\theta,\phi) values up to l = l_max, for a given x,y,z coordinate.
 The x,y,z vector is normalised to unit length first.
 Values are calculated using iterative method provided by new version NR,
 which would prevent unstatability underflow issues.
 */

void cal_ylm(double x,double y,double z, int l_max)
{
    int l,m;
    int msb;
/*********************
 phaser with factors
  exp(i m phi)*(x^2+y^2)^(m/2)
*********************/
    double *phr;
    double *phi;
/*********************
  P^m_m(z)
*********************/
    double *pmm;
/*********************
  P^m_l(z)
*********************/
    double **pml;
    pml = (double **)malloc ((l_max+1)*sizeof(double*));
    pmm = (double *) malloc ((l_max+1)*sizeof(double));
    phr = (double *) malloc ((l_max+1)*sizeof(double));
    phi = (double *) malloc ((l_max+1)*sizeof(double));
    for(l = 0; l <= l_max; l++)
    {
        pml[l] = (double *)malloc ((2*l+1)*sizeof(double));
    }

    double sqfac;
    double r;

    r = x*x + y*y + z*z;
    r = sqrt(r);

    x/=r;
    y/=r;
    z/=r;

//  comment out unnecessary calculation 2020.12.30
//    double rr = x*x + y*y;
//    rr = sqrt(rr);
//    double xx = x/rr;
//    double yy = y/rr;
//    double sqrt_rr = sqrt(rr);
    msb = 1;
    double fact = -1;
//  seems that phr acts like cos, phi acts like sin, cos(a+b)=cos(a)cos(b)-sin(a)sin(b); sin(a+b)=sin(a)cos(b)+cos(a)sin(b)
    for(l = 0;l <= l_max;l++)
    {
        m = l;
        if(m==0)
            pmm[m] = 1.;
        else
            pmm[m] = fact * pmm[m-1] / (fact+1.);
        fact += 2;

        if(m==0)
        {
            phr[m] = 1;
            phi[m] = 0;
        }
        else if(m==1)
        {
            phr[m] = x;
            phi[m] = y;
        }
        else
        {
            int m1,m2;
            if(m == msb*2)
                msb = m;
            m1 = msb;
            m2 = m^msb;
            if(m2 == 0)
            {
                m1 = m1 >> 1;
                m2 = m1;
            }
            phr[m] = phr[m1]*phr[m2] - phi[m1]*phi[m2] ;
            phi[m] = phi[m1]*phr[m2] + phr[m1]*phi[m2] ;
        }
    }

    sqfac = 1;
    for(m = 0;m <= l_max;m++)
    {
        pmm[m] = sqrt((2*m+1)*pmm[m]/(4*M_PI));
        if(m%2 == 1)
            pmm[m] = pmm[m];

        double oldfact;
        for(l = m;l <= l_max;l++)
        {
            if(l > m)
                fact = sqrt((4.*l*l-1.)/(l*l-m*m));
            if(m == l)
                pml[l][m] = pmm[m];
            else if(m == l-1)
                pml[l][m] = fact * z*pml[l-1][m];
            else
                pml[l][m] = fact * (z*pml[l-1][m] - pml[l-2][m]/oldfact);
            oldfact = fact;

            if(real_ylm)
            {
                if(m!=0)
                {
                    ylmr[l][m]        = pml[l][m] * phr[m] * sqrt(2.);
                    ylmr[l][-m+2*l+1] = pml[l][m] * phi[m] * sqrt(2.);
                }
                else
                {
                    ylmr[l][m]        = pml[l][m] * phr[m];
                }
            }
            else
            {
                ylmr[l][m] = pml[l][m] * phr[m];
                ylmi[l][m] = pml[l][m] * phi[m];
                if(m!=0)
                {
                    ylmr[l][-m+2*l+1] = pml[l][m] * phr[m];
                    ylmi[l][-m+2*l+1] =-pml[l][m] * phi[m];
                }
            }
        }
    }

    for(l = 0; l <= l_max; l++)
        free(pml[l]);
    free(phr);
    free(phi);
    free(pmm);
    free(pml);
    return;
}
/*
 * generate amplitudes from King model
 */
void gen_amp(int l_init,int l_max,double A,double beta,double Ec,double mu0,double amp_gs_r,double amp_gs_i,bool israndom)
{
    	int lnodidx=rank%lnod;
	double Glo_totalmass=0;
	if(rank==0) printf("mu0=%e\n",mu0);
	
	for(int masscount=0;masscount<2;masscount++){
		double totalmass=0;
//		if(rank==0) printf("A=%e,beta=%e,Ec=%e,eta0=%e\n",A,beta,Ec,eta0);//debug
		for(int l=l_init;l<=l_max;l++){
			int lpr=l-l_init;
			void *addr_of_buf = amplitude[lpr][0];
			int   size_of_buf = rfunc_num[l-l_init] * (2 * l + 1) * 2;
			if(rank==lnodidx)
			{
			for(int n=0;n<rfunc_num[l-l_init];n++){
				for(int m=0;m<l*2+1;m++){
//					double std=King(rv[l][n],A,beta,Ec);
					double std=Fermionic_King(rv[l-l_init][n],A,beta,Ec,mu0);
//					std=sqrt(std/2.);//2(std^2)=DF(E)
//					printf("std=%e\n",std);//debug
					if(l==0&&n==0){//ground state
						amplitude[lpr][n][m][0]=amp_gs_r;//real part of amplitude
						amplitude[lpr][n][m][1]=amp_gs_i;//imaginary part
					}
					else if(israndom){
	//				continue;//debug
						amplitude[lpr][n][m][0]=rand_normal(0.0,sqrt(std/2.));//real part of amplitude
						amplitude[lpr][n][m][1]=rand_normal(0.0,sqrt(std/2.));//imaginary part
					}
					else{
						amplitude[lpr][n][m][0]=sqrt(std/2.);//real part of amplitude
						amplitude[lpr][n][m][1]=sqrt(std/2.);//imaginary part
//						printf("hi\n");//debug
					}
				        totalmass+=(amplitude[lpr][n][m][0]*amplitude[lpr][n][m][0]+
	   		                amplitude[lpr][n][m][1]*amplitude[lpr][n][m][1]);
//					printf("totalmass=%e\n",totalmass);//debug
				}
			}
			}
//			MPI_Bcast(addr_of_buf, size_of_buf, MPI_DOUBLE, lnodidx, MPI_COMM_WORLD);
//			printf("l=%d rank=%d\n",l,rank);//debug
		}
        	MPI_Barrier(MPI_COMM_WORLD);
   		MPI_Reduce(&totalmass,&Glo_totalmass, 1, MPI_DOUBLE,MPI_SUM, 0, MPI_COMM_WORLD);
		
		if(rank==0)
		{
  		        printf("global total mass=%e\n",Glo_totalmass);//print total mass
//			printf("global total mass=%e; gs_mass=%e\n",Glo_totalmass, amplitude[0][0][0][0]*amplitude[0][0][0][0]+amplitude[0][0][0][1]*amplitude[0][0][0][1]);//print total mass and core_mass up to r_vir, for debug
	   	        A=A*halomass/Glo_totalmass;
		}
        	MPI_Barrier(MPI_COMM_WORLD);
		MPI_Bcast(&A, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	}
        MPI_Barrier(MPI_COMM_WORLD);
	return;
}
void read_amp(int l_max,char *filename)
{
    FILE *pf;
    FILE *pf_text;
    int n,l,m;

    if(rank == 0)
        pf = fopen(filename, "rb");
	

    for(l = 0;l <= l_max;l++)
        for(n = 0;n < rfunc_num[l];n++)
            for(m = 0; m < l * 2 + 1;m++)
            {
//		printf("l=%d n=%d m=%d\n",l,n,m);//debug
                int mm,ll,nn;
                mm = m > l ? m - (2*l+1) : m;
                double eigenvalue;
                double amp_r_in;
                double amp_i_in;
                if(rank == 0)
                {
                    	fread(&ll,         sizeof(int),    1, pf);
                    	fread(&nn,         sizeof(int),    1, pf);
                    	fread(&mm,         sizeof(int),    1, pf);
                    	fread(&eigenvalue, sizeof(double), 1, pf);
                    	fread(&amp_r_in  , sizeof(double), 1, pf);
                   	fread(&amp_i_in  , sizeof(double), 1, pf);
		}
                MPI_Bcast(&amp_r_in, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);	
                MPI_Bcast(&amp_i_in, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);	
                amplitude [l][n][m][0] = amp_r_in;
                amplitude [l][n][m][1] = amp_i_in;
            }
    if(rank == 0)
        fclose(pf);
    MPI_Barrier(MPI_COMM_WORLD);
    return;
};//end read_amp
