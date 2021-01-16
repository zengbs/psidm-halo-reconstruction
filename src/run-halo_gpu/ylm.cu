#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>
#include<cuda_runtime.h>
#include"macros.h"
#include"main.h"
#include"ylm.h"
#include"arr.h"

int lnodidx = Lidx_global;

__device__ void cal_index_r(int rsiz, float r, float dr, int *index_r, float *frac_r)
{
    float rr = r/dr;
    int i = (int)floor(rr);
    if(i >= rsiz)
        i = rsiz - 1;
    if(i <= 0)
        i = 1;
    *frac_r = rr - i;
    *index_r = i;
    return;
}

#ifdef OCTANT_DECOMPOSE
__global__ void cal_and_expand_ylm_O(int rsiz, int nx, int ny, int nz, int octant, float dx, float dy, float dz, float xcen, float ycen, float zcen, float dr, float r_max, float *arr_r, float *arr_i, int *rfunc_num, double ***rfunc, cmpx ***amplitude, int l_max,int l_init)
#else
__global__ void cal_and_expand_ylm(int rsiz, int nx, int ny, int nz, float dx, float dy, float dz, float xcen, float ycen, float zcen, float dr, float r_max, float *arr_r, float* arr_i, int *rfunc_num, double *** rfunc, cmpx ***amplitude, int l_max, int l_init)
#endif
{
//if (blockIdx.x==0&&blockIdx.y==0&&blockIdx.z==0)
//    printf("%d, %s\n", __LINE__, __FUNCTION__);
    int idx_x = threadIdx.x + blockIdx.x*blockDim.x;
    int idx_y = threadIdx.y + blockIdx.y*blockDim.y;
    int idx_z = threadIdx.z + blockIdx.z*blockDim.z;

//    extern __shared__ float rfunc_interp[];

    if (idx_x<nx&&idx_y<ny&&idx_z<nz)
    {
        float x, y, z, r;
#ifdef OCTANT_DECOMPOSE
        cal_dis_x_O(idx_x, nx, octant, dx, xcen, &x);
        cal_dis_y_O(idx_y, ny, octant, dy, ycen, &y);
        cal_dis_z_O(idx_z, nz, octant, dz, zcen, &z);
#else
        cal_dis_x(idx_x, dx, xcen, &x);
        cal_dis_y(idx_y, dy, ycen, &y);
        cal_dis_z(idx_z, dz, zcen, &z);
#endif
        cal_dis_r(&x, &y, &z, &r);        
//#ifdef DEBUG
//        if (idx_x==(int)(nx/2)&&idx_y==(int)(ny/2)&&idx_z==(int)(nz/2))
//        {
//            printf("Kernel executing...\n");
//            printf("Calculating distance r completed.\n");
//            printf("x = %.4e , y = %.4e , z= %.4e , r = %.4e , r_max = %.4e .\n", x, y, z, r, r_max);
//        }
//#endif
        if (r<r_max)
        {
            int n, l, m, ll;
            int index_r, index_site;
            float frac_r, rfunc_interp;
            float fact_1, fact_2 = 0., oldfact = 0., temp;  // for ylm computation
            float pmm_new, pmm_old, pml, pml_lm1, phr_new, phr_old, phi_new, phi_old;
            float ylmr[2], array_pos[2];

            cal_index_r(rsiz, r, dr, &index_r, &frac_r);
            index_site = get_pos(idx_x, idx_y, idx_z, nx, ny);
//            index_thread = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
//#ifdef DEBUG
//            if (idx_x==(int)(nx/2)&&idx_y==(int)(ny/2)&&idx_z==(int)(nz/2))
//                printf("index_site = %d ; index_r = %d ; frac_r = %.4e .\n", index_site, index_r, frac_r);
//#endif
            array_pos[0] = (float)0.;
            array_pos[1] = (float)0.;

            x /= r;
            y /= r;
            z /= r;
            fact_1 = -(float)1.;

//            int counter = 0;
//            int shift = ln_total_count*index_thread;
//            for (l=0; l<=l_max-l_init;l++)
//            {
//                for (n=0; n<rfunc[l]; n++)
//                    *(rfunc_interp+shift+counter+n) = (float)(rfunc[ll][n][index_r-1])*(1-frac_r) + (float)(rfunc[ll][n][index_r])*(frac_r);
//                counter += rfunc[l];
//            }

            for (m=0; m<=l_max; m++)
            {
                if (m==0)
                {
                    phr_new = (float)1.;
                    phi_new = (float)0.;
                    pmm_new = (float)1.;
                    pmm_old = (float)1.;
                }
                else
                {
                    if (m==1)
                    {
                        phr_new = x;
                        phi_new = y;
                        phr_old = phr_new;
                        phi_old = phi_new;
                    }
                    else
                    {
                        phr_new = phr_old*x - phi_old*y;
                        phi_new = phi_old*x + phr_old*y;
                        phr_old = phr_new;
                        phi_old = phi_new;
                    }
                    pmm_new = fact_1 * pmm_old / (fact_1+(float)1.);
                    pmm_old = pmm_new;
                }
                fact_1 += (float)2.;
        
                pmm_new = sqrtf(((float)2.*m+(float)1.)*pmm_new/((float)4.*(float)M_PI));
        
                for(l = m;l <= l_max; l++)
                {
                    if(l > m)
                        fact_2 = sqrtf(((float)4.*l*l-(float)1.)/(l*l-m*m));
                    if(l == m)
                        pml = pmm_new;
                    else if(l == m+(float)1.)
                    {
                        pml_lm1 = pml;
                        pml = fact_2*z*pml;
//#ifdef DEBUG
//                        if (idx_x==0 && idx_y==0 && idx_z==0)
//                        {
//                            if (m==0&&l==1)
//                                printf("fact_2 = %.6e ; z = %.6e ; pml_lm1 = %.6e ; pml = %.6e .\n", fact_2, z, pml_lm1, fact_2*z*pml_lm1);
//                        }
//#endif
                    }
                    else
                    {
                        temp = pml;
                        pml = fact_2*(z*pml-pml_lm1/oldfact);
                        pml_lm1 = temp;
                    }
                    oldfact = fact_2;
                    if(m!=0)
                    {
                        ylmr[0] = pml * phr_new * sqrtf((float)2.); // ylmr[0] = ylm[l][m]
                        ylmr[1] = pml * phi_new * sqrtf((float)2.); // ylmr[1] = ylmr[l][-m+2*l+1]
                        ll = l-l_init;
                        if (ll>=0)
                        {
                            for (n=0; n<rfunc_num[ll]; n++)
                            {
                                rfunc_interp = (float)(rfunc[ll][n][index_r-1])*((float)1.-frac_r) + (float)(rfunc[ll][n][index_r])*(frac_r);
                                array_pos[0] += ((float)(amplitude[ll][n][m][0])*ylmr[0]+(float)(amplitude[ll][n][-m+2*l+1][0])*ylmr[1])*rfunc_interp;
                                array_pos[1] += ((float)(amplitude[ll][n][m][1])*ylmr[0]+(float)(amplitude[ll][n][-m+2*l+1][1])*ylmr[1])*rfunc_interp;
//#ifdef DEBUG
//                                if (idx_x==(int)(nx/2)&&idx_y==(int)(ny/2)&&idx_z==(int)(nz/2))
//                                {
//                                    if (m==6&&ll==2&&n==10)
//                                        printf("amplitude[%d][%d][%d] = %.6e%+.6e ; amplitude[%d][%d][%d] = %.6e%+.6e ; ylmr[%d][%d] = %.6e ; ylmr[%d][%d] = %.6e ; rfunc_interp = %.6e .\n", l, n, m, (float)(amplitude[ll][n][m][0]), (float)(amplitude[ll][n][m][1]), l, n, -m+2*l+1, (float)(amplitude[ll][n][-m+2*l+1][0]), (float)(amplitude[ll][n][m][1]), l, m, ylmr[0], l, -m+2*l+1, ylmr[1], rfunc_interp);
//                                }
//#endif
                            }
                        }
                    }
                    else
                    {
                        ylmr[0] = pml * phr_new;
                        ll = l-l_init;
                        if (ll>=0)
                        {
                            for (n=0; n<rfunc_num[ll]; n++)
                            {
                                rfunc_interp = (float)(rfunc[ll][n][index_r-1])*(1-frac_r) + (float)(rfunc[ll][n][index_r])*(frac_r);
                                array_pos[0] += ((float)(amplitude[ll][n][0][0])*ylmr[0])*rfunc_interp;
                                array_pos[1] += ((float)(amplitude[ll][n][0][1])*ylmr[0])*rfunc_interp;
                            }
                        }
                    }
//#ifdef DEBUG
//                    if (idx_x==0 && idx_y==0 && idx_z==0)
//                    {
//                        if (m!=0)
//                        {
//                            printf("ylm[%d][%d] = %.6e .\n", l, m, ylmr[0] );
//                            printf("ylm[%d][%d] = %.6e .\n", l, -m+2*l+1, ylmr[1] );
//                        }
//                        else
//                            printf("ylm[%d][%d] = %.6e .\n", l, 0, ylmr[0] );
//                    }
//#endif
                } // end of for loop l
//#ifdef DEBUG
//                if (idx_x==(int)(nx/2) && idx_y==(int)(ny/2) && idx_z==(int)(nz/2))
//                    printf("array_r[%d] = %.4e ; array_i[%d] = %.4e \n", index_site, array_pos[0], index_site, array_pos[1]);
//#endif
            } // end of for loop m
            arr_r[index_site] = array_pos[0];
            arr_i[index_site] = array_pos[1];
        }
    }
    __syncthreads();
#ifdef DEBUG
//    printf("Calculating Expanding ylmr completed on site (%d,%d,%d) .\n", idx_x, idx_y, idx_z);
    if (threadIdx.x==0 && threadIdx.y==0 && threadIdx.z==0 && blockIdx.x==gridDim.x-1 && blockIdx.y==gridDim.y-1)
        printf("Calculating-expanding ylmr completed on grid (%d,%d,%d). \n", blockIdx.x, blockIdx.y, blockIdx.z);
#endif
}

void ylm_init(int l_init,int l_max)
{
    cudaError_t err;
    err = cudaMallocManaged((void**)&amplitude, (l_max+1-l_init)*sizeof(cmpx**));
//    cudaMallocManaged((void**)&amplitude, (l_max+1-l_init)*sizeof(cmpx**));
    if (err!=cudaSuccess)
    {
        printf("Allocation for pointer amplitufe failed!\n");
        exit(1);
    }
#ifdef DEBUG
    else
        printf("Allocation for pointer amplitude succeed!\n");
#endif

    int l,n;
    float *amplitude_n_buff, **amplitude_l_buff;
    for(l = l_init;l <= l_max;l++)
    {
        err = cudaMallocManaged((void**)&amplitude_l_buff, rfunc_num[l-l_init]*sizeof(cmpx*));
//        cmpx *amplitude_buf = (cmpx*) malloc((rfunc_num[l-l_init])*(2*l+1)*sizeof(cmpx));
//        memset(amplitude_buf, 0, rfunc_num[l-l_init]*(2*l+1)*sizeof(cmpx));

//	printf("rank=%d l=%d rfunc_num[%d]=%d\n",rank,l,l-l_init,rfunc_num[l-l_init]);//debug
        for(n = 0;n < rfunc_num[l-l_init];n++)
        {
            err = cudaMallocManaged((void**)&amplitude_n_buff, (2*l+1)*sizeof(cmpx));
            err = cudaMemcpy(&amplitude_l_buff[n], &amplitude_n_buff, sizeof(cmpx*), cudaMemcpyHostToDevice);
//                amplitude[l-l_init][n] = amplitude_buf;
//                amplitude_buf += (2*l+1);
            if (err!=cudaSuccess)
            {
                printf("Memory allocation for amplitude[%d][%d] failed!\n", l, n);
                exit(1);
            }
#ifdef DEBUG
            else
                printf("Memory allocation for amplitude[%d][%d] succeed!\n", l, n);
#endif
        }
        err = cudaMemcpy(&amplitude[l-l_init], &amplitude_l_buff, sizeof(cmpx**), cudaMemcpyHostToDevice);
        if (err!=cudaSuccess)
        {
            printf("Memory allocation for amplitude[%d] failed!\n", l);
            exit(1);
        }
#ifdef DEBUG
        else
            printf("Memory allocation for amplitude[%d] succeed!\n", l);
#endif
    }
    return;
}

void ylm_fin(int l_init,int l_max)
{
    int l;
    for(l = 0;l <= l_max;l++)
    {
	int ll=l-l_init;
//        for (int n=0; n<rfunc_num[ll]; n++)
//        {
//            cudaFree(amplitude[ll][n]);
//            free(amplitude_host[ll][n]);
//        }

        cudaFree(amplitude[ll]);
//        free(amplitude_host[ll]);
    }
    cudaFree(amplitude);
//    free(amplitude_host);
    printf("Amplitude for spherical harmonics from l_init=%d to l_max=%d free.\n", l_init, l_max);
    return;
}

void store_eigenstates(int l_max, char *filename)
{
    int l,n;
    FILE *pf;
    char str[80];
    int l_init;

    sprintf(str,"%s%d.bin",filename,lnodidx);
    pf = fopen(str, "wb");
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
    fclose(pf);

    return;
}
void free_egn(int l_max,int l_init)
{
    for(int l=l_init;l<=l_max;l++)
    {
//			printf("rank=%d rfunc_num[%d]=%d\n",rank,l-l_init,rfunc_num[l-l_init]);//debug
        for(int n=0;n<rfunc_num[l-l_init];n++)
            cudaFree(rfunc[l-l_init][n]);
        cudaFree(rv[l-l_init]);
        cudaFree(rfunc[l-l_init]);
    }
    cudaFree(rv);
    cudaFree(rfunc_num);
    cudaFree(rfunc);
    printf("Eigen values and functions from l_init=%d to l_max=%d free.\n", l_init, l_max);//debug
}

//void load_eigenstates(int *l_init,int *l_max, int *ln_total_count, char *filename,int lnodidx)
void load_eigenstates(int *l_init,int *l_max, char *filename,int lnodidx)
{
    int l,n;
    FILE *pf;
    char str[80];
    double *rv_buff, *rfunc_n_buff, **rfunc_l_buff;
    cudaError_t err;
    sprintf(str,"%s%d.bin",filename,lnodidx);
    pf = fopen(str, "rb");
    fread(l_init, 1, sizeof(int), pf);
    fread(l_max, 1, sizeof(int), pf);
//    *ln_total_count = 0;

    err = cudaMallocManaged((void **)&rfunc_num, (*l_max+1-*l_init)*sizeof(int));//rfunc_num[l] is the number of eigenvalues
    if (err!=cudaSuccess)
    {
        printf("Memory allocation for eigen value number failed!\n");
        exit(1);
    }
#ifdef DEBUG
    else
        printf("Memory allocation for eigen value number succeed!\n");
#endif
    cudaMallocManaged((void **)&rv, (*l_max+1-*l_init)*sizeof(double*));//rv[l][n] is the eigen_ln
    cudaMallocManaged((void **)&rfunc, (*l_max+1-*l_init)*sizeof(double**));//rc[l][n][r] is the eigenfunction_ln

    fread(rfunc_num, *l_max+1-*l_init, sizeof(int), pf);
    for(l = *l_init;l <= *l_max;l++)
    {
        int ll = l-*l_init;
        cudaMallocManaged((void **)&rv_buff, (*l_max+1-*l_init)*sizeof(double));
        err = cudaMemcpy(&rv[ll], &rv_buff, sizeof(double*), cudaMemcpyHostToDevice);
        if (err!=cudaSuccess)
        {
            printf("Memory allocation for eigen values with l = %d failed!\n", l);
            exit(1);
        }
#ifdef DEBUG
        else
            printf("Memory allocation for eigen values with l = %d succeed!\n", l);
#endif
	fread(rv[ll], rfunc_num[ll], sizeof(double), pf);

        cudaMallocManaged((void **)&rfunc_l_buff, rfunc_num[ll]*sizeof(double*));
        cudaMemcpy(&rfunc[ll], &rfunc_l_buff, sizeof(double**), cudaMemcpyHostToDevice);

	for(n = 0;n < rfunc_num[ll];n++)
	{
            cudaMallocManaged((void **)&rfunc_n_buff, rsiz*sizeof(double));
            err = cudaMemcpy(&rfunc[ll][n], &rfunc_n_buff, sizeof(double*), cudaMemcpyHostToDevice);
            if (err!=cudaSuccess)
            {
                printf("Memory allocation for eigen functions with l = %d , n = %d failed!\n", l, n);
                exit(1);
            }
//#ifdef DEBUG
//            else
//                printf("Memory allocation for eigen functions with l = %d , n = %d succeed!\n", l, n);
//#endif
	    fread(rfunc[ll][n], rsiz, sizeof(double), pf);
	}
        if (err!=cudaSuccess)
        {
            printf("Memory allocation for eigen functions with l = %d failed!\n", l);
            exit(1);
        }
#ifdef DEBUG
        else
            printf("Memory allocation for eigen functions with l = %d succeed!\n", l);
//        *ln_total_count += rfunc_num[ll];
#endif
    }
    fclose(pf);
    return;
}

void store_ylm(int l_init, int l_max, char *filename)
{
    int l;
    FILE *pf;
    char filen[80];

    sprintf(filen,"%s%s%d%s",filename,"L",lnodidx,".bin");
    pf = fopen(filen, "wb");

    for(l = l_init;l <= l_max;l++)
    {
	int ll=l-l_init;
        void *addr_of_buf = amplitude[ll][0];
        int   size_of_buf = rfunc_num[ll] * (2 * l + 1) * 2;
        fwrite(addr_of_buf, sizeof(double), size_of_buf, pf);
    }
    fclose(pf);

    return;
}

void load_ylm(int l_init, int l_max, char *filename,int lnodidx)
{
    int l, n;
    FILE *pf;
    char filen[80];
    cmpx ***amplitude_local = (cmpx***)malloc((l_max+1-l_init)*sizeof(cmpx**));
    
//  initailize amplitude_local
    for(l=0; l<=l_max-l_init;l++)
    {
        amplitude_local[l] = (cmpx**) malloc((rfunc_num[l])*sizeof(cmpx*));
        cmpx *amplitude_buf = (cmpx*) malloc((rfunc_num[l])*(2*(l+l_init)+1)*sizeof(cmpx));

//      printf("rank=%d l=%d rfunc_num[%d]=%d\n",rank,l,l-l_init,rfunc_num[l-l_init]);//debug
        for(n = 0;n < rfunc_num[l];n++)
        {
            amplitude_local[l][n] = amplitude_buf;
            amplitude_buf += (2*(l+l_init)+1);
        }
    }
//

// read from bin file and copy to unified memory
    sprintf(filen,"%s%s%d%s",filename,"L",lnodidx,".bin");
    pf = fopen(filen, "rb");
    for(l = l_init;l <= l_max;l++)
    {
        void *addr_of_buf = amplitude_local[l-l_init][0];
        int   size_of_buf = rfunc_num[l-l_init] * (2 * l + 1) * 2;
        fread(addr_of_buf, sizeof(double), size_of_buf, pf);
        for (n=0; n<rfunc_num[l-l_init]; n++)
        {
            for (int m=0; m<2*l+1; m++)
            {
                amplitude[l-l_init][n][m][0] = amplitude_local[l-l_init][n][m][0];
                amplitude[l-l_init][n][m][1] = amplitude_local[l-l_init][n][m][1];
            }
        }
        free(amplitude_local[l-l_init]);
    }
    free(amplitude_local);
    fclose(pf);
//

//#ifdef DEBUG
//    int test_l = 110 , test_n = 10, test_m = 88;
//    int ll = test_l-l_init;
//    printf("Amplitude Test:\n");
//    if (test_m!=0)
//    {
////        printf("Amp[%d][%d][%d] = %.4e%+.4e ; Amp[%d][%d][%d] = %.4e%+.4e .\n", test_l, test_n, test_m, amplitude_local[ll][test_n][test_m][0], amplitude_local[ll][test_n][test_m][1], test_l, test_n, 2*(test_l)+1-test_m, amplitude_local[ll][test_n][2*test_l+1-test_m][0], amplitude_local[ll][test_n][2*test_l+1-test_m][1]);
//        printf("Amp[%d][%d][%d] = %.4e%+.4e ; Amp[%d][%d][%d] = %.4e%+.4e .\n", test_l, test_n, test_m, amplitude[ll][test_n][test_m][0], amplitude[ll][test_n][test_m][1], test_l, test_n, 2*(test_l)+1-test_m, amplitude[ll][test_n][2*test_l+1-test_m][0], amplitude[ll][test_n][2*test_l+1-test_m][1]);
//    }
//    else
//    {
////        printf("Amp[%d][%d][%d] = %.4e%+.4e .\n", test_l, test_n, test_m, amplitude_local[ll][test_n][test_m][0], amplitude_local[ll][test_n][test_m][1]);
//        printf("Amp[%d][%d][%d] = %.4e%+.4e .\n", test_l, test_n, test_m, amplitude[ll][test_n][test_m][0], amplitude[ll][test_n][test_m][1]);
//    }
//#endif
    return;
}

//void show_ylm(int l_init,int l_max, char *filename,int iter)
//{
//    FILE *pf;
//    FILE *pf_text;
//    int n,l,m;
//    char filen[80];
//    
//    int lnodidx=rank%lnod;
//
//    if(rank == lnodidx)
//    {
//	sprintf(filen,"%s%s%d%s",filename,"L",lnodidx,".bin");
//        pf = fopen(filen, "wb");
//    }
//    if(rank == lnodidx)
//    {
//// change the way naming the file 2020.12.28
////        sprintf(filen,"%s%d%s","amp_rconL",lnodidx,".txt");
//        sprintf(filen,"%s%s%d%s",filename,"L",lnodidx,".txt");
////
////		printf("filename=%s\n",filen);//debug
//        pf_text = fopen(filen, "w");
//    }
//
//    for(l = l_init;l <= l_max;l++)
//        for(n = 0;n < rfunc_num[l-l_init];n++)
//            for(m = 0; m < l * 2 + 1;m++)
//            {
//                int mm;
//                mm = m > l ? m - (2*l+1) : m;
//                double eigenvalue = rv[l-l_init][n];
//                double amp_r = amplitude [l-l_init][n][m][0];
//                double amp_i = amplitude [l-l_init][n][m][1];
//                double amp_r_out = amp_r;
//                double amp_i_out = amp_i;
////                MPI_Reduce(&amp_r, &amp_r_out, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
////                MPI_Reduce(&amp_i, &amp_i_out, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
////                MPI_Barrier(MPI_COMM_WORLD);
//                if(rank == lnodidx)
//                {
//                    fwrite(&l ,         sizeof(int),    1, pf);
//                    fwrite(&n ,         sizeof(int),    1, pf);
//                    fwrite(&mm,         sizeof(int),    1, pf);
//                    fwrite(&eigenvalue, sizeof(double), 1, pf);
//                    fwrite(&amp_r_out , sizeof(double), 1, pf);
//                    fwrite(&amp_i_out , sizeof(double), 1, pf);
//
//                    fprintf(pf_text, "%5d ", l );
//                    fprintf(pf_text, "%5d ", n );
////  change the order of n and mm 2020.12.28
//                    fprintf(pf_text, "%5d ", mm);
////
//                    fprintf(pf_text, "%15.10le ", eigenvalue);
//                    fprintf(pf_text, "%15.10le ", amp_r_out );
//                    fprintf(pf_text, "%15.10le ", amp_i_out );
//                    fprintf(pf_text, "\n");
//                }
//            }
//    if(rank == lnodidx)
//        fclose(pf);
//    if(rank == lnodidx)
//        fclose(pf_text);
//
//    MPI_Barrier(MPI_COMM_WORLD);
//
//    return;
//}
//void do_expand_ylm(float *arr_r, float *arr_i, float *calculate_timer, float *expand_timer, int l_max,int l_init, int ln_total_count, bool timing_flag)
void do_expand_ylm(float *arr_r, float *arr_i, float *cal_and_expand_timer, int l_max,int l_init, bool timing_flag)
{
//    int size_shared = tpB*tpB*tpB*sizeof(float)*ln_total_count;
//    if (size_shard>64000)
//    {
//        printf("Size needed by shared memory too large! Exit!\n");
//        exit(1);
//    }
    cudaEvent_t start, stop;
    cudaError_t err;
    float time_temp;
    dim3 tpB(tpB_x, tpB_y, tpB_z);
    dim3 bpG(bpG_x, bpG_y, bpG_z);

    if (timing_flag)
    {
        err = cudaEventCreate(&start);
        err = cudaEventCreate(&stop);
        err = cudaEventRecord(start,0);
    }
    printf("Peek error befor kernel launchded: %s .\n", cudaGetErrorString(err));  

#ifdef OCTANT_DECOMPOSE
    cal_and_expand_ylm_O<<<bpG,tpB>>>(rsiz, nx, ny, nz, octant, (float)dx, (float)dy, (float)dz, (float)xcen, (float)ycen, (float)zcen, (float)dr, (float)r_max, arr_r, arr_i, rfunc_num, rfunc, amplitude, l_max, l_init);
#else
    cal_and_expand_ylm<<<bpG,tpB>>>(rsiz, nx, ny, nz, (float)dx, (float)dy, (float)dz, (float)xcen, (float)ycen, (float)zcen, (float)dr, (float)r_max, arr_r, arr_i, rfunc_num, rfunc, amplitude, l_max, l_init);
#endif

    err = cudaPeekAtLastError();
    if (err!=cudaSuccess)
    {
        int Lidx_temp = Lidx_global-1;
        printf("Calculating-exapnding ylm failed due to error: %s !\n", cudaGetErrorString(err));
        printf("Dump with Lidx = %d .\n", Lidx_temp);
        write_array(true,&Lidx_temp,file_r,file_i);
        exit(1);
    }
#ifdef DEBUG
    else
        printf("Calculating-expanding ylm succeed!\n");
#endif
    cudaDeviceSynchronize();

    if (timing_flag)
    {
        cudaEventRecord(stop,0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time_temp, start, stop);
        printf("Time for expanding ylm is %.4e s.\n", time_temp/1000.);
        *cal_and_expand_timer += time_temp/1000.;
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

///*
// * generate amplitudes from King model
// */
//void gen_amp(int l_init,int l_max,double A,double beta,double Ec,double mu0,double amp_gs_r,double amp_gs_i,bool israndom)
//{
//    	int lnodidx=rank%lnod;
//	double Glo_totalmass=0;
//	if(rank==0) printf("mu0=%e\n",mu0);
//	
//	for(int masscount=0;masscount<2;masscount++){
//		double totalmass=0;
////		if(rank==0) printf("A=%e,beta=%e,Ec=%e,eta0=%e\n",A,beta,Ec,eta0);//debug
//		for(int l=l_init;l<=l_max;l++){
//			int lpr=l-l_init;
//			void *addr_of_buf = amplitude[lpr][0];
//			int   size_of_buf = rfunc_num[l-l_init] * (2 * l + 1) * 2;
//			if(rank==lnodidx)
//			{
//			    for(int n=0;n<rfunc_num[l-l_init];n++){
//				for(int m=0;m<l*2+1;m++){
////					double std=King(rv[l][n],A,beta,Ec);
//					double std=Fermionic_King(rv[l-l_init][n],A,beta,Ec,mu0);
////					std=sqrtf(std/2.);//2(std^2)=DF(E)
////					printf("std=%e\n",std);//debug
//					if(l==0&&n==0){//ground state
//						amplitude[lpr][n][m][0]=amp_gs_r;//real part of amplitude
//						amplitude[lpr][n][m][1]=amp_gs_i;//imaginary part
//					}
//					else if(israndom){
//	//				continue;//debug
//						amplitude[lpr][n][m][0]=rand_normal(0.0,sqrtf(std/2.));//real part of amplitude
//						amplitude[lpr][n][m][1]=rand_normal(0.0,sqrtf(std/2.));//imaginary part
//					}
//					else{
//						amplitude[lpr][n][m][0]=sqrtf(std/2.);//real part of amplitude
//						amplitude[lpr][n][m][1]=sqrtf(std/2.);//imaginary part
////						printf("hi\n");//debug
//					}
//				        totalmass+=(amplitude[lpr][n][m][0]*amplitude[lpr][n][m][0]+
//	   		                amplitude[lpr][n][m][1]*amplitude[lpr][n][m][1]);
////					printf("totalmass=%e\n",totalmass);//debug
//				}
//			    }
//			}
////			MPI_Bcast(addr_of_buf, size_of_buf, MPI_DOUBLE, lnodidx, MPI_COMM_WORLD);
////			printf("l=%d rank=%d\n",l,rank);//debug
//		}
//        	MPI_Barrier(MPI_COMM_WORLD);
//   		MPI_Reduce(&totalmass,&Glo_totalmass, 1, MPI_DOUBLE,MPI_SUM, 0, MPI_COMM_WORLD);
//		
//		if(rank==0)
//		{
//  		        printf("global total mass=%e\n",Glo_totalmass);//print total mass
////			printf("global total mass=%e; gs_mass=%e\n",Glo_totalmass, amplitude[0][0][0][0]*amplitude[0][0][0][0]+amplitude[0][0][0][1]*amplitude[0][0][0][1]);//print total mass and core_mass up to r_vir, for debug
//	   	        A=A*halomass/Glo_totalmass;
//		}
//        	MPI_Barrier(MPI_COMM_WORLD);
//		MPI_Bcast(&A, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
//	}
//        MPI_Barrier(MPI_COMM_WORLD);
//	return;
//}
void read_amp(int l_max,char *filename)
{
    FILE *pf;
    int n,l,m;

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

              	fread(&ll,         sizeof(int),    1, pf);
              	fread(&nn,         sizeof(int),    1, pf);
              	fread(&mm,         sizeof(int),    1, pf);
              	fread(&eigenvalue, sizeof(double), 1, pf);
              	fread(&amp_r_in  , sizeof(double), 1, pf);
             	fread(&amp_i_in  , sizeof(double), 1, pf);

                amplitude [l][n][m][0] = (float)(amp_r_in);
                amplitude [l][n][m][1] = (float)(amp_i_in);
            }
    fclose(pf);
    return;
};//end read_amp
