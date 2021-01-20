#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>
#include<cuda_runtime.h>
#include"macros.h"
#include"main.h"
#include"ylm.h"
#include"arr.h"
#include"cuapi.h"

// for nan debugging
#define nan_index_x 272 //0
#define nan_index_y 220 //0
#define nan_index_z 0 //1


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
__global__ void cal_and_expand_ylm_O(int rsiz, int nx, int ny, int nz, int octant, float dx, float dy, float dz, float xcen, float ycen, float zcen, float dr, float r_max, float *arr_r, float *arr_i, int *rfunc_num, float ***rfunc, float ***amplitude_r, float ***amplitude_i, int l_max,int l_init)
#else
__global__ void cal_and_expand_ylm(int rsiz, int nx, int ny, int nz, float dx, float dy, float dz, float xcen, float ycen, float zcen, float dr, float r_max, float *arr_r, float* arr_i, int *rfunc_num, float ***rfunc, float ***amplitude_r , float ***amplitude_i, int l_max, int l_init)
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
//        if (idx_x==nan_index_x&&idx_y==nan_index_y&&idx_z==nan_index_z)
//        if (idx_x==(int)(nx/2)&&idx_y==(int)(ny/2)&&idx_z==(int)(nz/2))
//        {
//            printf("Kernel executing...\n");
//            printf("Calculating distance r completed.\n");
//            printf("x = %.4e , y = %.4e , z= %.4e , r = %.4e , r_max = %.4e .\n", x, y, z, r, r_max);
//              printf("index_x = %d , index_y = %d, index_z = %d ; x_cen = %.6e , y_cen = %.6e z_cen = %.6e ; r = %.8e, r_max = %.8e .\n", idx_x, idx_y, idx_z, xcen, ycen, zcen, r, r_max);
//        }
//#endif
        if (r<r_max)
        {
            int n, l, m, ll;
            int index_r, index_site;
            float frac_r, rfunc_interp;
            float fact_1, fact_2 = 0., oldfact = 0.;  // for ylm computation
            float pmm_new, pmm_old, phr_new, phr_old, phi_new, phi_old;
            float array_pos[2];
            double pml, pml_lm1, temp;
            double ylmr[2];

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
//            if (idx_x==nan_index_x&&idx_y==nan_index_y&&idx_z==nan_index_z)
//                printf("x = %.8e\t y = %.8e\t z = %.8e\n", x, y, z);

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
//                    if (idx_x==nan_index_x&&idx_y==nan_index_y&&idx_z==nan_index_z)
//                        printf("pmm_new = %.4e\n", pmm_new);
                    pmm_old = pmm_new;
                }
                fact_1 += (float)2.;
        
                pmm_new = sqrtf(((float)2.*m+(float)1.)*pmm_new/((float)4.*(float)M_PI));
//                if (idx_x==nan_index_x&&idx_y==nan_index_y&&idx_z==nan_index_z)
//                   printf("pmm_new = %.4e\n", pmm_new);
        
                for(l = m;l <= l_max; l++)
                {
                    if(l > m)
                    {
                        fact_2 = sqrtf(((float)4.*l*l-(float)1.)/(l*l-m*m));
//                        if (idx_x==nan_index_x&&idx_y==nan_index_y&&idx_z==nan_index_z)
//                           printf("fact_2 = %.4e\n", fact_2);
                    }
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
//                        if (idx_x==nan_index_x&&idx_y==nan_index_y&&idx_z==nan_index_z)
//                           printf("pml = %.4e\n", pml);
                        pml_lm1 = temp;
                    }
                    oldfact = fact_2;
                    if(m!=0)
                    {
                        ylmr[0] = pml * phr_new * sqrtf((float)2.); // ylmr[0] = ylm[l][m]
                        ylmr[1] = pml * phi_new * sqrtf((float)2.); // ylmr[1] = ylmr[l][-m+2*l+1]
                        // to avoid nan
//                        if (ylmr[0]!=ylmr[0])
//                            ylmr[0] = (float)(0.);
//                        if (ylmr[1]!=ylmr[1])
//                            ylmr[1] = (float)(0.);
                        //
                        ll = l-l_init;
                        if (ll>=0)
                        {
                            for (n=0; n<rfunc_num[ll]; n++)
                            {
                                rfunc_interp = rfunc[ll][n][index_r-1]*((float)1.-frac_r) + rfunc[ll][n][index_r]*frac_r;
                                array_pos[0] += (amplitude_r[ll][n][m]*(float)(ylmr[0]) + amplitude_r[ll][n][-m+2*l+1]*(float)(ylmr[1]))*rfunc_interp;
                                array_pos[1] += (amplitude_i[ll][n][m]*(float)(ylmr[0]) + amplitude_i[ll][n][-m+2*l+1]*(float)(ylmr[1]))*rfunc_interp;
//#ifdef DEBUG
//                                if (idx_x==nan_index_x && idx_y==nan_index_y && idx_z==nan_index_z)
//                                {
//                                    printf("ylm[%d][%d] = %.6e .\n", l, m, ylmr[0] );
//                                    printf("ylm[%d][%d] = %.6e .\n", l, -m+2*l+1, ylmr[1] );
//                                    printf("amplitude_r[%d][%d][%d] = %.6e .\n", l, n ,m, amplitude_r[ll][n][m]);
//                                    printf("amplitude_i[%d][%d][%d] = %.6e .\n", l, n ,m, amplitude_i[ll][n][m]);
//                                    printf("rfunc_interp = %.6e .\n", rfunc_interp);
//                                }
//#endif
                            }
                        }
                    }
                    else
                    {
                        ylmr[0] = pml * phr_new;
                        // to avoid nan
//                        if (ylmr[0]!=ylmr[0])
//                            ylmr[0] = (float)(0.);
                        //
                        ll = l-l_init;
                        if (ll>=0)
                        {
                            for (n=0; n<rfunc_num[ll]; n++)
                            {
                                rfunc_interp = rfunc[ll][n][index_r-1]*((float)1.-frac_r) + rfunc[ll][n][index_r]*frac_r;
                                array_pos[0] += (amplitude_r[ll][n][0]*(float)(ylmr[0]))*rfunc_interp;
                                array_pos[1] += (amplitude_i[ll][n][0]*(float)(ylmr[0]))*rfunc_interp;
//#ifdef DEBUG
//                                if (idx_x==nan_index_x && idx_y==nan_index_y && idx_z==nan_index_z)
//                                {
//                                    printf("ylm[%d][%d] = %.6e .\n", l, 0, ylmr[0] );
//                                    printf("amplitude_r[%d][%d][%d] = %.6e .\n", l, n ,m, amplitude_r[ll][n][0]);
//                                    printf("amplitude_i[%d][%d][%d] = %.6e .\n", l, n ,m, amplitude_i[ll][n][0]);
//                                    printf("rfunc_interp = %.6e .\n", rfunc_interp);
//                                }
//#endif
                            }
                        }
                    }
                } // end of for loop l
//#ifdef DEBUG
//                if (idx_x==(int)(nx/2) && idx_y==(int)(ny/2) && idx_z==(int)(nz/2))
//                    printf("array_r[%d] = %.4e ; array_i[%d] = %.4e \n", index_site, array_pos[0], index_site, array_pos[1]);
//#endif
            } // end of for loop m
            arr_r[index_site] = array_pos[0];
            arr_i[index_site] = array_pos[1];
//#ifdef DEBUG
//            if (idx_x==nan_index_x&&idx_y==nan_index_y&&idx_z==nan_index_z)
//            {
//                printf("arr_r[%d][%d][%d] = %.6e ; arr_i[%d][%d][%d] = %.6e\n", idx_x, idx_y, idx_z, arr_r[index_site], idx_x, idx_y, idx_z, arr_i[index_site]);
//            }
//#endif
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
    int l,n;
    bool cuda_error_flag;
    float *amplitude_r_n_buff, **amplitude_r_l_buff;
    float *amplitude_i_n_buff, **amplitude_i_l_buff;

    amplitude_host = (cmpx***) malloc((l_max+1-l_init)*sizeof(cmpx**));
    for(l = l_init;l <= l_max;l++)
    {
        amplitude_host[l-l_init] = (cmpx**) malloc((rfunc_num[l-l_init])*sizeof(cmpx*));
        cmpx *amplitude_buf = (cmpx*) malloc((rfunc_num[l-l_init])*(2*l+1)*sizeof(cmpx));
        memset(amplitude_buf, 0, rfunc_num[l-l_init]*(2*l+1)*sizeof(cmpx));

        for(n = 0;n < rfunc_num[l-l_init];n++)
        {
            amplitude_host[l-l_init][n] = amplitude_buf;
            amplitude_buf += (2*l+1);
        }
    }
#ifdef DEBUG
    printf("Memory allocation for amplitude_host completed!\n");
#endif
    
    cuda_error_flag = CUDA_CHECK_ERROR(cudaMallocManaged((void**)&amplitude_r, (l_max+1-l_init)*sizeof(float**)));
    if (!cuda_error_flag)
    {
        printf("Allocation for pointer amplitude_r failed!\n");
        exit(1);
    }
#ifdef DEBUG
    else
        printf("Allocation for pointer amplitude_r succeed!\n");
#endif
    cuda_error_flag = CUDA_CHECK_ERROR(cudaMallocManaged((void**)&amplitude_i, (l_max+1-l_init)*sizeof(float**)));
    if (!cuda_error_flag)
    {
        printf("Allocation for pointer amplitude_i failed!\n");
        exit(1);
    }
#ifdef DEBUG
    else
        printf("Allocation for pointer amplitude_i succeed!\n");
#endif

    for(l = l_init;l <= l_max;l++)
    {
        cuda_error_flag = CUDA_CHECK_ERROR(cudaMallocManaged((void**)&amplitude_r_l_buff, rfunc_num[l-l_init]*sizeof(float*)));
        cuda_error_flag = CUDA_CHECK_ERROR(cudaMallocManaged((void**)&amplitude_i_l_buff, rfunc_num[l-l_init]*sizeof(float*)));

//	printf("rank=%d l=%d rfunc_num[%d]=%d\n",rank,l,l-l_init,rfunc_num[l-l_init]);//debug
        for(n = 0;n < rfunc_num[l-l_init];n++)
        {
            cuda_error_flag = CUDA_CHECK_ERROR(cudaMallocManaged((void**)&amplitude_r_n_buff, (2*l+1)*sizeof(float)));
            cuda_error_flag = CUDA_CHECK_ERROR(cudaMemcpy(&amplitude_r_l_buff[n], &amplitude_r_n_buff, sizeof(float*), cudaMemcpyHostToDevice));
//                amplitude[l-l_init][n] = amplitude_buf;
//                amplitude_buf += (2*l+1);
            if (!cuda_error_flag)
            {
                printf("Memory allocation for amplitude_r[%d][%d] failed!\n", l, n);
                exit(1);
            }
#ifdef DEBUG
            else
                printf("Memory allocation for amplitude_r[%d][%d] succeed!\n", l, n);
#endif
            cuda_error_flag = CUDA_CHECK_ERROR(cudaMallocManaged((void**)&amplitude_i_n_buff, (2*l+1)*sizeof(float)));
            cuda_error_flag = CUDA_CHECK_ERROR(cudaMemcpy(&amplitude_i_l_buff[n], &amplitude_i_n_buff, sizeof(float*), cudaMemcpyHostToDevice));
            if (!cuda_error_flag)
            {
                printf("Memory allocation for amplitude_i[%d][%d] failed!\n", l, n);
                exit(1);
            }
#ifdef DEBUG
            else
                printf("Memory allocation for amplitude_i[%d][%d] succeed!\n", l, n);
#endif
        }
        cuda_error_flag = CUDA_CHECK_ERROR(cudaMemcpy(&amplitude_r[l-l_init], &amplitude_r_l_buff, sizeof(float**), cudaMemcpyHostToDevice));
        if (!cuda_error_flag)
        {
            printf("Memory allocation for amplitude_r[%d] failed!\n", l);
            exit(1);
        }
#ifdef DEBUG
        else
            printf("Memory allocation for amplitude_r[%d] succeed!\n", l);
#endif
        cuda_error_flag = CUDA_CHECK_ERROR(cudaMemcpy(&amplitude_i[l-l_init], &amplitude_i_l_buff, sizeof(float**), cudaMemcpyHostToDevice));
        if (!cuda_error_flag)
        {
            printf("Memory allocation for amplitude_i[%d] failed!\n", l);
            exit(1);
        }
#ifdef DEBUG
        else
            printf("Memory allocation for amplitude_i[%d] succeed!\n", l);
#endif
    }
    return;
}

void ylm_fin(int l_init,int l_max)
{
    int l;
    bool cuda_error_flag;
    for(l = l_init;l <= l_max;l++)
    {
	int ll=l-l_init;
        free(amplitude_host[ll][0]);
        cuda_error_flag = CUDA_CHECK_ERROR(cudaFree(amplitude_r[ll][0]));
        cuda_error_flag = CUDA_CHECK_ERROR(cudaFree(amplitude_i[ll][0]));

        free(amplitude_host[ll]);
        cuda_error_flag = CUDA_CHECK_ERROR(cudaFree(amplitude_r[ll]));
        cuda_error_flag = CUDA_CHECK_ERROR(cudaFree(amplitude_i[ll]));
    }

    free(amplitude_host);
    cuda_error_flag = CUDA_CHECK_ERROR(cudaFree(amplitude_r));
    cuda_error_flag = CUDA_CHECK_ERROR(cudaFree(amplitude_i));
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
    bool cuda_error_flag;
    
    for(int l=l_init;l<=l_max;l++)
    {
        for (int n=0; n<rfunc_num[l-l_init]; n++) 
        {
            CUDA_CHECK_ERROR(cudaFree(rfunc_l_pointer[l-l_init][n]));
        }
        cuda_error_flag = CUDA_CHECK_ERROR(cudaFree(rv_pointer[l-l_init]));
        cuda_error_flag = CUDA_CHECK_ERROR(cudaFree(rfunc_pointer[l-l_init]));
        free(rfunc_l_pointer[l-l_init]);
    }
    cuda_error_flag = CUDA_CHECK_ERROR(cudaFree(rv));
    cuda_error_flag = CUDA_CHECK_ERROR(cudaFree(rfunc_num));
    cuda_error_flag = CUDA_CHECK_ERROR(cudaFree(rfunc));
    free(rfunc_pointer);
    free(rfunc_l_pointer);
    free(rv_pointer);
    printf("Eigen values and functions from l_init=%d to l_max=%d free.\n", l_init, l_max);//debug
}

void load_eigenstates(int *l_init,int *l_max, int *eigen_num, char *filename,int lnodidx)
{
    int l, n;
    FILE *pf;
    char str[80];
    float *rv_local_f, *rfunc_local_f;
    double *rv_local, *rfunc_local;
    bool cuda_error_flag;
    sprintf(str,"%s%d.bin",filename,lnodidx);
    pf = fopen(str, "rb");
    fread(l_init, 1, sizeof(int), pf);
    fread(l_max, 1, sizeof(int), pf);

    if (*l_init>*l_max)
    {
        printf("l_init > l_max for Lidx=%d ! Pass!\n", Lidx_global);
        return;
    }
    
    *eigen_num = 0;
    rv_pointer = (float**)malloc((*l_max+1-*l_init)*sizeof(float*));
    rfunc_l_pointer = (float***)malloc((*l_max+1-*l_init)*sizeof(float**));
    rfunc_pointer = (float***)malloc((*l_max+1-*l_init)*sizeof(float**));

    cuda_error_flag = CUDA_CHECK_ERROR(cudaMallocManaged((void **)&rfunc_num, (*l_max+1-*l_init)*sizeof(int)));//rfunc_num[l] is the number of eigenvalues
    if (!cuda_error_flag)
    {
        printf("Memory allocation for eigen values numbers failed!\n");
        exit(1);
    }
#ifdef DEBUG
    else
        printf("Memory allocation for eigen value number succeed!\n");
#endif
    cuda_error_flag = CUDA_CHECK_ERROR(cudaMalloc((void **)&rv, (*l_max+1-*l_init)*sizeof(float*)));//rv[l][n] is the eigen_ln
    if (!cuda_error_flag)
    {
        printf("Memory allocation for eigen values failed!\n");
        exit(1);
    }
#ifdef DEBUG
    else
        printf("Memory allocation for eigen values succeed!\n");
#endif
    cuda_error_flag = CUDA_CHECK_ERROR(cudaMalloc((void **)&rfunc, (*l_max+1-*l_init)*sizeof(float**)));//rc[l][n][r] is the eigenfunction_ln
    if (!cuda_error_flag)
    {
        printf("Memory allocation for eigen functions failed!\n");
        exit(1);
    }
#ifdef DEBUG
    else
        printf("Memory allocation for eigen functions succeed!\n");
#endif

    fread(rfunc_num, *l_max+1-*l_init, sizeof(int), pf);

    for(l = *l_init;l <= *l_max;l++)
    {
        int ll = l-*l_init;
        *eigen_num += rfunc_num[ll];

        rv_local = (double*)malloc(rfunc_num[ll]*sizeof(double));
        rv_local_f = (float*)malloc(rfunc_num[ll]*sizeof(float));
        cuda_error_flag = CUDA_CHECK_ERROR(cudaMalloc((void **)&rv_pointer[ll], rfunc_num[ll]*sizeof(float)));
        if (!cuda_error_flag)
        {
            printf("Memory allocation for eigen values with l = %d failed!\n", l);
            exit(1);
        }
#ifdef DEBUG
        else
            printf("Memory allocation for eigen values with l = %d succeed!\n", l);
#endif
	fread(rv_local, rfunc_num[ll], sizeof(double), pf);

//        rfunc_pointer[ll] = (float**)malloc(rfunc_num[ll]*sizeof(float*));
        cuda_error_flag = CUDA_CHECK_ERROR(cudaMalloc((void**)&rfunc_pointer[ll], rfunc_num[ll]*sizeof(float*)));
        rfunc_l_pointer[ll] = (float**)malloc(rfunc_num[ll]*sizeof(float*));

	for(n = 0;n < rfunc_num[ll];n++)
	{
            rv_local_f[n] = (float)(rv_local[n]);
            rfunc_local = (double*)malloc(rsiz*sizeof(double));
            rfunc_local_f = (float*)malloc(rsiz*sizeof(float));

            cuda_error_flag = CUDA_CHECK_ERROR(cudaMalloc((void **)&rfunc_l_pointer[ll][n], rsiz*sizeof(float)));

	    fread(rfunc_local, rsiz, sizeof(double), pf);
            for (int r=0; r<rsiz; r++)
                rfunc_local_f[r] = (float)(rfunc_local[r]);

            cuda_error_flag = CUDA_CHECK_ERROR(cudaMemcpy(rfunc_l_pointer[ll][n], rfunc_local_f, rsiz*sizeof(float), cudaMemcpyHostToDevice));
            if (!cuda_error_flag)
            {
                printf("Memory copy for eigen functions to rfunc_pointer with l = %d and n = %d failed!\n", l, n);
                exit(1);
            }
#ifdef DEBUG
            else
                printf("Memory copy for eigen functions with l = %d and n = %d succeed!\n", l, n);
#endif
            free(rfunc_local_f);
            free(rfunc_local);
	}

        cuda_error_flag = CUDA_CHECK_ERROR(cudaMemcpy(rfunc_pointer[ll], rfunc_l_pointer[ll], rfunc_num[ll]*sizeof(float*), cudaMemcpyHostToDevice));
        if (!cuda_error_flag)
        {
            printf("Memory copy for eigen functions to rfunc_pointer with l = %d failed!\n", l);
            exit(1);
        }
#ifdef DEBUG
        else
            printf("Memory copy for eigen functions to rfunc_pointer with l = %d succeed!\n", l);
#endif

        cuda_error_flag = CUDA_CHECK_ERROR(cudaMemcpy(rv_pointer[ll], rv_local_f, rfunc_num[ll]*sizeof(float), cudaMemcpyHostToDevice));
        if (!cuda_error_flag)
        {
            printf("Memory copy for eigen values to rv_pointer with l = %d failed!\n", l);
            exit(1);
        }
#ifdef DEBUG
        else
            printf("Memory copy for eigen values to rv_pointer with l = %d succeed!\n", l);
#endif
        
        free(rv_local_f);
        free(rv_local);
//        *ln_total_count += rfunc_num[ll];
    }
    cuda_error_flag = CUDA_CHECK_ERROR(cudaMemcpy(rv, rv_pointer, (*l_max+1-*l_init)*sizeof(float*), cudaMemcpyHostToDevice));
    if (!cuda_error_flag)
    {
        printf("Memory copy for eigen values failed!\n");
        exit(1);
    }
#ifdef DEBUG
    else
        printf("Memory copy for eigen functions succeed!\n", l);
#endif
    cuda_error_flag = CUDA_CHECK_ERROR(cudaMemcpy(rfunc, rfunc_pointer, (*l_max+1-*l_init)*sizeof(float**), cudaMemcpyHostToDevice));
    if (!cuda_error_flag)
    {
        printf("Memory copy for eigen functions failed!\n");
        exit(1);
    }
#ifdef DEBUG
    else
        printf("Memory copy for eigen functions succeed!\n", l);
#endif
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
        void *addr_of_buf = amplitude_host[ll][0];
        int   size_of_buf = rfunc_num[ll] * (2 * l + 1) * 2;
        fwrite(addr_of_buf, sizeof(double), size_of_buf, pf);
    }
    fclose(pf);

    return;
}

void load_ylm(int l_init, int l_max, char *filename, int lnodidx)
{
    int l, n, m;
    FILE *pf;
    char filen[80];

    sprintf(filen,"%s%s%d%s",filename,"L",lnodidx,".bin");
    pf = fopen(filen, "rb");
    
    for(l = l_init;l <= l_max;l++)
    {
        void *addr_of_buf = amplitude_host[l-l_init][0];
        int   size_of_buf = rfunc_num[l-l_init] * (2 * l + 1) * 2;
        fread(addr_of_buf, sizeof(double), size_of_buf, pf);
#ifdef DEBUG
        printf("Amplitude of L=%d have been loaded to amplitude_host .\n", l);
#endif
        for (n=0; n<rfunc_num[l-l_init]; n++)
        {
            for (m=0; m<2*l+1; m++)
            {
//                printf("%.4e\t%.4e\n", amplitude_host[l-l_init][n][m][0], amplitude_host[l-l_init][n][m][1]);
                amplitude_r[l-l_init][n][m] = (float)(amplitude_host[l-l_init][n][m][0]);
#ifdef DEBUG
                printf("Amplitude of L=%d, n=%d, m=%d have been copied to amplitude_r.\n", l, n, m);
#endif
                amplitude_i[l-l_init][n][m] = (float)(amplitude_host[l-l_init][n][m][1]);
#ifdef DEBUG
                printf("Amplitude of L=%d, n=%d, m=%d have been copied to amplitude_i.\n", l, n, m);
#endif
            }
#ifdef DEBUG
            printf("Amplitude of L=%d, n= %d have been copied to amplitude_r and amplitude_i.\n", l, n);
#endif
        }
#ifdef DEBUG
        printf("Amplitude of L=%d have been copied to amplitude_r and amplitude_i.\n", l);
#endif
    }
    fclose(pf);
//

//#ifdef DEBUG
//    int test_l = 110 , test_n = 10, test_m = 88;
//    int ll = test_l-l_init;
//    printf("Amplitude Test:\n");
//    if (test_m!=0)
//    {
////        printf("Amp[%d][%d][%d] = %.4e%+.4e ; Amp[%d][%d][%d] = %.4e%+.4e .\n", test_l, test_n, test_m, amplitude_host[ll][test_n][test_m][0], amplitude_hostl[ll][test_n][test_m][1], test_l, test_n, 2*(test_l)+1-test_m, amplitude_host[ll][test_n][2*test_l+1-test_m][0], amplitude_host[ll][test_n][2*test_l+1-test_m][1]);
//        printf("Amp[%d][%d][%d] = %.4e%+.4e ; Amp[%d][%d][%d] = %.4e%+.4e .\n", test_l, test_n, test_m, amplitude_r[ll][test_n][test_m], amplitude_i[ll][test_n][test_m], test_l, test_n, 2*(test_l)+1-test_m, amplitude_r[ll][test_n][2*test_l+1-test_m], amplitude_i[ll][test_n][2*test_l+1-test_m]);
//    }
//    else
//    {
////        printf("Amp[%d][%d][%d] = %.4e%+.4e .\n", test_l, test_n, test_m, amplitude_host[ll][test_n][test_m][0], amplitude_host[ll][test_n][test_m][1]);
//        printf("Amp[%d][%d][%d] = %.4e%+.4e .\n", test_l, test_n, test_m, amplitude_r[ll][test_n][test_m], amplitude_i[ll][test_n][test_m]);
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
    bool cuda_error_flag;
    float time_temp;
    dim3 tpB(tpB_x, tpB_y, tpB_z);
    dim3 bpG(bpG_x, bpG_y, bpG_z);

    if (timing_flag)
    {
        cuda_error_flag = CUDA_CHECK_ERROR(cudaEventCreate(&start));
        cuda_error_flag = CUDA_CHECK_ERROR(cudaEventCreate(&stop));
        cuda_error_flag = CUDA_CHECK_ERROR(cudaEventRecord(start,0));
    }

#ifdef OCTANT_DECOMPOSE
    cal_and_expand_ylm_O<<<bpG,tpB>>>(rsiz, nx, ny, nz, octant, (float)dx, (float)dy, (float)dz, (float)xcen, (float)ycen, (float)zcen, (float)dr, (float)r_max, arr_r, arr_i, rfunc_num, rfunc, amplitude_r, amplitude_i, l_max, l_init);
#else
    cal_and_expand_ylm<<<bpG,tpB>>>(rsiz, nx, ny, nz, (float)dx, (float)dy, (float)dz, (float)xcen, (float)ycen, (float)zcen, (float)dr, (float)r_max, arr_r, arr_i, rfunc_num, rfunc, amplitude_r, amplitude_i, l_max, l_init);
#endif

    cuda_error_flag = CUDA_CHECK_ERROR(cudaPeekAtLastError());
    if (!cuda_error_flag)
    {
        int Lidx_temp = Lidx_global-1;
        printf("Dump with Lidx = %d .\n", Lidx_temp);
        write_array(true,&Lidx_temp,file_r,file_i);
        strcpy(file_r, buff_r);
        strcpy(file_i, buff_i);
//        exit(1);
    }
#ifdef DEBUG
    else
        printf("Calculating-expanding ylm succeed!\n");
#endif
    cuda_error_flag = CUDA_CHECK_ERROR(cudaDeviceSynchronize());

    if (timing_flag)
    {
        cuda_error_flag = CUDA_CHECK_ERROR(cudaEventRecord(stop,0));
        cuda_error_flag = CUDA_CHECK_ERROR(cudaEventSynchronize(stop));
        cuda_error_flag = CUDA_CHECK_ERROR(cudaEventElapsedTime(&time_temp, start, stop));
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

                amplitude_host [l][n][m][0] = (float)(amp_r_in);
                amplitude_host [l][n][m][1] = (float)(amp_i_in);
            }
    fclose(pf);
    return;
};//end read_amp
