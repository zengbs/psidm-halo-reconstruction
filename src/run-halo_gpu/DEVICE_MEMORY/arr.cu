#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<cuda_runtime.h>
#include"cuapi.h"
#include"macros.h"
#include"main.h"
#include"arr.h"
#include"ylm.h"

__device__ int get_pos(int i, int j, int k, int nx, int ny)
{
    int pos = (i + nx * (j + ny * k));
    return pos;
}

__device__ void cal_dis_x(int i, float dx, float xcen, float *x)
{
    *x = i * dx - xcen;
}

__device__ void cal_dis_y(int j, float dy, float ycen, float *y)
{
    *y = j * dy - ycen;
}

__device__ void cal_dis_z(int k, float dz, float zcen, float *z)
{
    *z = k * dz - zcen;
}

__device__ void cal_dis_x_O(int i, int nx, int octant, float dx, float xcen, float *x)
{
    if((octant%4<=1))
        *x = (i+nx) * dx - xcen;
    else
        *x = i * dx - xcen;
}

__device__ void cal_dis_y_O(int j, int ny, int octant, float dy, float ycen, float *y)
{
    if((octant-1)%4<=1)
        *y = (j+ny) * dy - ycen;
    else
        *y = j * dy - ycen;
}

__device__ void cal_dis_z_O(int k, int nz, int octant, float dz, float zcen, float *z)
{
    if(octant<=4)
        *z = (k+nz) * dz - zcen;
    else
        *z = k * dz - zcen;
}

__device__ void cal_dis_r(float *x, float *y, float *z, float *r)
{
    *r = sqrtf((*x) * (*x) + (*y) * (*y) + (*z) * (*z));
}

void read_array(bool restart, int restart_id, char *filename_r, char *filename_i)
{
    FILE *f_temp[2];
    f_temp[0] = fopen(filename_r, "rb");
    f_temp[1] = fopen(filename_i, "rb");

#ifdef DEBUG
    if (f_temp[0]!=NULL)
        printf("Restart file for real part opened successfully!.\n");
    else
    {
        printf("Fail to open restart file for real part!.\n");
        exit(1);
    }
    if (f_temp[1]!=NULL)
        printf("Restart file for imaginary part opened successfully!.\n");
    else
    {
        printf("Fail to open restart file for imaginary part!.\n");
        exit(1);
    }
#endif

    if (restart)
    {
        int restart_id_check_r, nbox_check_r;
        int restart_id_check_i, nbox_check_i;
        fread(&restart_id_check_r, sizeof(int), 1, f_temp[0]);
        fread(&nbox_check_r, sizeof(int), 1, f_temp[0]);
        fread(&restart_id_check_i, sizeof(int), 1, f_temp[1]);
        fread(&nbox_check_i, sizeof(int), 1, f_temp[1]);

        restart_id_check_r ++;
        restart_id_check_i ++;
        if (restart_id_check_r!=restart_id||restart_id_check_i!=restart_id)
        {
            printf("RESTART_ID check failed while reading array (RESTART_ID=%d ; RESTART_ID_CHECK_R=%d ; RESTART_ID_CHECK_I=%d )! Exit!\n", restart_id, restart_id_check_r, restart_id_check_i);
            exit(1);
        }
        if (nbox_check_r!=nbox||nbox_check_i!=nbox)
        {
            printf("NBOX check failed while reading array (NBOX=%d ; NBOX_CHECK_R=%d ; NBOX_CHECK_I=%d )! Exit!\n", nbox, nbox_check_r, nbox_check_i);
            exit(1);
        }
    }
    fread(array_r_host, sizeof(float), nx*ny*nz, f_temp[0]);
    fread(array_i_host, sizeof(float), nx*ny*nz, f_temp[1]);

    fclose(f_temp[0]);
    fclose(f_temp[1]);
    return;
}
//

void write_array(bool dump, int *dump_id, char *filename_r, char *filename_i)
{
#ifdef OCTANT_DECOMPOSE
    sprintf(filename_r, "%s_octant%d", filename_r, octant);
    sprintf(filename_i, "%s_octant%d", filename_i, octant);
#endif

    if (dump)
    {
        sprintf(filename_r, "%s_DUMP_ID=%d", filename_r, *dump_id);
        sprintf(filename_i, "%s_DUMP_ID=%d", filename_i, *dump_id);
    }

    FILE *f_temp[2];
    f_temp[0] = fopen(filename_r, "wb");
    f_temp[1] = fopen(filename_i, "wb");

#ifdef DEBUG
    if (f_temp[0]!=NULL)
        printf("File for writing real part opened successfully!.\n");
    else
    {
        printf("Fail to open file for writing real part!.\n");
        exit(1);
    }
    if (f_temp[1]!=NULL)
        printf("File for writing imaginary part opened successfully!.\n");
    else
    {
        printf("Fail to open file for writing imaginary part!.\n");
        exit(1);
    }
#endif

    if (dump)
    {
        fwrite(dump_id, sizeof(int), 1, f_temp[0]);
        fwrite(&nbox, sizeof(int), 1, f_temp[0]);
        fwrite(dump_id, sizeof(int), 1, f_temp[1]);
        fwrite(&nbox, sizeof(int), 1, f_temp[1]);
    }

    fwrite(array_r_host, sizeof(float), nx*ny*nz, f_temp[0]);
    fwrite(array_i_host, sizeof(float), nx*ny*nz, f_temp[1]);

    fclose(f_temp[0]);
    fclose(f_temp[1]);
    return;
}

void array_init()
{
	array_r_host = (float*)malloc(N_site*sizeof(float));
	array_i_host = (float*)malloc(N_site*sizeof(float));
        memset(array_r_host, 0, N_site*sizeof(float));
        memset(array_i_host, 0, N_site*sizeof(float));
	return;
}

void array_fin()
{
    free(array_r_host);
    free(array_i_host);
}

void array_add_ylm(int l_max,int l_init,bool firstadd)
{
    bool timing_flag;
    bool cuda_error_flag;
    cudaEvent_t start, stop;
    float total_execute, total_cal_and_expand;
    float *array_r_device, *array_i_device;
    float *array_r_l, *array_i_l;

    cuda_error_flag = CUDA_CHECK_ERROR(cudaEventCreate(&start));
    cuda_error_flag = CUDA_CHECK_ERROR(cudaEventCreate(&stop));
    cuda_error_flag = CUDA_CHECK_ERROR(cudaEventRecord(start));
#ifdef OCTANT_DECOMPOSE
    printf( "Working on Octant [%d/8] -- Lidx [%3d/%3d] ...\n", octant_global, Lidx_global, lnod-1);
#else
    printf( "Working on Lidx [%3d/%3d] ...\n", Lidx_global, lnod-1);
#endif

    timing_flag = true;
    if (timing_flag)
        total_cal_and_expand = 0.;

    cuda_error_flag = CUDA_CHECK_ERROR(cudaMalloc((void **)&array_r_device, N_site*sizeof(float)));
    if (!cuda_error_flag)
    {
        printf("Memory allocating array_r_device failed!\n");
	exit(1);
    }
#ifdef DEBUG
    else
        printf("Memory allocating array_r_device succeed!\n");
#endif
    cuda_error_flag = CUDA_CHECK_ERROR(cudaMalloc((void **)&array_i_device, N_site*sizeof(float)));
    if (!cuda_error_flag)
    {
        printf("Memory allocating array_i_device failed!\n");
	exit(1);
    }
#ifdef DEBUG
    else
        printf("Memory allocating array_i_device succeed!\n");
    printf("CUDA malloc completed...\n");
#endif
    cuda_error_flag = CUDA_CHECK_ERROR(cudaMemset(array_r_device, 0, N_site*sizeof(float)));
    if (!cuda_error_flag)
    {
        printf("Memory setting array_r failed!\n");
	exit(1);
    }
#ifdef DEBUG
    else
        printf("Memory setting array_r succeed!\n");
#endif
    cuda_error_flag = CUDA_CHECK_ERROR(cudaMemset(array_i_device, 0, N_site*sizeof(float)));
    if (!cuda_error_flag)
    {
        printf("Memory setting array_i failed!\n");
	exit(1);
    }
#ifdef DEBUG
    else
        printf("Memory setting array_i succeed!\n");
#endif
//    array_r_l = (float*)malloc(N_site*sizeof(float));
//    array_i_l = (float*)malloc(N_site*sizeof(float));
    cuda_error_flag = CUDA_CHECK_ERROR(cudaMallocHost((void**)&array_r_l, N_site*sizeof(float)));
    cuda_error_flag = CUDA_CHECK_ERROR(cudaMallocHost((void**)&array_i_l, N_site*sizeof(float)));

    do_expand_ylm(array_r_device, array_i_device, &total_cal_and_expand, l_max,l_init, timing_flag);
    
    cuda_error_flag = CUDA_CHECK_ERROR(cudaMemcpy(array_r_l, array_r_device, N_site*sizeof(float), cudaMemcpyDeviceToHost));
    if (!cuda_error_flag)
        printf("Copying array_r failed!\n");
#ifdef DEBUG
    else
        printf("Copying array_r succeed!\n");
#endif
        
    cuda_error_flag = CUDA_CHECK_ERROR(cudaMemcpy(array_i_l, array_i_device, N_site*sizeof(float), cudaMemcpyDeviceToHost));
    if (!cuda_error_flag)
        printf("Copying array_i failed!\n");
#ifdef DEBUG
    else
        printf("Copying array_i succeed!\n");
#endif
    
    cuda_error_flag = CUDA_CHECK_ERROR(cudaFree(array_r_device));
    cuda_error_flag = CUDA_CHECK_ERROR(cudaFree(array_i_device));

    for (int site=0; site<N_site; site++)
    {
        array_r_host[site] += array_r_l[site];
        array_i_host[site] += array_i_l[site];
    }
//    free(array_r_l);
//    free(array_i_l);
    cuda_error_flag = cudaFreeHost(array_r_l);
    cuda_error_flag = cudaFreeHost(array_i_l);

    cuda_error_flag = CUDA_CHECK_ERROR(cudaEventRecord(stop));
    cuda_error_flag = CUDA_CHECK_ERROR(cudaEventSynchronize(stop));
    cuda_error_flag = CUDA_CHECK_ERROR(cudaEventElapsedTime(&total_execute, start, stop));

    printf("Total time for expanding-calculating ylm of l_init=%d to l_max=%d over all k takes %.4e s.\n", l_init, l_max, total_cal_and_expand);
    printf("Total time for adding contribution of l_init=%d to l_max=%d over all k takes %.4e s.\n", l_init, l_max, total_execute/1000.);
    fflush(stdout);

    return;
}

void cal_r_max(void)
{
    double x_max;
    double y_max;
    double z_max;

    x_max = min( xcen, (nx-1)*dx - xcen);
    y_max = min( ycen, (ny-1)*dy - ycen);
    z_max = min( zcen, (nz-1)*dz - zcen);

    r_max = nx*dx + ny*dy + nz*dz;
    r_max = min(r_max, x_max);
    r_max = min(r_max, y_max);
    r_max = min(r_max, z_max);
    r_max *= r_fact;
    return;
}

void cal_r_max_O(void)
{
    dx=dx/2.;
    dy=dy/2.;
    dz=dz/2.;

    double x_max;
    double y_max;
    double z_max;

    x_max = min( xcen, (2*nx-1)*dx - xcen);
    y_max = min( ycen, (2*ny-1)*dy - ycen);
    z_max = min( zcen, (2*nz-1)*dz - zcen);

    r_max = nx*dx + ny*dy + nz*dz;
    r_max = min(r_max, x_max);
    r_max = min(r_max, y_max);
    r_max = min(r_max, z_max);
    r_max *= r_fact;
//	printf("r_max=%e\n",r_max);//debug
    return;
}
