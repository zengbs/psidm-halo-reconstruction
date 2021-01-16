/* move extern int Lidx_global and extern int octant_global to arr.h  2020.12.29*/
/* fix typo in cal_dis_O() 2020.12.30 */
/* add timing in array_add_ylm() 2020.12.31 */
/* add r_max debug in array_add_ylm() 2020.12.31 */
/* add auto renaming output file in write_array() 2021.01.03 */
/* add iter_test for testing the code serval times 2021.01.07*/
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<cuda_runtime.h>
#include"cuapi.h"
#include"macros.h"
#include"text_io.h"
#include"main.h"
#include"arr.h"
#include"ylm.h"
#define RESTART_FILENAME_R "RESTART_R"
#define RESTART_FILENAME_I "RESTART_I"

void run_single_lidx(void);

void run(void)
{
    run_single_lidx();
}

void run_single_lidx(void)
{
    float time_temp;
    char buff_r[128], buff_i[128];
    
//    solve_init();
//    gen_r(r_1,r);
	
    int l_max=0;
    int l_init=0;
    
// add data dump and restart 2021.01.04
    cudaEvent_t eigen_copy_start, eigen_copy_stop, amplitude_copy_start, amplitude_copy_stop, lidx_start, lidx_stop;
    cudaSetDevice(0);

    cudaEventCreate(&eigen_copy_start);
    cudaEventCreate(&eigen_copy_stop);
    cudaEventCreate(&amplitude_copy_start);
    cudaEventCreate(&amplitude_copy_stop);
    cudaEventCreate(&lidx_start);
    cudaEventCreate(&lidx_stop);

    cudaEventRecord(lidx_start,0);
    cudaEventRecord(eigen_copy_start,0);
    
    /*** load eigenstates and broadcast to every nodes***/
    load_eigenstates(&l_init,&l_max,"egn",Lidx_global);
#ifdef DEBUG
    printf("Eigen states loading completed!\n");
#endif
    cudaEventRecord(eigen_copy_stop,0);
    cudaEventSynchronize(eigen_copy_stop);
    cudaEventElapsedTime(&time_temp, eigen_copy_start, eigen_copy_stop);
    printf("Total time for copying eigen states from l_init=%d to l_max=%d is %.4e s.\n", l_init, l_max, time_temp/1000.);
//  ignoring all operations if l_max<l_init 2020.12.29
    if (l_max<l_init)
    {
        printf("l_init = %d @ Lidx = %d will be ignored since l_max < l_init !\n", l_init, Lidx_global);
        return;
    }
#ifdef DEBUG
    printf("Start kernel preparation...\n");
#endif
    /*** generate upper bound ***/
//        gen_ubound(rv[0][0] * 0., l_max,l_init);
    ylm_init(l_init,l_max);
#ifdef DEBUG
    printf("Memory allocation for spherical harmonics completed.\n");
#endif

    cudaEventRecord(amplitude_copy_start,0);
    /*** read amplitudes and Bcast ***/
    load_ylm(l_init,l_max,"amp_only",Lidx_global);
    /*** generate density in simulation box ***/
#ifdef DEBUG
    printf("Amplitudes of spherical harmonics loading completed.\n");
#endif
    cudaEventRecord(amplitude_copy_stop,0);
    cudaEventSynchronize(amplitude_copy_stop);
    cudaEventElapsedTime(&time_temp, amplitude_copy_start, amplitude_copy_stop);
    printf("Total time for copying  amplitudes from l_init=%d to l_max=%d is %.4e s.\n", l_init, l_max, time_temp/1000.);

    array_add_ylm(l_max,l_init,true);

    cudaDeviceSynchronize();
    ylm_fin(l_init,l_max);
    free_egn(l_max,l_init);

    cudaEventRecord(lidx_stop,0);
    cudaEventSynchronize(lidx_stop);
    cudaEventElapsedTime(&time_temp, lidx_start, lidx_stop);
    printf("Total time for running through l_init=%d to l_max=%d takes %.4e s.\n",l_init, l_max, time_temp/1000.);

    cudaDeviceReset();

    return;
}
