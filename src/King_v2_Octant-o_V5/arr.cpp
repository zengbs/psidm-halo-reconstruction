/* move extern int Lidx_global and extern int octant_global to arr.h  2020.12.29*/
/* fix typo in cal_dis_O() 2020.12.30 */
/* add timing in array_add_ylm() 2020.12.31 */
/* add r_max debug in array_add_ylm() 2020.12.31 */
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<mpi.h>
#include"macros.h"
#include"main.h"
#include"arr.h"
#include"ylm.h"

int get_pos(int i, int j, int k)
{
    int pos = (i + nx * (j + ny * k));
    return pos;
}

void read_array(char *filename_r, char *filename_i)
{
    int i,j,k;

    MPI_File fh[2];
    MPI_Offset off;
    MPI_Status stat;

    float *(array_buf[2]);
    array_buf[0] = (float*) malloc(nx * ny * sizeof(float));
    array_buf[1] = (float*) malloc(nx * ny * sizeof(float));

    array = (cmpxf*) malloc(nx*ny*(nz/size+1)*sizeof(cmpxf));

    MPI_File_open(MPI_COMM_WORLD, filename_r, MPI_MODE_RDONLY, MPI_INFO_NULL, &(fh[0]));
    MPI_File_open(MPI_COMM_WORLD, filename_i, MPI_MODE_RDONLY, MPI_INFO_NULL, &(fh[1]));
    MPI_File_set_view((fh[0]), 0, MPI_FLOAT, MPI_FLOAT, "native", MPI_INFO_NULL);
    MPI_File_set_view((fh[1]), 0, MPI_FLOAT, MPI_FLOAT, "native", MPI_INFO_NULL);

    for(k = 0;k < nz;k++)
    {
        int kk = k/size;
        off = nx*ny*k;
        if(k % size == rank)
        {
            MPI_File_read_at(fh[0], off, array_buf[0], nx*ny, MPI_FLOAT, &stat);
            MPI_File_read_at(fh[1], off, array_buf[1], nx*ny, MPI_FLOAT, &stat);
            for(j = 0;j < ny;j++)
            {
                for(i = 0;i < nx;i++)
                 {
                    int pos     = i + nx * (j + ny * kk);
                    int pos_buf = i + nx * j;
                    array[pos][0] = array_buf[0][pos_buf];
                    array[pos][1] = array_buf[1][pos_buf];
                }
            }
        }
    }

    MPI_File_close(&fh[0]);
    MPI_File_close(&fh[1]);

    free (array_buf[0]);
    free (array_buf[1]);

    return;
}

void write_array(char *filename_r, char *filename_i)
{
    int i,j,k;

    MPI_File fh[2];
    MPI_Offset off;
    MPI_Status stat;

    float *(array_buf[2]);
    array_buf[0] = (float*) malloc(nx * ny * sizeof(float));
    array_buf[1] = (float*) malloc(nx * ny * sizeof(float));

    MPI_File_open(MPI_COMM_WORLD, filename_r, MPI_MODE_RDWR|MPI_MODE_CREATE, MPI_INFO_NULL, &(fh[0]));
    MPI_File_open(MPI_COMM_WORLD, filename_i, MPI_MODE_RDWR|MPI_MODE_CREATE, MPI_INFO_NULL, &(fh[1]));
    MPI_File_set_view((fh[0]), 0, MPI_FLOAT, MPI_FLOAT, "native", MPI_INFO_NULL);
    MPI_File_set_view((fh[1]), 0, MPI_FLOAT, MPI_FLOAT, "native", MPI_INFO_NULL);




    for(k = 0;k < nz;k++)
    {
        int kk = k/size;//kk:0~13
        off = nx*ny*k;
    	//printf("inside loop size = %d\n", size );
        if(k % size == rank)
        {
            for(j = 0;j < ny;j++)
            {
                for(i = 0;i < nx;i++)
                {
                    int pos     = i + nx * (j + ny * kk);
                    int pos_buf = i + nx * j;
//			printf("inside loop pos_buf = %d,rank=%i\n", pos_buf ,rank);
//			printf("inside loop array[pos][0] = %f,rank=%i\n", array[pos][0] ,rank);
                    array_buf[0][pos_buf] = array[pos][0];
                    array_buf[1][pos_buf] = array[pos][1];
//			if(pos%100000==0) printf("array[%i][0]=%f\n",pos,array[pos][0]);//debug
                }
            }
            MPI_File_write_at(fh[0], off, array_buf[0], nx*ny, MPI_FLOAT, &stat);
            MPI_File_write_at(fh[1], off, array_buf[1], nx*ny, MPI_FLOAT, &stat);
        }
    }

//    printf("finish loop MPI_Rank = %d\n", rank );
    MPI_File_close(&fh[0]);
    MPI_File_close(&fh[1]);

    free (array_buf[0]);
    free (array_buf[1]);

    return;
}

void write_sliceXY(char *filename_r, char *filename_i)
{
    int i,j,k;
    int neig_range = 64;
    int ixcen = (int)rint(xcen / dx);
    int iycen = (int)rint(ycen / dy);
    int izcen = (int)rint(zcen / dz);

    MPI_File fh[2];
    MPI_Offset off;
    MPI_Status stat;

    float *(array_buf[2]);
    array_buf[0] = (float*) malloc(2*neig_range*2*neig_range* sizeof(float));
    array_buf[1] = (float*) malloc(2*neig_range*2*neig_range* sizeof(float));

    MPI_File_open(MPI_COMM_WORLD, filename_r, MPI_MODE_RDWR|MPI_MODE_CREATE, MPI_INFO_NULL, &(fh[0]));
    MPI_File_open(MPI_COMM_WORLD, filename_i, MPI_MODE_RDWR|MPI_MODE_CREATE, MPI_INFO_NULL, &(fh[1]));
    MPI_File_set_view((fh[0]), 0, MPI_FLOAT, MPI_FLOAT, "native", MPI_INFO_NULL);
    MPI_File_set_view((fh[1]), 0, MPI_FLOAT, MPI_FLOAT, "native", MPI_INFO_NULL);

    for(k = izcen;k < izcen+1;k++)
    {
        int kk = k/size;
        off = 0;
        if(k % size == rank)
        {
            for(j = iycen-neig_range;j < iycen+neig_range;j++)
            {
                for(i = ixcen-neig_range;i < ixcen+neig_range;i++)
                {
                    int pos     = i + nx * (j + ny * kk);
                    int pos_buf = (i-ixcen+neig_range) + (neig_range*2) * (j-iycen+neig_range);
                    array_buf[0][pos_buf] = array[pos][0];
                    array_buf[1][pos_buf] = array[pos][1];
//		if(pos%30==0)printf("array[i][j][k][0]=%f\n",array[pos][0]);//debug
                }
            }
            MPI_File_write_at(fh[0], off, array_buf[0], (neig_range*2)*(neig_range*2), MPI_FLOAT, &stat);
            MPI_File_write_at(fh[1], off, array_buf[1], (neig_range*2)*(neig_range*2), MPI_FLOAT, &stat);
//		printf("write files\n");//debug
        }
    }

    MPI_File_close(&fh[0]);
    MPI_File_close(&fh[1]);

    free (array_buf[0]);
    free (array_buf[1]);

    return;
}

void write_sliceYZ(char *filename_r, char *filename_i)//MPI_IO
{
    int i,j,k;
    int neig_range = 64;
    int ixcen = (int)rint(xcen / dx);
    int iycen = (int)rint(ycen / dy);
    int izcen = (int)rint(zcen / dz);

    MPI_File fh[2];
    MPI_Offset off;
    MPI_Status stat;

    float *(array_buf[2]);
    array_buf[0] = (float*) malloc(2*neig_range * sizeof(float));
    array_buf[1] = (float*) malloc(2*neig_range * sizeof(float));

    MPI_File_open(MPI_COMM_WORLD, filename_r, MPI_MODE_RDWR|MPI_MODE_CREATE, MPI_INFO_NULL, &(fh[0]));
    MPI_File_open(MPI_COMM_WORLD, filename_i, MPI_MODE_RDWR|MPI_MODE_CREATE, MPI_INFO_NULL, &(fh[1]));
    MPI_File_set_view((fh[0]), 0, MPI_FLOAT, MPI_FLOAT, "native", MPI_INFO_NULL);
    MPI_File_set_view((fh[1]), 0, MPI_FLOAT, MPI_FLOAT, "native", MPI_INFO_NULL);

    for(k = izcen-neig_range;k < izcen+neig_range;k++)
    {
        int kk = k/size;
        off = (k-izcen+neig_range)*(2*neig_range);
        if(k % size == rank)
        {
//	printf("rank=%d kk=%d\n",rank,kk);//debug
            for(j = iycen-neig_range;j < iycen+neig_range;j++)
            {
                for(i = ixcen;i < ixcen+1;i++)
                {
                    int pos     = i + nx * (j + ny * kk);
                    int pos_buf = (j-iycen+neig_range);
                    array_buf[0][pos_buf] = array[pos][0];
                    array_buf[1][pos_buf] = array[pos][1];
//		if(pos%30==0)printf("array[%d][%d][%d][0]=%f\n",i,j,k,array[pos][0]);//debug
//			printf("off=%d izcen=%d,k=%d array_buf[0][%d]=%f\n",off,izcen,k,pos_buf,array_buf[0][pos_buf]);//debug
                }
            }
//		int result;
            MPI_File_write_at(fh[0], off, array_buf[0], (2*neig_range), MPI_FLOAT, &stat);
            MPI_File_write_at(fh[1], off, array_buf[1], (2*neig_range), MPI_FLOAT, &stat);
//		if(result!=MPI_SUCCESS) sample_error(result, "MPI_File_write_at");
        }
    }

    MPI_File_close(&fh[0]);
    MPI_File_close(&fh[1]);

    free (array_buf[0]);
    free (array_buf[1]);

    return;
}


void find_cen(void)
{
    int i,j,k;
    double den_max = 0;
    int x_max = 0;
    int y_max = 0;
    int z_max = rank;

    for(k = 0;k < nz;k++)
    {
        int kk = k/size;
        if(k % size == rank)
        {
            for(j = 0;j < ny;j++)
                for(i = 0;i < nx;i++)
                {
//                    if(i < nx/2 - nx/8 || i >= nx/2 + nx/8)
//                        continue;
//                    if(j < ny/2 - ny/8 || j >= ny/2 + ny/8)
//                        continue;
//                    if(k < nz/2 - nz/8 || k >= nz/2 + nz/8)
//                        continue;
                    int pos = i + nx*(j + ny*kk);
                    double arr_r = array[pos][0];
                    double arr_i = array[pos][1];
                    double den = arr_r * arr_r + arr_i * arr_i;
                    if(den_max < den)
                    {
                        den_max = den;
                        x_max = i;
                        y_max = j;
                        z_max = k;
                    }
            }
        }
    }

    struct {
        double val;
        int rank;
    } den_comm, den_comm_out;

    den_comm.val = den_max;
    den_comm.rank = rank;

    MPI_Allreduce(&den_comm, &den_comm_out, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
    int rank_max = den_comm_out.rank;

    MPI_Bcast(&x_max, 1, MPI_INT, rank_max, MPI_COMM_WORLD);
    MPI_Bcast(&y_max, 1, MPI_INT, rank_max, MPI_COMM_WORLD);
    MPI_Bcast(&z_max, 1, MPI_INT, rank_max, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    rank_max       = (z_max     + size) % size;
    int rank_max_m = (z_max - 1 + size) % size;
    int rank_max_p = (z_max + 1 + size) % size;
    double den_centre_block[3][3][3];

    for(k = z_max - 1;k <= z_max + 1;k++)
    {
        int kk = k/size;
        if(k % size == rank)
        {
            for(j = y_max - 1;j <= y_max + 1;j++)
            {
                for(i = x_max - 1;i <= x_max + 1;i++)
                {
                    int pos = i + nx*(j + ny*kk);
                    double arr_r = array[pos][0];
                    double arr_i = array[pos][1];
                    double den = arr_r * arr_r + arr_i * arr_i;
                    den_centre_block[k - (z_max - 1)][j - (y_max - 1)][i - (x_max - 1)] = den;
                }
            }
        }
    }
    MPI_Bcast(den_centre_block[0], 9, MPI_DOUBLE, rank_max_m, MPI_COMM_WORLD);
    MPI_Bcast(den_centre_block[1], 9, MPI_DOUBLE, rank_max,   MPI_COMM_WORLD);
    MPI_Bcast(den_centre_block[2], 9, MPI_DOUBLE, rank_max_p, MPI_COMM_WORLD);

    if(rank == rank_max)
    {
        double den;
        double den_m;
        double den_p;
        den_m = den_centre_block[1][1][0];
        den   = den_centre_block[1][1][1];
        den_p = den_centre_block[1][1][2];
        xcen = x_max + (den_m - den_p)/2/(den_m + den_p - 2*den);
        den_m = den_centre_block[1][0][1];
        den   = den_centre_block[1][1][1];
        den_p = den_centre_block[1][2][1];
        ycen = y_max + (den_m - den_p)/2/(den_m + den_p - 2*den);
        den_m = den_centre_block[0][1][1];
        den   = den_centre_block[1][1][1];
        den_p = den_centre_block[2][1][1];
        zcen = z_max + (den_m - den_p)/2/(den_m + den_p - 2*den);

        xcen *= dx;
        ycen *= dy;
        zcen *= dz;

        FILE *pf = fopen("cen.txt", "w");
        fprintf(pf, " centre position : ");
        fprintf(pf, "( %15.10le", xcen);
        fprintf(pf, ", %15.10le", ycen);
        fprintf(pf, ", %15.10le", zcen);
        fprintf(pf, ")\n");
        fclose(pf);
    }
    MPI_Bcast(&xcen, 1, MPI_DOUBLE, rank_max, MPI_COMM_WORLD);
    MPI_Bcast(&ycen, 1, MPI_DOUBLE, rank_max, MPI_COMM_WORLD);
    MPI_Bcast(&zcen, 1, MPI_DOUBLE, rank_max, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    return;
}

void cal_dis(int i, int j, int k, double *x, double *y, double *z, double *r)
{
    *x = i * dx - xcen;
    *y = j * dy - ycen;
    *z = k * dz - zcen;

    *r = (*x) * (*x) + (*y) * (*y) + (*z) * (*z);
    *r = sqrt(*r);
    return;
}

void cal_dis_O(int i, int j, int k, double *x, double *y, double *z, double *r, int octant)//distance accroding to octant
{
//  fixed typo 2020.12.30
    // here dx, dy, dz have been divided by 2 (in cal_r_max_O(), which is called in run_halo.cpp)
    if(octant==1)
    {
	    *x = (i+nx) * dx - xcen;
	    *y = (j+ny) * dy - ycen;
	    *z = (k+nz) * dz - zcen;
    }
    else if(octant==2)
    {
	    *x = i * dx - xcen;
	    *y = (j+ny) * dy - ycen;
	    *z = (k+nz) * dz - zcen;
    }
    else if(octant==3)
    {
	    *x = i * dx - xcen;
	    *y = j * dy - ycen;
	    *z = (k+nz) * dz - zcen;
    }
    else if(octant==4)
    {
//	    *x = (i+nz) * dx - xcen;
	    *x = (i+nx) * dx - xcen;
	    *y = j * dy - ycen;
	    *z = (k+nz) * dz - zcen;
    }
    else if(octant==5)
    {
//	    *x = (i+nz) * dx - xcen;
	    *x = (i+nx) * dx - xcen;
	    *y = (j+ny) * dy - ycen;
	    *z = k * dz - zcen;
    }
    else if(octant==6)
    {
	    *x = (i) * dx - xcen;
	    *y = (j+ny) * dy - ycen;
	    *z = k * dz - zcen;
    }
    else if(octant==7)
    {
	    *x = (i) * dx - xcen;
	    *y = (j) * dy - ycen;
	    *z = k * dz - zcen;
    }
    else if(octant==8)
    {
//	    *x = (i+nz) * dx - xcen;
	    *x = (i+nx) * dx - xcen;
	    *y = j * dy - ycen;
	    *z = k * dz - zcen;
    }
//	if(i==0&&j==0&&k==0)printf("x=%e,dx=%e,y=%e,dy=%e,z=%e,dz=%e\n",*x,dx,*y,dy,*z,dz);//debug

    *r = (*x) * (*x) + (*y) * (*y) + (*z) * (*z);
    *r = sqrt(*r);
//    if(i==0&&j==0)printf("r=%e,rank=%d\n",*r,rank);//debug
    return;
}

void array_pro(double *rho)//density profile
{
    int i,j,k;

    for(i = 0;i < rsiz;i++)
        rho[i] = 0;
	
//	printf("in array_pro\n");//debug

    for(k = 0;k < nz;k++)
    {
        int kk = k/size;
        if(k % size == rank)
        {
            for(j = 0;j < ny;j++)
            {
                for(i = 0;i < nx;i++)
                {
                    double x,y,z;
                    double r;
                    int pos = i + nx*(j + ny*kk);

                    cal_dis(i, j, k, &x, &y, &z, &r);
/*
                    if(i * dx > xoff_cnt && i * dx < xoff_cnt + dx_cnt * (nx_cnt - 1)
                    && j * dy > yoff_cnt && j * dy < yoff_cnt + dy_cnt * (ny_cnt - 1)
                    && k * dz > zoff_cnt && k * dz < zoff_cnt + dz_cnt * (nz_cnt - 1))
                        continue;
*/
                    if(r < r_max)
                    {
                        double arr_r = array[pos][0];
                        double arr_i = array[pos][1];
                        double den = arr_r * arr_r + arr_i * arr_i;
                        int r_i = (int)rint(r / dr);
                        if(r_i <= 0)
                            r_i = 1;
                        if(r_i >= rsiz)
                            r_i = rsiz;
//			if(r_i==1) printf("den=%e, arr_r=%e, arr_i=%e,i=%d,j=%d,k=%d\n",den,arr_r,arr_i,i,j,k);//debug
                        rho[r_i-1] += den * (dx * dy * dz);
                    }
                }
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, rho, rsiz, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    return;
}

void array_decompose_ylm(int l_max)
{
    int i,j,k;
    int count = 0;
    for(k = 0;k < nz;k++)
    {
        int kk = k/size;
        if(k % size == rank)
        {
            for(j = 0;j < ny;j++)
            {
                for(i = 0;i < nx;i++)
                {
                    double x,y,z;
                    double r;
                    int pos = i + nx*(j + ny*kk);

                    cal_dis(i, j, k, &x, &y, &z, &r);

                    if(i * dx > xoff_cnt && i * dx < xoff_cnt + dx_cnt * (nx_cnt - 1)
                    && j * dy > yoff_cnt && j * dy < yoff_cnt + dy_cnt * (ny_cnt - 1)
                    && k * dz > zoff_cnt && k * dz < zoff_cnt + dz_cnt * (nz_cnt - 1))
                        continue;

                    if(r < r_max)
                    {
                        double arr_r = array[pos][0];
                        double arr_i = array[pos][1];
                        arr_r *= dx * dy * dz;
                        arr_i *= dx * dy * dz;
                        do_decompose_ylm(x, y, z, r, arr_r, arr_i, l_max);
                        count++;
                    }
                }
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    return;
}
void array_init()
{
	int gridnum=nx*ny*(nz/size+1);
	array = (cmpxf*) malloc(gridnum*sizeof(cmpxf));

	return;
}


void array_add_ylm(int l_max,int l_init,bool firstadd)
{
    int i,j,k;
    int count = 0;
// adding timing_flag 2020.12.31
    bool timing_flag;
    double k_start, k_stop;
//

//    printf("in array_add_ylm rank=%d\n",rank);//debug
// add timing 2020.12.31
    if (rank==0)
       k_start = MPI_Wtime();
//
    for(k = 0;k < nz;k++)
    {

        if ( rank == 0 )
        {
#ifdef OCTANT_DECOMPOSE
           printf( "Working on Octant [%d/8] -- Lidx [%3d/%3d] -- k [%4d/%4d] ...\n", octant_global, Lidx_global+1, lnod, k+1, nz );
#else
           printf( "Working on Lidx [%3d/%3d] -- k [%4d/%4d] ...\n", Lidx_global+1, lnod, k+1, nz );
#endif
           fflush( stdout );
        }

        int kk = k/size;
//		printf("rank=%i,k=%d\n",rank,k);//debug
//		printf("r_max=%e\n",r_max);//debug
        if(k % size == rank)
	{
		
	    for(j = 0;j < ny;j++)
            {
                for(i = 0;i < nx;i++)
                {
// add timing and timing_flag 2020.12.31
                    double start, stop;
                    timing_flag = false;
                    if ( j==(int)(ny/2) && i==(int)(nx/2))
                    {
                        timing_flag = true;
                        start = MPI_Wtime();
                    }
//
                    double x,y,z;
                    double r;
//		    int pos = i + nx/2.*(j + ny/2.*k);
                    int pos = i + nx*(j + ny*kk);

#ifdef OCTANT_DECOMPOSE
                    cal_dis_O(i, j, k, &x, &y, &z, &r,octant);
#else
                    cal_dis(i, j, k, &x, &y, &z, &r);
#endif
//		    if(rank==0 &&pos==0) printf("i=%d j=%d k=%d,inside loop ...r =%e r_max=%e\n",i,j,k,r,r_max);//debug
/*
                    if(i * dx > xoff_cnt && i * dx < xoff_cnt + dx_cnt * (nx_cnt - 1)
                    && j * dy > yoff_cnt && j * dy < yoff_cnt + dy_cnt * (ny_cnt - 1)
                    && k * dz > zoff_cnt && k * dz < zoff_cnt + dz_cnt * (nz_cnt - 1))
                        continue;
*/
	 	    if(firstadd){
			array[pos][0]=0.;
			array[pos][1]=0.;	
		    }
		
                    if(r < r_max)
                    {
                        double arr_r = 0;
                        double arr_i = 0;
//			printf("i=%d j=%d k=%d pos=%d before rank=%d\n",i,j,k,pos,rank);//debug
                        do_expand_ylm(x, y, z, r, &arr_r, &arr_i, l_max,l_init,timing_flag);
//			if(pos==78)printf("i=%d j=%d k=%d pos=%d after rank=%d,arr_r=%e\n",i,j,k,pos,rank,arr_r);//debug
                        array[pos][0] += arr_r;
                        array[pos][1] += arr_i;
//			if(pos==78)printf("array[%d][0]=%e,rank=%d\n",pos,array[pos][0],rank);//debug
                        count++;
                    }
// add r_max debug 2020.12.31
    //                else
    //                    if (rank==0)
    //                        printf("(i,j,k) = (%d,%d,%d) with r=%.4e is outside r_max=%.4e!\n",i, j, k, r, r_max);
//
//			if(pos%10000==0)printf("array[%i][0] in arr_add_ylm=%f\n",pos,array[pos][0]);//debug
// add timing 2020.12.31
                    if (timing_flag)
                    {
                        stop = MPI_Wtime();
                        printf("Time for adding contribution of l_init=%d to l_max=%d at a single poisition @ rank%d takes %.4e s.\n", l_init, l_max, rank, stop-start);
//
                    }
                }
            }
	}
    }
//	printf("array[0][0]=%f\n",array[0][0]);//debug
//  printf("count=%d,rank=%d\n",count,rank);//debug
// add timing 2020.12.31
    if (rank==0)
    {
        k_stop = MPI_Wtime();
        double total_time = k_stop-k_start;
        int hr = (int)(total_time/3600);
        int min = (int)((total_time-3600*hr)/60);
        double sec = total_time - 3600.*hr - 60.*min;
        printf("Total time for adding contribution of l_init=%d to l_max=%d over all k takes %.4e s.\n", l_init, l_max, k_stop-k_start);
        fflush(stdout);
    }
 
    MPI_Barrier(MPI_COMM_WORLD);
    return;
}

/*
void array_add_ylm2(int l_max)
{
    int i,j,k;
    int count = 0;

    array = (cmpxf*) malloc(nbox2*nbox2*nbox2*sizeof(cmpxf));
	
    for(k = 0;k < 2*nbox2;k++)
    {
//        int kk = k/size;
//        if(k % size == rank)
///	{
//		printf("rank=%i\n",rank);//debug
//		printf("r_max=%e\n",r_max);//debug
		
	    for(j = 0;j < 2*nbox2;j++)
            {
                for(i = 0;i < 2*nbox2;i++)
                {
                    double x,y,z;
                    double r;
                    int pos = i + nx*(j + ny*k);

                    cal_dis2(i, j, k, &x, &y, &z, &r);
//			if(pos%100000==0) printf("inside loop ^o^...r =%d r_max=%d \n",r,r_max);//debug
//
                    if(i * dx > xoff_cnt && i * dx < xoff_cnt + dx_cnt * (nx_cnt - 1)
                    && j * dy > yoff_cnt && j * dy < yoff_cnt + dy_cnt * (ny_cnt - 1)
                    && k * dz > zoff_cnt && k * dz < zoff_cnt + dz_cnt * (nz_cnt - 1))
                        continue;

                    if(r < r_max)
                    {
                        double arr_r = 0;
                        double arr_i = 0;
                        do_expand_ylm(x, y, z, r, &arr_r, &arr_i, l_max);
//			if(i==128&j==128&k==128) printf("arr_r in arr_add_ylm is %e\n",arr_r);//debug
                        array[pos][0] = arr_r;
                        array[pos][1] = arr_i;
                        count++;
                    }
//			if(pos%100000==0) printf("array[%i][0] in arr_add_ylm=%f\n",pos,array[pos][0]);//debug
                }
            }
//       }
    }
//	printf("array[0][0]=%f\n",array[0][0]);//debug
    MPI_Barrier(MPI_COMM_WORLD);
    return;
}
*/

void array_sub_ylm(int l_max,int l_init)
{
    int i,j,k;
    int count = 0;
// add timing_flag 2020.12.31
    bool timing_flag;
//
    for(k = 0;k < nz;k++)
    {
        int kk = k/size;
        if(k % size == rank)
        {
            for(j = 0;j < ny;j++)
            {
                for(i = 0;i < nx;i++)
                {
                    double x,y,z;
                    double r;
                    int pos = i + nx*(j + ny*kk);

                    cal_dis(i, j, k, &x, &y, &z, &r);
/*
                    if(i * dx > xoff_cnt && i * dx < xoff_cnt + dx_cnt * (nx_cnt - 1)
                    && j * dy > yoff_cnt && j * dy < yoff_cnt + dy_cnt * (ny_cnt - 1)
                    && k * dz > zoff_cnt && k * dz < zoff_cnt + dz_cnt * (nz_cnt - 1))
                        continue;
*/
                    if(r < r_max)
                    {
                        double arr_r = 0;
                        double arr_i = 0;
// add timing_flag 2020.12.31
                        do_expand_ylm(x, y, z, r, &arr_r, &arr_i, l_max,l_init, timing_flag);
//
                        array[pos][0] -= arr_r;
                        array[pos][1] -= arr_i;
                        count++;
                    }
                }
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    return;
}

void cal_r_max(void)
{

    if(rank == 0)
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
    }
    MPI_Bcast( &r_max, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    return;
}
void cal_r_max_O(void)
{
    dx=dx/2.;
    dy=dy/2.;
    dz=dz/2.;

    if(rank == 0)
    {
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
	printf("r_max=%e\n",r_max);//debug
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast( &r_max, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    return;
}


void over_density(double *rho, double target)
{
    int i;
    double enc_den = 0;
    double r;
    for(i = 0;i < rsiz;i++)
    {
        r = (i + 1 + .5) * dr;
        double vol = 4 * M_PI/3 * r*r*r;
        enc_den += rho[i];
        double den = enc_den / vol;
        if(den < target && enc_den != 0)
            break;
    }

    if(rank == 0)
    if(r > r_max)
        r = r_max;
    r_max = r;

    for(i = 0;i < rsiz;i++)
    {
        r = (i + 1) * dr;
        if(r > r_max)
            rho[i] = 0;
    }

    MPI_Bcast( &r_max, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    return;
}

void sample_error(int error, char *string)
{
  fprintf(stderr, "Error %d in %s\n", error, string);
  MPI_Finalize();
  exit(-1);
}
void add_global_array()
{
	array_cnt = (cmpxf*) malloc(nx*ny*nz*sizeof(cmpxf));
	MPI_Reduce(array,array_cnt, nx*ny*nz*2, MPI_FLOAT,MPI_SUM, 0, MPI_COMM_WORLD);

	return;
}

