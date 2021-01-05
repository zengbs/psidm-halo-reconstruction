#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<mpi.h>
#include"macros.h"
#include"main.h"
#include"arr.h"
#include"ylm.h"
#include"ubound.h"

int gen_lbound(int l_max)
{
    int l,n;

    lbound_l = ubound_l;
    if(lbound_n != NULL)
        free(lbound_n);

    lbound_n = (int*) malloc((l_max+1)*sizeof(int));

    for(l = 0;l <= lbound_l;l++)
    {
        lbound_n[l] = ubound_n[l];
    }
//    MPI_Bcast(&lbound_l,       1, MPI_INT, 0, MPI_COMM_WORLD);
//    MPI_Bcast( lbound_n, l_max+1, MPI_INT, 0, MPI_COMM_WORLD);

    return 0;
}

int gen_ubound(double e_ubound, int l_max, int l_init)
{
    int l,n;

    ubound_l = l_max;
    if(ubound_n != NULL)
        free(ubound_n);

    ubound_n = (int*) malloc((l_max+1-l_init)*sizeof(int));

    for(l = l_init;l <= l_max;l++)
    {
        ubound_n[l-l_init] = rfunc_num[l-l_init] - 1;
        for(n = 0;n < rfunc_num[l-l_init];n++)
        {
            if(rv[l-l_init][n] > e_ubound)
            {
                ubound_n[l-l_init] = n - 1;
                break;
            }
        }
        if(ubound_n[l-l_init] < 0)
        {
            ubound_l = l-l_init - 1;
            break;
        }
    }
//    MPI_Bcast(&ubound_l,       1, MPI_INT, 0, MPI_COMM_WORLD);
//    MPI_Bcast( ubound_n, l_max+1-l_init, MPI_INT, 0, MPI_COMM_WORLD);

    return 0;
}
