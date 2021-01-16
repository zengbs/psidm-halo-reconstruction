#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include"macros.h"
#include"text_io.h"

void show_txt(int **indices, int icount, double **values, int fcount, double count, char *filename)
{
    int i,j;

    FILE *pf = fopen(filename,"w");
    for(i = 0;i < count;i++)
    {
        for(j = 0; j < icount; j++)
            fprintf(pf,"%7d ", indices[j][i]);
        for(j = 0; j < fcount; j++)
            fprintf(pf,"%15.10le ", values[j][i]);
        fprintf(pf,"\n");
    }
    fclose(pf);
    return;
}

void read_txt(int **indices, int icount, double **values, int fcount, double count, char *filename)
{
    int i,j;

    FILE *pf = fopen(filename,"r");
    for(i = 0;i < count;i++)
    {
        for(j = 0; j < icount; j++)
            fscanf(pf,"%d", &(indices[j][i]));
        for(j = 0; j < fcount; j++)
            fscanf(pf,"%lg", &(values[j][i]));
        fprintf(pf,"\n");
    }
    fclose(pf);
    return;
}

