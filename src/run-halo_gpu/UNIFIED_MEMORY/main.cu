#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<unistd.h>
#include<time.h>
#include<math.h>
#include<stdbool.h>
#include<cuda_runtime.h> 
#include"main.h"
#include"arr.h"
#include"ylm.h"
#include"macros.h"
#include"text_io.h"
#include"solve_eigenvalues.h"
#include"extension.h"
//

using namespace std; 

/*
 global parameters
 */

int Lidx_global=-1;
int octant_global=-1;
int eigen_num;

double mass      = 0.;
double newton_g  = 0.;
double planck_h  = 0.;
double eta       = 0.;
double red_shift = 0.;
double scale_fac = 0.;

extern double newton_g_mks;
extern double mass_of_sun;
extern double kpc_to_meter;
extern double electron_charge;
extern double speed_of_light;
extern double hbar_mks;

// adding data dumping and restarting 2021.01.04
int dump_flag = -1;
int dump_interval = 0;
int dump_id = 0;
int restart_flag = -1;
int restart_id = 0;
// adding cosmology parameters 2020.12.10
double Omega_m0  = 0.;
double Omega_lambda0  = 0.;
double h0 = 0.;
//
// adding newly defined parameters 2020.12.10
double m_a_22 = 0.;
double mass_core_ratio = 0.;
double phase_between_RI = 0.;  // in unit of 2\pi
double potential_scale = 0.;
//
// adding NFW parameters 2020.12.10
double NFW_C = 0.;
double NFW_A = 0.;
double NFW_rho_0 = 0.;
double r_vir = 0.;
double r_e = 0.;
double R_s = 0.;
//
// adding convergence criteria 2020.12.14
double criteria = 0.;
//
// adding contrast factor for determining virial radius 2020.12.16
double CF = 0.;
//
// adding GPU parameters 2021.01.09
int tpB_x, tpB_y, tpB_z;
int bpG_x, bpG_y, bpG_z;
int N_site;
//

double box_range = 0.;
double r_range   = 0.;
double den_vir   = 0.;
double r_eigen   =0.;//energy cut= V(r_eigen)

double box_dr = 0.;
double dx = box_dr;
double dy = box_dr;
double dz = box_dr;

double box_dr_cnt = 0.;
double dx_cnt;
double dy_cnt;
double dz_cnt;

int scal_cnt = 1;
double xoff_cnt = 0.;
double yoff_cnt = 0.;
double zoff_cnt = 0.;


double dr;
double r_max;

/*
box size
*/
int nbox = 0;
int nbox2= 0;
int nx = nbox;
int ny = nbox;
int nz = nbox;

int nbox_cnt = 0;
int nx_cnt;
int ny_cnt;
int nz_cnt;
/*
the total grid points of the eigenstates solver
maximum l number
maximum states number per l mode
*/
int rsiz = 4096;
int lsiz = 100;
int Lsiz = 100;
int nsiz = 100;
int lnod = 1;
//modify lsiz calculation (2020.12.25)
int l_rest = 0; 

double xcen;
double ycen;
double zcen;
/*
 array file
*/
char   file_r[strsiz];
char   file_i[strsiz];

char   file_r_cnt[strsiz];
char   file_i_cnt[strsiz];
/* initial potential guess file */
char   init_pot[strsiz];
/* final potential output file */
char   final_pot[strsiz];
/* convergence error output file */
//char   convergence_file[strsiz];

/*
 global variables
*/

//cmpxf  *array     = NULL;
//cmpxf  *array_cnt = NULL;
int    *rfunc_num = NULL;
float **rv       = NULL;
float ***rfunc   = NULL;
//double **rv       = NULL;
//double ***rfunc   = NULL;
float *array_r_host, *array_i_host;
float ***amplitude_r, ***amplitude_i;

cmpx ***amplitude_host;
//int *rfunc_num_host;
//double **rv_host;
//double ***rfunc_host;

int ubound_l   = -1;
int *ubound_n  = NULL;
int lbound_l   = -1;
int *lbound_n  = NULL;
/*
char filename_amp[strsiz] = "amp";
char filename_ylm[strsiz] = "ylm";
char filename_eigen[strsiz] = "egn";
*/
/*
Fermionic King model parameters.
distribution function:	f(E)=A(exp(E-Ec)-1)/(1+A/mu0*exp(E-Ec))
 */
double A=0.;
double beta=0.;
double Ec=0.;
double mu0=0.;
double amp_gs_r=0.;  // will be M_soliton_r(r_e)*cos(phase)
double amp_gs_i=0.;  // will be M_soliton_r(r_e)*sin(phase)

double halomass=0.;
double perturb_frac=0.;
double r_c=0.;

int iter_i=0;
int iter_max=12;
int octant=0;

int main(int argc, char** argv)
{
//	long seed = 123456789;
	long seed = 49651525;

#       ifdef OCTANT_DECOMPOSE
        printf( "Octant decomposition: ON\n" );
#       else
        printf( "Octant decomposition: OFF\n" );
#       endif

//	clock_t t;
	srand(seed);
//	srand(time(NULL));

//	read_para("parameter_r.txt");//read parameters

  	read_para( argv[1] );

        octant_global = octant;

//        if ( lnod > size )
//        {
//           fprintf( stderr, "ERROR : LNOD (%d) > # of MPI ranks (%d) !!\n", lnod, size );
//           MPI_Abort( MPI_COMM_WORLD, 1 );
//        }

	run();
	
	return 0;
}

void set_para(char *paraname, char *paravalue, char *readname, char *readvalue, char type)
{
        if(strcmp(readname, paraname) == 0)
        {
            if(type == 'i')
                *((int    *)paravalue) = atoi(readvalue);
            if(type == 'd')
                *((double *)paravalue) = atof(readvalue);
            if(type == 's')
                strcpy (paravalue, readvalue);
        }
}

//void gen_arti_halo(clock_t t,int l_init,int l_max)
//{
//	/// generate amplitudes ///
//	gen_amp(l_init,l_max,A,beta,Ec,mu0,amp_gs_r,amp_gs_i,false);//gen amp in rank 0 and Bcast to all nodes
//	printf("amplitude finished\n");
//	
//	/// call r_max ///
//	cal_r_max();
//
//	/// generate upper bound ///
//	gen_ubound(rv[0][0] * 0., l_max,l_init);
//
//	/// construct density profile ///
//	t=clock();
//	den_profile(rho,l_init,l_max);
//	t=clock()-t;
//	printf("density profile takes %e minutes, rank=%d\n",((float)t)/CLOCKS_PER_SEC/60.,rank);
//
//	/// generate artificial halo density profile and potential ///
//	//bool isartihalo=true;
//	//gen_output(rho, isartihalo);
//
//	return;
//}//end of gen_arti_halo

void read_para(const char *parafile)
{

    FILE *pf;
        pf = fopen(parafile,"r");

    while(1)
    {
        char paraname [strsiz];
        char paravalue[strsiz];
        int ret;
        ret = fscanf(pf,"%s%s", paraname, paravalue);
        if(ret == EOF)
            break;
        if(ret != 2)
            continue;

        set_para("MASS"      , (char *)(& mass      ), paraname, paravalue, 'd');
//        set_para("NEWTON_G"  , (char *)(& newton_g  ), paraname, paravalue, 'd');
        set_para("PLANCK_H"  , (char *)(& planck_h  ), paraname, paravalue, 'd');
//        set_para("ETA"       , (char *)(& eta       ), paraname, paravalue, 'd');
        set_para("RED_SHIFT" , (char *)(& red_shift ), paraname, paravalue, 'd');
//	set_para("SCALE_FAC" , (char *)(& scale_fac ), paraname, paravalue, 'd');
	set_para("FILE_R"    , (char *)(& file_r    ), paraname, paravalue, 's');
	set_para("FILE_I"    , (char *)(& file_i    ), paraname, paravalue, 's');
//      set_para("FILE_R_CNT", (char *)(& file_r_cnt), paraname, paravalue, 's');
//	set_para("FILE_I_CNT", (char *)(& file_i_cnt), paraname, paravalue, 's');
        set_para("BOX_RANGE" , (char *)(& box_range ), paraname, paravalue, 'd');
        set_para("R_RANGE"   , (char *)(& r_range   ), paraname, paravalue, 'd');
        set_para("R_EIGEN"   , (char *)(& r_eigen   ), paraname, paravalue, 'd');
//	set_para("DEN_VIR"   , (char *)(& den_vir   ), paraname, paravalue, 'd');
        set_para("NBOX"      , (char *)(& nbox      ), paraname, paravalue, 'i');
        set_para("NBOX2"     , (char *)(& nbox2     ), paraname, paravalue, 'i');
        set_para("RSIZ"      , (char *)(& rsiz      ), paraname, paravalue, 'i');
        set_para("LSIZ"      , (char *)(& Lsiz      ), paraname, paravalue, 'i');
        set_para("NSIZ"      , (char *)(& nsiz      ), paraname, paravalue, 'i');
        set_para("LNOD"      , (char *)(& lnod      ), paraname, paravalue, 'i');
// A will be calculated automatically 2020.12.21
//        set_para("A"         , (char *)(& A         ), paraname, paravalue, 'd');
        set_para("BETA"      , (char *)(& beta      ), paraname, paravalue, 'd');
        set_para("ESCAPE_E"  , (char *)(& Ec        ), paraname, paravalue, 'd');
        set_para("MU0"       , (char *)(& mu0      ), paraname, paravalue, 'd');
        set_para("HALOMASS"  , (char *)(& halomass  ), paraname, paravalue, 'd');
//        set_para("XCEN"      , (char *)(& xcen      ), paraname, paravalue, 'd');
//        set_para("YCEN"      , (char *)(& ycen      ), paraname, paravalue, 'd');
//        set_para("ZCEN"      , (char *)(& zcen      ), paraname, paravalue, 'd');
//        set_para("GSAMP_R"   , (char *)(& amp_gs_r  ), paraname, paravalue, 'd');
//        set_para("GSAMP_I"   , (char *)(& amp_gs_i  ), paraname, paravalue, 'd');
        set_para("PERTURB"   , (char *)(&perturb_frac),paraname, paravalue, 'd');
//        set_para("CORE_R"    , (char *)(&r_c        ), paraname, paravalue, 'd');
        set_para("ITER_I"    , (char *)(&iter_i     ), paraname, paravalue, 'i');
        set_para("ITER_MAX"  , (char *)(&iter_max   ), paraname, paravalue, 'i');
        set_para("OCTANT"    , (char *)(&octant     ), paraname, paravalue, 'i');
//	set_para("NBOX_CNT"  , (char *)(& nbox_cnt  ), paraname, paravalue, 'i');
//	set_para("SCAL_CNT"  , (char *)(& scal_cnt  ), paraname, paravalue, 'i');
	set_para("INIT_POT"  , (char *)(& init_pot  ), paraname, paravalue, 's');
// adding final potential output file 2020.12.27
	set_para("FINAL_POT"  , (char *)(& final_pot  ), paraname, paravalue, 's');
// adding parameters 2020.12.10
	set_para("OMEGA_M0"  , (char *)(& Omega_m0  ), paraname, paravalue, 'd');
//	set_para("OMEGA_LAMBDA0"  , (char *)(& Omega_lambda0  ), paraname, paravalue, 'd');
	set_para("SMALL_H0"  , (char *)(& h0  ), paraname, paravalue, 'd');
	set_para("MASS_CORE_RATIO"  , (char *)(& mass_core_ratio  ), paraname, paravalue, 'd');
	set_para("PHASE_BETWEEN_RI"  , (char *)(& phase_between_RI  ), paraname, paravalue, 'd');
	set_para("NFW_A"  , (char *)(& NFW_A  ), paraname, paravalue, 'd');
	set_para("NFW_C"  , (char *)(& NFW_C  ), paraname, paravalue, 'd');
// adding convergence check 2020.12.14
	set_para("CRITERIA"  , (char *)(& criteria  ), paraname, paravalue, 'd');
//	set_para("CONVERGENCE_FILE"  , (char *)(& convergence_file  ), paraname, paravalue, 's');
//
// adding contrast factor for determing virial radius 2020.12.16
	set_para("CONTRAST_FACTOR"  , (char *)(& CF  ), paraname, paravalue, 'd');
//
// adding data dumping and restarting 2021.01.04
	set_para("DUMP_FLAG"  , (char *)(& dump_flag  ), paraname, paravalue, 'i');
	set_para("DUMP_INTERVAL"  , (char *)(& dump_interval  ), paraname, paravalue, 'i');
	set_para("RESTART_FLAG"  , (char *)(& restart_flag  ), paraname, paravalue, 'i');
	set_para("RESTART_ID"  , (char *)(& restart_id  ), paraname, paravalue, 'i');
//       
// adding GPU parameters 2021.01.09
	set_para("THREADS_PER_BLOCK_X"  , (char *)(& tpB_x  ), paraname, paravalue, 'i');
	set_para("THREADS_PER_BLOCK_Y"  , (char *)(& tpB_y  ), paraname, paravalue, 'i');
	set_para("THREADS_PER_BLOCK_Z"  , (char *)(& tpB_z  ), paraname, paravalue, 'i');
//	set_para("BLOCKS_PER_GRID"  , (char *)(& bpG  ), paraname, paravalue, 'i');
//        
    }
    fclose(pf);

    nx = nbox;
    ny = nbox;
    nz = nbox;
    nx_cnt = nbox_cnt;
    ny_cnt = nbox_cnt;
    nz_cnt = nbox_cnt;

    box_dr = box_range / nbox;
    dx = box_dr;
    dy = box_dr;
    dz = box_dr;
    box_dr_cnt = box_dr / scal_cnt;
    dx_cnt = box_dr_cnt;
    dy_cnt = box_dr_cnt;
    dz_cnt = box_dr_cnt;
    xoff_cnt = (nx - 1) * dx / 2 - (nx_cnt - 1) * dx_cnt / 2;
    yoff_cnt = (ny - 1) * dy / 2 - (ny_cnt - 1) * dy_cnt / 2;
    zoff_cnt = (nz - 1) * dz / 2 - (nz_cnt - 1) * dz_cnt / 2;

    bpG_x = (nbox+tpB_x-1)/tpB_x;
    bpG_y = (nbox+tpB_y-1)/tpB_y;
    bpG_z = (nbox+tpB_z-1)/tpB_z;
    N_site = nx*ny*nz;
    dr = r_range / rsiz;
    scale_fac = 1./(1.+red_shift);
    phase_between_RI *= 2.*M_PI;
//    determing xcen, ycen, zcen automatically 2020.12.29
    srand(3413567);
    float randnum;
    randnum = 5e-3*(((float)rand()/RAND_MAX)-0.5);
    xcen = box_range*(0.5+randnum);
    randnum = 5e-3*(((float)rand()/RAND_MAX)-0.5);
    ycen = box_range*(0.5+randnum);
    randnum = 5e-3*(((float)rand()/RAND_MAX)-0.5);
    zcen = box_range*(0.5+randnum);
//
//    modify lsiz calculation 2020.12.25
    l_rest = (Lsiz+1)%lnod;
//    debug
//    printf("%.4f\t%.4f\t%.4f\n", halomass, mass_core_ratio, red_shift);
//    adding calculation of m_a_22 2020.12.10
    m_a_22 = mass/1e-22;
//    adding calculation of Omega_lambda0 2020.12.10
    Omega_lambda0 = 1.-Omega_m0;
//    adding calculation of r_c 2020.12.10
    r_c = r_core();  // in unit of Mpc/h
//    printf("Core radius is %.16e Mpc/h.\n", r_c);
//    adding calculation of NFW parameters
    NFW_A = log(1+NFW_C)-NFW_C/(1.+NFW_C);
    double H0 = h0*100./kpc_to_meter;                                                                           //  km/s/Mpc converted to unit of s^-1
    double rho_critical = 3.*H0*H0/8./M_PI/newton_g_mks;                                                        // in unit of kg/m^3
    double rho_critical_z_matter = (rho_critical*Omega_m0)/pow(1./(1.+red_shift),3.);                           // in unit of kg/m^3
    r_vir = 1e-3*h0*pow((halomass*1e11*mass_of_sun/(4./3.*M_PI*CF*rho_critical_z_matter)),1./3.)/kpc_to_meter; // in unit of Mpc/h
    NFW_rho_0 = halomass*1e11/(4.*M_PI*pow(r_vir/NFW_C/r_c,3.)*NFW_A);                                          // in unit of 1e11 M_sun/r_c^3
    R_s = r_vir/NFW_C/r_c;                                                                                      // in unit of r_c
    r_e = find_re_bisection(rho_soliton_r, rho_NFW_r, 1.*r_c, 10.*r_c);                                         // in unit of Mpc/h
    double M_soliton_at_re = M_soliton_r(r_e);
    amp_gs_r = sqrt(M_soliton_at_re)*cos(phase_between_RI);
    amp_gs_i = sqrt(M_soliton_at_re)*sin(phase_between_RI);
    newton_g = (3.*Omega_m0*pow(H0,2.)/(8.*M_PI))/pow(H0,2.);
    potential_scale = newton_g*scale_fac/(r_c)/newton_g_mks;
    eta = pow(newton_g*pow(hbar_mks/(m_a_22*1e-22*electron_charge/pow(speed_of_light,2.))/scale_fac,2.)/newton_g_mks/(pow(kpc_to_meter*1e3/h0,4.)*(rho_critical*Omega_m0)),-0.5);

// automatically calculate A 2020.12.21
    A = pow(2.*pow(dr/h0*1e3*kpc_to_meter,3.)*rho_critical*Omega_m0/mass_of_sun/1e11,0.5);
//
// determine r_eigen by virial radius if R_EIGEN is not specified 2021.01.04 */
    if (r_eigen==0.)
        r_eigen = r_vir*1.01;
//

    printf("Input Parameters:\n");
    printf("R_RANGE = %.8e\n", r_range);
    printf("BOX_RANGE = %.8e\n", box_range);
    printf("R_EIGEN = %.8e\n", r_eigen);
    printf("NBOX = %d\n", nbox);
    printf("RSIZ = %d\n", rsiz);
    printf("LSIZ = %d\n", Lsiz);
    printf("NSIZ = %d\n", nsiz);
    printf("LNOD = %d\n", lnod);
    printf("A = %.8e\n", A);
    printf("BETA = %.8e\n", beta);
    printf("ESCAPE_E = %.8e\n", Ec);
    printf("MU0 = %.8e\n", mu0);
    printf("HALO_MASS = %.8e\n", halomass);
    printf("PERTURB = %.8e\n", perturb_frac);
    printf("OMEGA_M0 = %.8e\n", Omega_m0);
    printf("SMALL_H0 = %.8e\n", h0);
    printf("MASS_CORE_RATIO = %.8e\n", mass_core_ratio);
    printf("NFW_C = %.8e\n", NFW_C);
    printf("PHASE_BETWEEN_RI = %.8e\n", phase_between_RI);
    printf("CRITERIA = %.8e\n", criteria);
    printf("CONTRAST_FACTOR = %.8e\n", CF);
# ifdef OCTANT_DECOMPOSE
    printf("OCTANT = %d\n", octant);
# endif
    printf("DUMP_FLAG = %d\n", dump_flag);
    printf("DUMP_INTERVAL = %d\n", dump_interval);
    printf("RESTART_FLAG = %d\n", restart_flag);
    printf("RESTART_ID = %d\n", restart_id);
    printf("THREADS_PER_BLOCK_X = %d\n", tpB_x);
    printf("THREADS_PER_BLOCK_Y = %d\n", tpB_y);
    printf("THREADS_PER_BLOCK_Z = %d\n", tpB_z);
    printf("BLOCKS_PER_GRID_X = %d\n", bpG_x);
    printf("BLOCKS_PER_GRID_Y = %d\n", bpG_y);
    printf("BLOCKS_PER_GRID_Z = %d\n", bpG_z);
    printf("=================================================\n");
    printf("Calculated Parameters:\n");
    printf("ETA = %.8e\n", eta);
    printf("CORE_R = %.8e\n", r_c);
    printf("VIRIAL_R = %.8e ; POTENTIAL_VIRIAL = %.8e .\n", r_vir, phi_total_r(r_vir));
    printf("EQUAL_R = %.8e\n", r_e);
    printf("GSAMPR = %.8e ; GSAMP_I = %.8e ; squre sum = % .8e\n", amp_gs_r, amp_gs_i, M_soliton_at_re);
    printf("XCEN = %.4e ; YCEN = %.4e ; ZCEN = % .4e\n", xcen, ycen, zcen);
    printf("=================================================\n");

    FILE *parameter_log = fopen("Record__Note", "w");
    fprintf(parameter_log, "Input Parameters:\n");
    fprintf(parameter_log, "R_RANGE = %.8e\n", r_range);
    fprintf(parameter_log, "BOX_RANGE = %.8e\n", box_range);
    fprintf(parameter_log, "R_EIGEN = %.8e\n", r_eigen);
    fprintf(parameter_log, "NBOX = %d\n", nbox);
    fprintf(parameter_log, "RSIZ = %d\n", rsiz);
    fprintf(parameter_log, "LSIZ = %d\n", Lsiz);
    fprintf(parameter_log, "NSIZ = %d\n", nsiz);
    fprintf(parameter_log, "LNOD = %d\n", lnod);
    fprintf(parameter_log, "A = %.8e\n", A);
    fprintf(parameter_log, "BETA = %.8e\n", beta);
    fprintf(parameter_log, "ESCAPE_E = %.8e\n", Ec);
    fprintf(parameter_log, "MU0 = %.8e\n", mu0);
    fprintf(parameter_log, "HALO_MASS = %.8e\n", halomass);
    fprintf(parameter_log, "PERTURB = %.8e\n", perturb_frac);
    fprintf(parameter_log, "OMEGA_M0 = %.8e\n", Omega_m0);
    fprintf(parameter_log, "SMALL_H0 = %.8e\n", h0);
    fprintf(parameter_log, "MASS_CORE_RATIO = %.8e\n", mass_core_ratio);
    fprintf(parameter_log, "NFW_C = %.8e\n", NFW_C);
    fprintf(parameter_log, "PHASE_BETWEEN_RI = %.8e\n", phase_between_RI);
    fprintf(parameter_log, "CRITERIA = %.8e\n", criteria);
    fprintf(parameter_log, "CONTRAST_FACTOR = %.8e\n", CF);
# ifdef OCTANT_DECOMPOSE
    fprintf(parameter_log, "OCTANT = %d\n", octant);
# endif
    fprintf(parameter_log, "DUMP_FLAG = %d\n", dump_flag);
    fprintf(parameter_log, "DUMP_INTERVAL = %d\n", dump_interval);
    fprintf(parameter_log, "RESTART_FLAG = %d\n", restart_flag);
    fprintf(parameter_log, "RESTART_ID = %d\n", restart_id);
    fprintf(parameter_log, "THREADS_PER_BLOCK_X = %d\n", tpB_x);
    fprintf(parameter_log, "THREADS_PER_BLOCK_Y = %d\n", tpB_y);
    fprintf(parameter_log, "THREADS_PER_BLOCK_Z = %d\n", tpB_z);
    fprintf(parameter_log, "BLOCKS_PER_GRID_X = %d\n", bpG_x);
    fprintf(parameter_log, "BLOCKS_PER_GRID_Y = %d\n", bpG_y);
    fprintf(parameter_log, "BLOCKS_PER_GRID_Z = %d\n", bpG_z);
    fprintf(parameter_log, "=================================================\n");
    fprintf(parameter_log, "Calculated Parameters:\n");
    fprintf(parameter_log, "ETA = %.8e\n", eta);
    fprintf(parameter_log, "CORE_R = %.8e\n", r_c);
    fprintf(parameter_log, "VIRIAL_R = %.8e ; POTENTIAL_VIRIAL = %.8e .\n", r_vir, phi_total_r(r_vir));
    fprintf(parameter_log, "EQUAL_R = %.8e\n", r_e);
    fprintf(parameter_log, "GSAMPR = %.8e ; GSAMP_I = %.8e ; squre sum = % .8e\n", amp_gs_r, amp_gs_i, M_soliton_at_re);
    fprintf(parameter_log, "XCEN = %.4e ; YCEN = %.4e ; ZCEN = % .4e\n", xcen, ycen, zcen);
    fprintf(parameter_log, "=================================================\n");
    fclose(parameter_log);

//    debug
//    printf("Virial radius is %.16e Mpc/h.\n", r_vir);
//    printf("The radius where rho_soliton and rho_NFW are equal is %.16e Mpc/h.\n", r_e);
//    printf("NFW Mass inside %.8e is %.16e .\n", r_vir, M_NFW_r(r_vir));
//    printf("%.16e\n", phi_total_r(4.*r_c));
//    printf("%.16e\n", phi_NFW_r(3.*r_c));
//    printf("%.8e\t%.8e\n", newton_g, eta);

//    adding calculation of mass of soliton inside a given radius
//    double M_core = M_soliton_r(r_c);
//    printf("Soliton mass inside %.8e is %.16e .\n", r_c, M_core);
//    printf("Soliton mass inside %.8e is %.16e .\n", r_e, M_soliton_at_re);
    return;
}
/*The authors of this work have released all rights to it and placed it
 *in the public domain_single_lidx_single_lidx under the Creative Commons CC0 1.0 waiver
 *(http://creativecommons.org/publicdomain_single_lidx_single_lidx/zero/1.0/).
 * 
 *THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 *EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 *MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 *IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 *CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 *TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 *SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 * 
 *Retrieved from: http://en.literateprograms.org/Box-Muller_transform_(C)?oldid=7011
 */
double rand_normal(double mean, double stddev)
{//Box muller method
    static double n2 = 0.0;
    static int n2_cached = 0;
    if (!n2_cached)
    {
        double x, y, r;
        do
        {
            x = 2.0*rand()/RAND_MAX - 1;
            y = 2.0*rand()/RAND_MAX - 1;

            r = x*x + y*y;
        }
        while (r == 0.0 || r > 1.0);
        {
            double d = sqrt(-2.0*log(r)/r);
            double n1 = x*d;
            n2 = y*d;
            double result = n1*stddev + mean;
            n2_cached = 1;
            return result;
        }
    }
    else
    {
        n2_cached = 0;
        return n2*stddev + mean;
    }
}
//double King(double energy,double A,double beta,double Ec)
//{	
//	double p=A*(exp(-beta*(energy-Ec))-1.);
//	if(p>=0.)
//		return p;
//	else
//		return 0.;
//}
//double Fermionic_King(double energy,double A,double beta,double Ec,double mu0)
//{
//	double p=A*(exp(-beta*(energy-Ec))-1.)/(1.+exp(-beta*(energy-Ec-mu0)));
//	if(p>=0.)
//		return p;
//	else
//		return 0.;
//}
//void Reset_GS_amp(double* amp_gs_r,double* amp_gs_i,double gs_eigenfun[])
//{
//	double mass_times_radius=8.56942249562e-07;
//	double* gs_rho;
//	double mass_radius=0.;
//	bool reset=true;
//
//	if(rank==0)
//	{
//		gs_rho=(double*)malloc(rsiz*sizeof(double));
//		for(int i=0;i<rsiz;i++)
//		{
//			gs_rho[i]=((*amp_gs_r)*(*amp_gs_r)+(*amp_gs_i)*(*amp_gs_i))*gs_eigenfun[i]*gs_eigenfun[i]/4./M_PI;
//			if(gs_rho[i]<=0.5*gs_rho[0]){
//				mass_radius=MassRadius(gs_rho,r,i);
//				printf("mass_radius=%e\n",mass_radius);//debug
//				if(reset){
//				(*amp_gs_r)=(*amp_gs_r)*sqrt(mass_times_radius/mass_radius);
//				(*amp_gs_i)=(*amp_gs_i)*sqrt(mass_times_radius/mass_radius);
//					i=-1;
//					reset=false;
//					continue;
//				}
//
//				break;
//			}
//				
//		}
//
//		
//		free(gs_rho);
//	}
//        MPI_Barrier(MPI_COMM_WORLD);
//	MPI_Bcast(amp_gs_r, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
//	MPI_Bcast(amp_gs_i, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
//	return;
//}//end Reset_GS_amp
//double MassRadius(double gs_rho[],double r[],int i)
//{
//	double mass_radius=0.;
//	double mass=0.;
//	
//	for(int ri=0;ri<=i;ri++)
//	{
//		mass+=4.*M_PI*r[ri]*r[ri]*dr*gs_rho[ri];
//	}
//	printf("i=%d,mass=%e,radius=%e\n",i,mass,r[i]);
//	mass_radius=mass*r[i];
//	
//	return mass_radius;
//}
//
//void set_2ndbox(void)
//{
//	strcpy(file_r,"32Cube_Center_lv6_Binary_Real");
//	strcpy(file_i,"32Cube_Center_lv6_Binary_Imag");
//
//	int ixcen = (int)rint(xcen / dx);
//    	int iycen = (int)rint(ycen / dy);
//    	int izcen = (int)rint(zcen / dz);
//
//	nx=nbox2*2;
//	ny=nbox2*2;
//	nz=nbox2*2;
//	
//	xcen=xcen-(ixcen-nbox2/2)*dx;
//	ycen=ycen-(iycen-nbox2/2)*dy;
//	zcen=zcen-(izcen-nbox2/2)*dz;
////	printf("xcen=%e ycen=%e zcen=%e\n",xcen,ycen,zcen);
//
//	dx=dx/2.;
//	dy=dy/2.;
//	dz=dz/2.;
//
//	return;
//}
