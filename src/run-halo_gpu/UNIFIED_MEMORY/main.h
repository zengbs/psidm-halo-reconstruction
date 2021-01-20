/*
 COMPLEX
 */

// adding header extension.h at 2020.12.12
//
// adding convergence check 2020.12.14
// modifying lsize calculation 2020.12.25
// adding output final potential profile 2020.12.27
// adding data dumping and restarting 2021.01.04
// adding iter_test 2021.01.07

typedef double cmpx[2];
typedef float cmpxf[2];

const int strsiz = 1024;

const double r_fact = .999;

//extern cmpxf* array;
//extern cmpxf* array_cnt;
//extern int    *rfunc_num;
//extern double **rv   ;
//extern double ***rfunc;
extern int ubound_l;
extern int *ubound_n;
extern int lbound_l;
extern int *lbound_n;

extern double mass;
extern double newton_g;
extern double planck_h;
extern double eta      ;
extern double red_shift;
extern double scale_fac;
extern double box_range;
extern double r_range;
extern double den_vir;
extern double r_eigen;//energy cut= V(r_eigen)

//adding cosmology parameters
//extern double Omega_m0;
//extern double Omega_lambda0;
//extern double h0;
//extern double mass_core_ratio;
//
//adding convergence criteria
extern double criteria;

extern int rsiz;
extern int lsiz;
extern int Lsiz;
extern int nsiz;
extern int lnod;
extern int lnod_id;
extern int l_rest;

extern int nbox;
extern int nbox2;
extern int nx;
extern int ny;
extern int nz;
extern double box_dr;
extern double dx;
extern double dy;
extern double dz;

extern int nbox_cnt;
extern int nx_cnt;
extern int ny_cnt;
extern int nz_cnt;
extern double box_dr_cnt;
extern double dx_cnt;
extern double dy_cnt;
extern double dz_cnt;

extern int    scal_cnt;
extern double xoff_cnt;
extern double yoff_cnt;
extern double zoff_cnt;

extern double dr;

extern double xcen;
extern double ycen;
extern double zcen;
extern double r_max;

extern char   file_r[strsiz];
extern char   file_i[strsiz];

extern char   file_r_cnt[strsiz];
extern char   file_i_cnt[strsiz];
extern char   init_pot[strsiz];
extern char   final_pot[strsiz];
//extern char   convergence_file[strsiz];

extern int size;
extern int rank;

extern double *dis_hist;
extern double *dos_hist;
extern double *dis_lbin;
extern double *dos_lbin;
extern double **power;
extern double **power_dev;

extern char filename_amp[strsiz];
extern char filename_ylm[strsiz];
extern char filename_eigen[strsiz];

/*
Fermionic King model parameters.
distribution function:	f(E)=A(exp(E-Ec)-1)/(1+A/mu0*exp(E-Ec))
 */
extern double A;
extern double beta;
extern double Ec;
extern double mu0;
extern double amp_gs_r;
extern double amp_gs_i;
extern double halomass;
extern double perturb_frac;
extern double r_c;

extern int iter_i;
extern int iter_max;
extern int octant;

// adding cosmology parameters 2020.12.10
extern double Omega_m0;
extern double Omega_lambda0;
extern double h0;
//
// adding newly defined parameters 2020.12.10
extern double m_a_22;
extern double potential_scale;
extern double mass_core_ratio;
//
// adding NFW parameters 2020.12.10
extern double NFW_C;
//extern double NFW_A;
extern double NFW_rho_0;
extern double r_vir;
extern double r_e;
extern double R_s;
//
// adding data dumping and restarting 2021.01.04
extern int dump_flag;
extern int dump_interval;
extern int dump_id;
extern int restart_flag;
extern int restart_id;
//
// adding iter_test 2021.01.07
extern int iter_test;

//void fake_arr(int nx, int ny, int nz);
void read_para(const char *parafile);
void run(void);
double rand_normal(double mean, double stddev);
double King(double energy,double A,double beta,double Ec);
double Fermionic_King(double energy,double A,double beta,double Ec,double mu0);
void array_add_ylm(int l_max);
void Reset_GS_amp(double* amp_gs_r,double* amp_gs_i,double gs_eigenfun[]);
double MassRadius(double gs_rho[],double r[],int i);
void gen_arti_halo(clock_t t,int l_init,int l_max);
void set_2ndbox(void);
