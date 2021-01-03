/*addind convergence check 2020.12.14*/
int solve_init(void);
int solve_fin(void);
int do_solve(bool readpot,char *potfile);
int gen_ubound(double e_ubound, int l_max);
int gen_r(double *r_1, double *r);
int gen_pot(double *pot, double *rho, double *r_1,double g);
int gen_ang(double *pot_l, double *pot, double *r_1, int l);
int solve_eigenvalue(double *pot, double e_max, int *eg_count, double *eigenvalues, double *eigenfunctions);
void load_pot(double *pot,char potfile[]);
void gen_output(double *rho,bool arti);
void show_output(int iteration,bool perturb);
double eigenv_perturb(int l,int n,int l_init);
void do_perturb(int l_init,int l_max);
void den_profile(double* rho,int l_init,int l_max);
void Set_GS_rfunc(double GSrfunc[]);
double pot_diff(void);
void rho_init(double* rho);
int Global_l_max(int l_max);

extern double *r    ;
extern double *r_1  ;
extern double *pot  ;
extern double *pot_l;
extern double *rho  ;
//set pot_arti as external variable 2020.12.14
extern double *pot_arti;
//

// tunable parameters
const int show_eigenvalues  = 1;
const int show_eigenvectors = 0;
const int show_average_r    = 1;
