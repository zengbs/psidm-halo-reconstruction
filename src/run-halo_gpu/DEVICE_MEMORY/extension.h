/* This extension is for density, mass, force, and potential calculation for both soliton and NFW, with other necessary function, from which other input parameters can be derived. */

//double newton_g_mks;
//double mass_of_sun;
//double kpc_to_meter;
//double electron_charge;
//double speed_of_light;
//double hbar_mks;

double zeta(double);
double r_core(void);
double rho_soliton_r(double);
double M_soliton_r(double);
double force_soliton_r(double);
double phi_soliton_r(double);
double rho_NFW_r(double);
double M_NFW_r(double);
double force_NFW_r(double);
double phi_NFW_r(double);
double rho_total_r(double);
double phi_total_r(double);
double find_re_bisection(double(*)(double), double(*)(double), double, double);
void gen_init_pot_guess(char *);
void gen_init_den_profile(char *);
void gen_final_pot_profile(char *);
void print_out_extra_quantities(int);
double convergence_check();
