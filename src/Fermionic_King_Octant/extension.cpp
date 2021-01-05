/* This extension is for density, mass, force, and potential calculation for both soliton and NFW, with other necessary function, from which other input parameters can be derived. Edited at 2020.12.12*/
/* add error_calculation for convergence check 2020.12.14*/
/* add print_out_extra_physical_quantities 2020.12.23*/
/* add gen_final_pot_profile 2020.12.27 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include "main.h"
#include "solve_eigenvalues.h"

// physical constant in mks unit 
double newton_g_mks = 6.6743e-11;
double mass_of_sun = 1.98892e30;
double kpc_to_meter = 3.08567758128e19;
double electron_charge = 1.602176634e-19;
double speed_of_light = 299792458.0;
double hbar_mks = 1.0545718176461565e-34;
//

double zeta(double z)
{
	double Omega_m = Omega_m0*pow(1.+z,3.)/(Omega_m0*pow(1.+z,3.)+Omega_lambda0);
   	return (18.*pow(M_PI,2.)+82.*(Omega_m-1.)-39.*pow(Omega_m-1.,2.))/Omega_m;
}

double r_core(void)
{
	double r_c = 1.6/m_a_22*sqrt(1./(1.+red_shift))*pow(zeta(red_shift)/zeta(0.),-1./6.)*pow(halomass*100.,-1./3.);
//        printf("%.6e\t%.6e\t%.6e\n", r_c,h0,mass_core_ratio);
	return r_c*1e-3*h0/mass_core_ratio; // change the unit from kpc to Mpc/h
}

double rho_soliton_r(double x)
{
    x /= r_c;
    double factor = pow(pow(2.,1./8.)-1.,0.5);
    return 1.9e9/scale_fac/pow(m_a_22*10,2.)/(r_c*1000./h0)/pow(1.+pow(factor*x,2.),8.)/1e11;  // in unit of 1e11M_sun/r_c^3, where r_c in unit of kpc
}

double M_soliton_r(double x)
{
	x /= r_c;
	double factor = pow(pow(2.,1./8.)-1.,0.5);
    	double M_s_at_r = 4.077703890131877e6/scale_fac/pow(m_a_22*10,2.)/(r_c*1000./h0)/pow(1.+pow(factor*x,2),7)*(3465*pow(factor*x,13.)+23100*pow(factor*x,11.)+65373*pow(factor*x,9.)+101376*pow(factor*x,7.)+92323*pow(factor*x,5.)+48580*pow(factor*x,3.)-3465*(factor*x)+3465*pow(pow(factor*x,2)+1.,7.)*atan(factor*x));  // in unit of M_sun; r_c is converted to kpc for this calculation
	return M_s_at_r/1e11; // convert to 1e11 M_sun
}

double force_soliton_r(double x)
{
        return -newton_g_mks*M_soliton_r(x)/pow(x/r_c, 2.); // in unit of 1e11 GM_sun/r_c^2
}

double phi_soliton_r(double x)
{
        x /= r_c;
	double factor = pow(pow(2.,1./8.)-1.,0.5);
  	double X = factor*x;
	return  newton_g_mks*potential_scale*4.077703890131877e6/scale_fac/pow(m_a_22*10,2.)/(r_c*1000./h0)*(factor*(-1732.5/(X*X+1.)-6641.25/pow(X*X+1.,2.)+3927/pow(X*X+1.,3.)-5915.25/pow(X*X+1.,4.)+324.5/pow(X*X+1.,5.)-1568.75/pow(X*X+1.,6.)+288.75*pow(X,12.)/pow(X*X+1.,6.)+3465*log(factor))-3465*atan(X)/x)/1e11; // in unit of 1e11 GM_sun/r_c, where r_c in unit of Mpc/h
}

double rho_NFW_r(double x)
{
	double r = x/(r_vir/NFW_C); // R_s=R_{vir}/c
	return NFW_rho_0/(r*pow(1.+r,2.))/1e11; // in unit of 1e11 M_sun/r_c^3, where r_c in unit of kpc
}

double M_NFW_r(double x)
{
        x /= r_c;
    	double r = x/R_s;
    	return 4.*M_PI*NFW_rho_0*pow(R_s,3.)*(log(1.+r)+1./(1.+r)-1.)/1e11; // in unit of 1e11 M_sun
}

double force_NFW_r(double x)
{
        return -newton_g_mks*M_NFW_r(x)/pow(x/r_c, 2.); // in unit of 1e11 GM_sun/r_c^2
}

double phi_NFW_r(double x)
{
        x /= r_c;
    	double r = x/R_s;
    	return -4.*potential_scale*M_PI*newton_g_mks*NFW_rho_0*pow(R_s,3.)*log(1.+r)/x/1e11; // in unit of 1e11 GM_sun/r_c, where r_c in unit of Mpc/h
}

double rho_total_r(double x)
{
	if (x>r_e)
	    return rho_NFW_r(x);
	else
	    return rho_soliton_r(x);
}

double phi_total_r(double x)
{
	double B = potential_scale*(force_NFW_r(r_e)-force_soliton_r(r_e))*pow(r_e/r_c,2.);
        if (x<=r_e)
           return phi_soliton_r(x)-phi_soliton_r(r_e)+phi_NFW_r(r_e)-B/(r_e/r_c);
        else
           return phi_NFW_r(x)-B/(x/r_c);
}

double find_re_bisection(double(*f_1)(double), double(*f_2)(double), double lower_bound, double upper_bound)
{
//	printf("%.6e\t%.6e\t%.6e\t%.6e\n", f_1(lower_bound), f_2(lower_bound), f_1(upper_bound), f_2(upper_bound));
	if ((f_1(lower_bound)-f_2(lower_bound))*(f_1(upper_bound)-f_2(upper_bound))>0)
	{
	    printf("Function return at lower bound and upper bound must be opposite sign! Exit!\n");
	    exit(1);
	}
	else 
	{
            bool flag = true;
	    double dx = (upper_bound-lower_bound)/1e6;
	    double x = lower_bound;
            double temp_1 = (f_1(x)-f_2(x));
	    double temp_2;
            while (x<upper_bound)
            {
		x += dx;
		if (flag)
                    temp_2 = (f_1(x)-f_2(x));
                else 
                    temp_1 = (f_1(x)-f_2(x));
                if (temp_1*temp_2<0.)
                    break;
                flag = !flag;
	    }
            return x-(temp_2-temp_1)/temp_2*dx;  // in unit of Mpc/h
	}
}

void gen_init_pot_guess(char file_name[])
{
    double phi_init;
    FILE *output = fopen(file_name, "wb");
    for (int i=0; i<rsiz; i++)
    {
        phi_init = phi_total_r(dr*(i+1));
	fwrite(&phi_init, sizeof(double), 1, output);
    }
    fclose(output);
}

void gen_init_den_profile(char file_name[])
{
    double rho_init;
    FILE *output = fopen(file_name, "wb");
    for (int i=0; i<rsiz; i++)
    {
        rho_init = rho_total_r(dr*(i+1))/pow(r_c,3.);
	fwrite(&rho_init, sizeof(double), 1, output);
    }
    fclose(output);
}

void gen_final_pot_profile(char file_name[])
{
    FILE *output = fopen(file_name, "wb");
    fwrite(pot, sizeof(double), rsiz, output);
}

double convergence_check(void)
{
    double D = 0.0;
    for (int i=0; i<rsiz; i++)
        D += 2.*dr/r_range*pow((pot[i]-pot_arti[i])/(pot[i]+pot_arti[i]),2.);
    return D;
}

// add print out extra physical quantities 2020.12.23
// radius(in Mpc/h)  Mass(in solar mass)  Mean_Density (in solar mass/(Mpc/h)^3)  velocity(in km/s)
void print_out_extra_quantities(int iter)
{
    char file_name [128];
    sprintf(file_name,"extra_phyiscal_quantities%d.txt", iter);
    FILE* output = fopen(file_name, "w");
    fprintf(output, "#Radius (Mpc/h)\tMass (Solar Mass)\tMean Density (Solar Mass/(Mpc/h)^3)\tVelocity (km/s)\n");
    fprintf(output, "#===========================================================================================\n");
    double M_sum = 0;
    for (int i=0; i<rsiz; i++)
    {
        M_sum += rho[i];
        fprintf(output, "%.16e\t%.16e\t%.16e\t%.16e\n", r[i], M_sum*1.e11, M_sum*1.e11/(4.*M_PI*pow(r[i],3.)/3.), 1e-3*pow(M_sum*1e11*mass_of_sun*newton_g_mks/(r[i]/h0*1e3*kpc_to_meter),0.5) );
    }
    fclose(output);
}
