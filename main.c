#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#include "mh.h"

/* C implementation of metropolis hastings for evaluating path integrals */
	
int main()
{

    
    int n_rand = 10;
    double eps = 0.01;
    double omega_eps = 0.25;
    double omega = omega_eps / eps;
    double kappa = 0.25*pow(eps, 2)*pow(omega, 2);
    double g = (1 - kappa) / (1 + kappa);
    
    int n_burn = 10000;
    double D = 1.5;
    int nt = 1000;
    int n_paths = 40000;
    int n_sweeps = 100;
    double path_min = -10.0;
    double path_max = 10.0;
    
    int tau_eps_max = 15;
    int te_0 = nt / 2;

    srand(time(NULL));
    rand();
    
    /* try to get some sample paths */
    double **paths;
    paths = get_sample_paths(n_paths, nt, n_sweeps, n_burn, g, D);
    /*print_path(paths[n_paths-1], nt);*/
    printf("Done!\n");
    
    /* compute correlations */
    double* correlations = compute_correlation(paths, n_paths, te_0, tau_eps_max);
    
    char* outfile_name = "correlations.dat";
    write_to_csv(outfile_name, correlations, tau_eps_max);
    
    return 0;

}