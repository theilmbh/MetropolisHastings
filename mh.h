/* mh.h 
 * Header file for Metropolis-Hastings algorithm
 * Brad Theilman September 2015
 */
 
#ifndef MH_H
#define MH_H
 
 
/* Macros? */
 
/* Declarations */

/* metropolis hastings */
double metropolis_update(double* path, double g, int n_sweeps, int nt, double D);
double burn_in(double* path, int n_burn, int nt, double g, double D);
double** get_sample_paths(int n_paths, int nt, int n_sweeps, int n_burn, double g, double D);
 
/* mhmath */
double rand_in_range(double min, double max);
double* generate_random_initial_path(int nt, double min, double max);
 
/* mhio */
int write_to_csv(char* fname, double* data, int n_data);
int write_paths_to_csv(char* fname, double** paths, int n_paths, int nt);
int print_path(double* path, int nt);
void display_progress(int completed, int total, int cmp_mod);

/* mhmeasure */
double* compute_correlation(double **paths, int n_paths, int te_0, int tau_eps_max);

/* lagrangian */
double S_euclidean(double *path, double g, int nt);
double delta_S(double* path, double g, double delta, int path_index);
 
 
 
#endif