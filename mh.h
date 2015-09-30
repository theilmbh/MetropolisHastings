/* mh.h 
 * Header file for Metropolis-Hastings algorithm
 * Brad Theilman September 2015
 */
 
 #ifndef MH_H
 #define MH_H
 
 /* Macros? */
 
 /* Declarations */
 double metropolis_update(double* path, double g, int n_sweeps, int nt, double D);
 double burn_in(double* path, int n_burn, int nt, double g, double D);
 double** get_sample_paths(int n_paths, int nt, int n_sweeps, int n_burn, double g, double D);
 
 
 
 #endif