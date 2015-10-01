#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#include "mh.h"



/* C implementation of metropolis hastings for evaluating path integrals */


double metropolis_update(double* path, double g, int n_sweeps, int nt, double D)
{

    int sweep, tstep;
    double shift, dS, accept_prob;
    
    /* Return path after performing metropolis hastings updating for n_sweeps sweeps */
    for(sweep = 0; sweep < n_sweeps; sweep++)
    {
        for(tstep = 0; tstep < nt; tstep++)
        {
            /* Get a Random Shift */
            shift = rand_in_range(-1.0*D, D);
            
            /* Compute change in action */
            dS = delta_S(path, g, shift, tstep);
            
            /* Determine acceptance probability */
            accept_prob = fmin(1.0, exp(-1.0*dS));  
            if(rand_in_range(0.0, 1.0) <= accept_prob)
            {
                /* accept the shift */
                path[tstep] += shift;
            }
        }
    }
                
    return 0;
    
}

double burn_in(double* path, int n_burn, int nt, double g, double D)
{

    /* Burn in path for n_burn reps */
    int burn;
    int n_burn_sweeps = 100;
    
    printf("Burning in...\n");
    for(burn = 0; burn < n_burn; burn++)
    {
        metropolis_update(path, g, n_burn_sweeps, nt, D);
        display_progress(burn, n_burn, 250);
    }
    return 0;
    
}


double** get_sample_paths(int n_paths, int nt, int n_sweeps, int n_burn, double g, double D)
{

    /* Use the metropolis hastings algorithm to generate n_paths samples according to the action */
    int i;
    double *paths = calloc(n_paths*nt, sizeof(double));
    double **rows = malloc(n_paths*sizeof(double*));
    double *path = generate_random_initial_path(nt, -5.0, 5.0);
    for(i=0; i<n_paths; ++i)
    {
        rows[i] = paths + i*nt;
    }
    
    burn_in(path, n_burn, nt, g, D);
    
    
    printf("Getting sample paths...\n");
    for(i=0; i<n_paths; i++)
    {
        
        metropolis_update(path, g, n_sweeps, nt, D);
        memcpy(rows[i], path, nt*sizeof(double));
        display_progress(i, n_paths, n_paths / 10);
    }
    
    return rows;
    
}


