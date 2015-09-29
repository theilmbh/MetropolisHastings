#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

/* C implementation of metropolis hastings for evaluating path integrals */

double rand_in_range(double min, double max)
{
	
	return min+(((double)rand() / (double)RAND_MAX )* (max-min));
	
}

/* Action definitions */

double S_euclidean(double *path, double g, int nt)
{

    /* Returns the value of the action for a path */
    int i, j;
    double A = 0;
    double B = 0;
    
    /* np.diagonal(np.dot(paths[:, 1:-1].T, paths[:, 1:-1])) - np.diagonal(g*np.dot(paths[:, 0:-2].T, paths[:, 1:])) */
	for(j = 0; j < nt-1; j++)
    {
    	A += path[j+1]*path[j+1];
    	B += g*path[j]*path[j+1];
    } 
    
    return A+B;

}

double delta_S(double* path, double g, double delta, int path_index)
{

    /* Returns value of change in action due to updating value at path_index by delta */
    return delta*(delta + 2*path[path_index] - g*(path[path_index-1] + path[path_index+1]));

}

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
	for(burn = 0; burn < n_burn; burn++)
	{
		metropolis_update(path, g, n_burn_sweeps, nt, D);
	}
	return 0;
	
}

int get_sample_paths(int n_paths, int nt, int n_burn, double g, double D)
{

	/* TODO: Use the metropolis hastings algorithm to generate n_paths samples according to the action */
	double **paths = malloc(n_paths*sizeof(double*));
	
	double* path = generate_random_initial_path(nt, -5.0, 5.0);
	burn_in(path, n_burn, nt, g, D);
	
	int i;
	double *path_sample
	/*for(i=0, i<=n_paths, i++)*/
	
	
	return 0;
	
}

double* generate_random_initial_path(int nt, double min, double max)
{

	int i;
	double *path = malloc(nt*sizeof(double));
	
	for(i=0; i<nt; i++)
	{
		path[i] = rand_in_range(min, max);
	}
	return path;
	
}

int print_path(double* path, int nt)
{
	/* print path */
	printf("---- CURRENT PATH VALUES ---- \n");
	for(int i = 0; i<nt; i++)
	{
		printf("%f\n", path[i]);
	}
	printf("\n");
	return 0;
}

int main()
{

	/* Test all of the functions */
	double rand_out;
	int n_rand = 10;
	srand(time(NULL));
	rand();
	for(int i = 0; i<n_rand; i++)
	{
		rand_out = rand_in_range(0.0, 1.0);
		printf("Result of rand_in_range(0.0, 1.0): %f\n", rand_out);
	}
	
	/* Generate random initial path */
	int nt = 10;
	double path_min = -10.0;
	double path_max = 10.0;
	double *path = generate_random_initial_path(nt, path_min, path_max);
	print_path(path, nt);
	
	/* burn in path */
	printf("Burning in...\n");
	double g = 1.0;
	int n_burn = 10000;
	double D = 1.5;
	burn_in(path, n_burn, nt, g, D);
	print_path(path, nt);
	
	return 0;

}