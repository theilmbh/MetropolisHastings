#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

/* C implementation of metropolis hastings for evaluating path integrals */

double rand_in_range(double min, double max)
{
	
	return min+(((double)rand() / (double)RAND_MAX )* (max-min));
	
}

void display_progress(int completed, int total, int cmp_mod)
{

	/* clear line */
	/*printf("\33[2K\r");*/
	if(completed % cmp_mod == 0)
	{
		double pct_complete = 100* ((double)completed / (double)total);
		printf("%d of %d (%f%%) Completed.\n", completed, total, pct_complete);
	}

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
	
	printf("Burning in...\n");
	for(burn = 0; burn < n_burn; burn++)
	{
		metropolis_update(path, g, n_burn_sweeps, nt, D);
		display_progress(burn, n_burn, 250);
	}
	return 0;
	
}


double* generate_random_initial_path(int nt, double min, double max)
{

	int i;
	double *path = malloc(nt*sizeof(double));
	
	printf("Generating a random initial path of length %d\n", nt);
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

void test_random(n_rand)
{
	double rand_out;
	for(int i = 0; i<n_rand; i++)
	{
		rand_out = rand_in_range(0.0, 1.0);
		printf("Result of rand_in_range(0.0, 1.0): %f\n", rand_out);
	}
}

int main()
{

	
	int n_rand = 10;
	double g = 1.0;
	int n_burn = 10000;
	double D = 1.5;
	int nt = 1000;
	int n_paths = 10000;
	int n_sweeps = 100;
	double path_min = -10.0;
	double path_max = 10.0;

	srand(time(NULL));
	rand();
	
	/* try to get some sample paths */
	double **paths;
	paths = get_sample_paths(n_paths, nt, n_sweeps, n_burn, g, D);
	/*print_path(paths[n_paths-1], nt);*/
	printf("Done!\n");
	
	return 0;

}